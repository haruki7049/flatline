"""Pre-bakes flatline's cold, canned responses into EnCodec RVQ token assets.

Replaces the real-time Bark inference in stream_decoder.py with a one-time offline
build: for each fixed phrase, generate audio via Bark-small, capture the exact
EnCodec fine-acoustic tokens that produced it (same hook technique as
export_bark_tokens.py), trim leading/trailing silence at EnCodec-frame boundaries,
and write the result as a token binary + preview WAV under assets/responses/.

At runtime, a token router can then skip generation entirely and just play back
one of these pre-built token sequences through the existing EnCodec decoder
(src/decoder.zig / `hoge --tokens`), which is the low-latency "embedding-driven
token router" this script is meant to feed.

Usage:
    python3 python/experiments/build_response_assets.py
"""

import json
import wave
from pathlib import Path

import numpy as np
import torch
from encodec import EncodecModel
from transformers import AutoProcessor, BarkModel

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "assets" / "responses"

SAMPLE_RATE = 24000
FRAME_RATE_HZ = 75  # EnCodec 24kHz: 24000 / (2*4*5*8 downsampling) = 75 frames/s
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE_HZ  # 320
NUM_CODEBOOKS = 8

VOICE_PRESET = "v2/en_speaker_6"
FLAT_TEMPERATURE = 0.7
SEMANTIC_MAX_NEW_TOKENS = 96
# Lower min_eos_p makes the semantic stage more willing to emit EOS early, which
# keeps these short canned phrases tight instead of padding out with filler.
SEMANTIC_MIN_EOS_P = 0.05

TRIM_RMS_THRESHOLD = 0.01
TRIM_PAD_FRAMES = 1  # keep one frame of margin on each side so onsets aren't clipped

RESPONSES = {
    "ack": ["Acknowledged.", "Confirmed."],
    "status": ["All systems operational.", "Standing by."],
    "reject": ["Negative.", "Command unrecognized."],
    "complete": ["Processing complete.", "Task logged."],
}


def trim_silence_frames(tokens: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Drops leading/trailing EnCodec frames whose decoded audio is near-silent.

    tokens: [T, 8] uint16. audio: the PCM decoded from exactly these tokens, so
    sample-to-frame alignment is exact (frame f covers samples
    [f*SAMPLES_PER_FRAME, (f+1)*SAMPLES_PER_FRAME)).
    """
    num_frames = tokens.shape[0]
    usable_samples = num_frames * SAMPLES_PER_FRAME
    frames = audio[:usable_samples].reshape(num_frames, SAMPLES_PER_FRAME)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    voiced = np.where(rms > TRIM_RMS_THRESHOLD)[0]
    if voiced.size == 0:
        return tokens  # entirely silent; nothing sensible to trim, keep as-is

    start = max(0, int(voiced[0]) - TRIM_PAD_FRAMES)
    end = min(num_frames, int(voiced[-1]) + 1 + TRIM_PAD_FRAMES)
    return tokens[start:end]


def write_wav_mono16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    print("Loading suno/bark-small...")
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    bark_model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float32).to(device)
    bark_model.eval()

    print("Loading EnCodec 24kHz...")
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(6.0)
    codec.to(device)
    codec.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "frame_rate_hz": FRAME_RATE_HZ,
        "codebooks": NUM_CODEBOOKS,
        "responses": [],
    }

    for category, phrases in RESPONSES.items():
        for idx, text in enumerate(phrases):
            print(f'[{category}:{idx}] Generating "{text}"...')

            inputs = processor(text, voice_preset=VOICE_PRESET, return_tensors="pt").to(device)

            # Hook codec_decode to capture the exact fine-acoustic (EnCodec) tokens
            # that generate() is about to decode, without disturbing its output.
            captured = []
            original_codec_decode = bark_model.codec_decode

            def intercepting_codec_decode(fine_output, output_lengths=None):
                captured.append(fine_output.detach())
                return original_codec_decode(fine_output, output_lengths)

            bark_model.codec_decode = intercepting_codec_decode
            try:
                with torch.no_grad():
                    bark_model.generate(
                        **inputs,
                        semantic_temperature=FLAT_TEMPERATURE,
                        coarse_temperature=FLAT_TEMPERATURE,
                        fine_temperature=FLAT_TEMPERATURE,
                        semantic_max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
                        min_eos_p=SEMANTIC_MIN_EOS_P,
                    )
            finally:
                bark_model.codec_decode = original_codec_decode

            fine_tokens = captured[0]  # [1, 8, T]
            tokens_np = fine_tokens.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.uint16)

            with torch.no_grad():
                full_audio = codec.decode([(fine_tokens, None)]).squeeze().cpu().numpy()

            trimmed_tokens = trim_silence_frames(tokens_np, full_audio)

            # Re-decode the trimmed token slice (rather than slicing full_audio) so
            # the preview WAV matches what a fresh decode of the .bin will sound
            # like at runtime (e.g. via `hoge --tokens`).
            trimmed_tensor = (
                torch.from_numpy(trimmed_tokens.astype(np.int64))
                .transpose(0, 1)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                trimmed_audio = codec.decode([(trimmed_tensor, None)]).squeeze().cpu().numpy()

            bin_name = f"{category}_{idx}.bin"
            wav_name = f"{category}_{idx}.wav"
            trimmed_tokens.tofile(OUTPUT_DIR / bin_name)
            write_wav_mono16(OUTPUT_DIR / wav_name, trimmed_audio, SAMPLE_RATE)

            num_frames = int(trimmed_tokens.shape[0])
            duration_s = num_frames / FRAME_RATE_HZ
            manifest["responses"].append(
                {
                    "category": category,
                    "id": idx,
                    "text": text,
                    "bin": bin_name,
                    "wav": wav_name,
                    "num_frames": num_frames,
                    "duration_s": round(duration_s, 3),
                }
            )
            print(f"  -> {num_frames} frames ({duration_s:.2f}s)")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest['responses'])} response assets to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
