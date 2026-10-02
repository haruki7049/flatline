import random
import struct
import sys
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, BarkModel

# Select optimal acceleration device
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

SAMPLE_RATE = 24000
MAGIC_NUMBER = 0xAA55
HEADER_FMT = "<HBBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# 0.2 was too low: it pushed Bark's autoregressive coarse/fine stages into mode
# collapse (babbling / a repeated plosive loop instead of words). 0.7 keeps the
# tone calm and monotone without the sampling degenerating like that.
FLAT_TEMPERATURE = 0.7

# Pins the speaker identity so it doesn't drift between utterances (no voice_preset
# samples a new, unpredictable voice/timbre on every call).
VOICE_PRESET = "v2/en_speaker_6"

# The fixed response phrases are all 1-5 words. Left uncapped, Bark's semantic stage
# can run out to ~10s+ of audio regardless of how short the text is.
SEMANTIC_MAX_NEW_TOKENS = 96

# Cold, affectless acknowledgements. One is picked at random per committed utterance.
RESPONSE_PHRASES = [
    "Acknowledged.",
    "Signal received. Stand by.",
    "Input logged.",
    "Processing complete.",
    "State confirmed.",
    "Noted.",
]

sys.stderr.write("[flatline-decoder] Loading suno/bark-small...\n")
BARK_MODEL_ID = "suno/bark-small"
bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float32).to(device)
bark_model.eval()

# Calling bark_model.generate() directly (rather than driving
# semantic/coarse_acoustics/fine_acoustics.generate() separately) lets transformers
# carry EOS detection and attention masks between the three stages correctly, and
# avoids hand-built GenerationConfig objects that triggered deprecation warnings.
# The semantic_*/coarse_*/fine_* prefixed kwargs below are the documented way to
# override each stage's generation settings through this single entry point.
GENERATE_KWARGS = dict(
    semantic_temperature=FLAT_TEMPERATURE,
    coarse_temperature=FLAT_TEMPERATURE,
    fine_temperature=FLAT_TEMPERATURE,
    semantic_max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
)

sys.stderr.write(f"[flatline-decoder] Ready on [{device}]. Listening for committed utterances...\n")


def generate_flat_response_audio(text: str) -> np.ndarray:
    """Text -> flat/monotone PCM audio via Bark-small (EnCodec decode happens inside generate())."""
    inputs = bark_processor(text, voice_preset=VOICE_PRESET, return_tensors="pt").to(device)

    with torch.no_grad():
        audio_array = bark_model.generate(**inputs, **GENERATE_KWARGS)

    return audio_array.cpu().numpy().squeeze()


TRIM_RMS_THRESHOLD = 0.01
TRIM_FRAME_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms


def trim_trailing_silence(audio: np.ndarray) -> np.ndarray:
    """Drops trailing low-RMS frames so a capped-but-still-silent tail isn't played."""
    for end in range(len(audio), 0, -TRIM_FRAME_SAMPLES):
        start = max(0, end - TRIM_FRAME_SAMPLES)
        frame = audio[start:end]
        if frame.size and np.sqrt(np.mean(frame**2)) > TRIM_RMS_THRESHOLD:
            return audio[:end]
    return audio


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


try:
    while True:
        hdr_bytes = read_exact(HEADER_SIZE)
        if hdr_bytes is None:
            break

        magic, version, reserved, payload_len = struct.unpack(
            HEADER_FMT, hdr_bytes
        )
        if magic != MAGIC_NUMBER:
            sys.stderr.write(f"Invalid magic: {hex(magic)}\n")
            break

        # The committed utterance's raw PCM is only a trigger here: its content is
        # discarded. We are not transcribing it (yet) - see docs/architecture.md.
        payload = read_exact(payload_len)
        if payload is None:
            break

        text = random.choice(RESPONSE_PHRASES)
        sys.stderr.write(f'[flatline-decoder] Utterance committed. Responding: "{text}"\n')

        audio_out = trim_trailing_silence(generate_flat_response_audio(text))

        sys.stderr.write(
            f"[flatline-decoder] Playing response ({len(audio_out) / SAMPLE_RATE:.2f}s)...\n"
        )
        sd.play(audio_out, samplerate=SAMPLE_RATE)
        sd.wait()

except KeyboardInterrupt:
    sys.stderr.write("\nDecoder stopped.\n")
