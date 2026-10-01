import sys
import os
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

def main():
    sample_rate = 24000
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0) # 8 stages (75 fps)

    # 1. Load input audio or generate a synthetic voice tone
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        input_path = sys.argv[1]
        print(f"Loading input audio from: {input_path}")
        wav, sr = torchaudio.load(input_path)
        wav = convert_audio(wav, sr, sample_rate, 1) # convert to mono 24kHz
    else:
        print("No input audio specified. Generating test voice harmonic series...")
        # Synthesize a human-voice-like formant tone (vowel 'a' approximate: ~120Hz fundamental + harmonics)
        duration = 2.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        f0 = 125.0 # fundamental pitch
        wav = (
            0.4 * torch.sin(2 * torch.pi * f0 * t) +
            0.3 * torch.sin(2 * torch.pi * 2 * f0 * t) +
            0.2 * torch.sin(2 * torch.pi * 4 * f0 * t) +
            0.1 * torch.sin(2 * torch.pi * 6 * f0 * t)
        ).unsqueeze(0)

    # Ensure shape is [1, 1, T]
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    duration_sec = wav.shape[-1] / sample_rate
    print(f"Audio duration: {duration_sec:.2f} seconds ({wav.shape[-1]} samples)")

    # 2. Encode with official EnCodec
    with torch.no_grad():
        frames = model.encode(wav)
        # Concatenate frames along time dimension if multi-chunk
        tokens_list = [f[0].squeeze(0) for f in frames] # each [8, T_chunk]
        tokens = torch.cat(tokens_list, dim=-1) # [8, T_total]

    num_stages, num_frames = tokens.shape
    print(f"Encoded into {num_frames} frames ({num_stages} codebook stages)")

    # 3. Save as tokens_speech.bin (uint16 little-endian, [num_frames, 8])
    tokens_t = tokens.transpose(0, 1).contiguous()
    tokens_np = tokens_t.cpu().numpy().astype("uint16")
    tokens_np.tofile("tokens_speech.bin")
    print(f"Saved {tokens_np.nbytes} bytes to tokens_speech.bin")

    # 4. Save PyTorch reference output
    with torch.no_grad():
        decoded_ref = model.decode(frames)
        torchaudio.save("ref_speech.wav", decoded_ref.squeeze(0).cpu(), sample_rate)
        print("Saved reference audio to ref_speech.wav")

if __name__ == "__main__":
    main()
