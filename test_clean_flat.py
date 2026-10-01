import torch
import torchaudio
from encodec import EncodecModel

# 1. Load model and audio
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0) # 8 stages

wav, sr = torchaudio.load("my_voice.wav")
if sr != 24000:
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
wav = wav.unsqueeze(0)

# 2. Encode
with torch.no_grad():
    frames = model.encode(wav)
    tokens, scale = frames[0] # tokens: [1, 8, T]

# 3. Cut to first 3 codebooks cleanly (Stage 0, 1, 2)
flat_tokens = tokens[:, :3, :] # [1, 3, T]

# 4. Decode using official decode pipeline with scale preserved
with torch.no_grad():
    # Pass as a valid Encodec frame: list of (tokens, scale)
    out_wav = model.decode([(flat_tokens, scale)]) # [1, 1, T_samples]
    torchaudio.save("clean_flat.wav", out_wav.squeeze(0).cpu(), 24000)

print("Successfully saved cleanly flattened audio to clean_flat.wav")
