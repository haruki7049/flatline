import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# 1. Load official EnCodec model (24kHz)
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0) # 8 stages of RVQ (75 frames/sec)

# 2. Generate a simple 1.0 second test tone (440Hz A4)
sample_rate = 24000
duration = 1.0
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = 0.5 * torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0).unsqueeze(0) # [batch, channels, time]

# 3. Encode to discrete tokens
with torch.no_grad():
    frames = model.encode(waveform)
    # frames[0][0] shape: [batch, num_stages(8), num_frames(75)]
    tokens = frames[0][0].squeeze(0) # [8, num_frames]

num_stages, num_frames = tokens.shape
print(f"Encoded {num_frames} frames ({num_stages} stages). Duration: {duration}s")

# 4. Transpose to [num_frames, 8] and save as raw binary (uint16 little-endian)
tokens_t = tokens.transpose(0, 1).contiguous() # [num_frames, 8]
tokens_np = tokens_t.cpu().numpy().astype("uint16")
tokens_np.tofile("tokens.bin")
print(f"Saved {tokens_np.nbytes} bytes to tokens.bin")

# 5. Decode with official model as reference
with torch.no_grad():
    decoded_official = model.decode(frames)
    torchaudio.save("official_output.wav", decoded_official.squeeze(0).cpu(), sample_rate)
    print("Saved reference audio to official_output.wav")
