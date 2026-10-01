import sys
import torch
from encodec import EncodecModel

# Initialize EnCodec 24kHz model on ROCm GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

# Generate 1 second of dummy audio (440Hz sine wave, 24kHz sampling rate)
sample_rate = 24000
t = torch.linspace(0, 1.0, sample_rate, device=device)
audio = torch.sin(2 * 3.1415926535 * 440.0 * t).unsqueeze(0).unsqueeze(0)

# Run GPU inference to extract discrete audio tokens
with torch.no_grad():
    frames = model.encode(audio)
    # Extract codebook token tensor: shape (batch, codebooks, timesteps)
    tokens = frames[0][0]

# Flatten tokens, convert to 16-bit signed integers, and write raw bytes to stdout
token_data = tokens.to(dtype=torch.int16, device="cpu").flatten().numpy()
sys.stdout.buffer.write(token_data.tobytes())
sys.stdout.buffer.flush()
