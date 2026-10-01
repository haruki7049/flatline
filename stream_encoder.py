import struct
import sys
import torch
from encodec import EncodecModel

# Select optimal acceleration device
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Initialize EnCodec 24kHz model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

# Generate 1 second of dummy audio (440Hz sine wave, 24kHz sampling rate)
sample_rate = 24000
t = torch.linspace(0, 1.0, sample_rate, device=device)
audio = torch.sin(2 * 3.1415926535 * 440.0 * t).unsqueeze(0).unsqueeze(0)

# Run inference to extract discrete audio tokens
with torch.no_grad():
    frames = model.encode(audio)
    tokens = frames[0][0]

# Flatten tokens and prepare byte payload
token_data = tokens.to(dtype=torch.int16, device="cpu").flatten().numpy().tobytes()

# Frame header: Magic(2B)=0xAA55, Version(1B)=1, Reserved(1B)=0, PayloadLength(4B)
header = struct.pack("<HBB I", 0xAA55, 1, 0, len(token_data))

# Write framed packet to stdout
sys.stdout.buffer.write(header + token_data)
sys.stdout.buffer.flush()
