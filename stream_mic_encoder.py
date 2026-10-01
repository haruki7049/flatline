import queue
import struct
import sys
import numpy as np
import sounddevice as sd
import torch
from encodec import EncodecModel

# Select optimal acceleration device
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

SAMPLE_RATE = 24000
CHUNK_SAMPLES = 960  # 40ms chunk (320 samples * 3 frames)
MAGIC_NUMBER = 0xAA55

# Initialize EnCodec 24kHz model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        sys.stderr.write(f"Audio buffer status: {status}\n")
    # Queue single channel float32 audio
    audio_q.put(indata[:, 0].copy())


# Start recording stream from microphone
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SAMPLES,
    callback=audio_callback,
)

frame_seq = 0
header_fmt = "<HBB I"

with stream:
    sys.stderr.write(
        f"Microphone stream active on [{device}]. Transmitting frames...\n"
    )
    try:
        while True:
            chunk = audio_q.get()

            # Shape: (batch=1, channels=1, samples=CHUNK_SAMPLES)
            audio_tensor = (
                torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                frames = model.encode(audio_tensor)
                tokens = frames[0][0]

            token_data = (
                tokens.to(dtype=torch.int16, device="cpu")
                .flatten()
                .numpy()
                .tobytes()
            )

            header = struct.pack(
                header_fmt,
                MAGIC_NUMBER,
                1,
                0,
                len(token_data),
            )

            sys.stdout.buffer.write(header + token_data)
            sys.stdout.buffer.flush()

            frame_seq += 1
    except KeyboardInterrupt:
        sys.stderr.write(f"\nStopped. Transmitted {frame_seq} frames.\n")
