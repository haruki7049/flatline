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
CHUNK_SAMPLES = 960  # 40ms chunk
MAGIC_NUMBER = 0xAA55
RMS_THRESHOLD = 0.015  # Speech energy threshold

# Initialize EnCodec 24kHz model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        sys.stderr.write(f"Audio buffer status: {status}\n")
    audio_q.put(indata[:, 0].copy())


stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SAMPLES,
    callback=audio_callback,
)

frame_seq = 0
header_fmt = "<HBBI"

with stream, torch.no_grad():
    sys.stderr.write(
        f"Microphone stream active on [{device}]. Transmitting frames...\n"
    )
    try:
        while True:
            chunk = audio_q.get()

            # Calculate RMS energy for reliable speech boundary detection
            rms = np.sqrt(np.mean(chunk**2))
            is_speech = 1 if rms > RMS_THRESHOLD else 0

            audio_tensor = (
                torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            )

            frames = model.encode(audio_tensor)
            tokens = frames[0][0][0]  # Shape: (codebooks=8, timesteps=3)

            # Transpose to (timesteps=3, codebooks=8)
            tokens_t = tokens.transpose(0, 1).contiguous()
            token_data = (
                tokens_t.to(dtype=torch.int16, device="cpu")
                .flatten()
                .numpy()
                .tobytes()
            )

            # reserved field carries is_speech flag (0 = silence, 1 = speech)
            header = struct.pack(
                header_fmt,
                MAGIC_NUMBER,
                1,
                is_speech,
                len(token_data),
            )

            sys.stdout.buffer.write(header + token_data)
            sys.stdout.buffer.flush()

            frame_seq += 1
    except KeyboardInterrupt:
        sys.stderr.write(f"\nStopped. Transmitted {frame_seq} frames.\n")

