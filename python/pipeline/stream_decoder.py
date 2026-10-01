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
MAGIC_NUMBER = 0xAA55
HEADER_FMT = "<HBBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Initialize EnCodec 24kHz model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(device)
model.eval()

sys.stderr.write(f"[flatline-decoder] Ready on [{device}]. Listening for committed tokens...\n")


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

        payload = read_exact(payload_len)
        if payload is None:
            break

        # Convert bytes back to int16 tokens
        tokens_np = np.frombuffer(payload, dtype=np.int16)

        num_codebooks = 8
        total_tokens = len(tokens_np)
        if total_tokens % num_codebooks != 0:
            sys.stderr.write(f"Corrupted frame size: {total_tokens}\n")
            continue

        num_timesteps = total_tokens // num_codebooks

        # Reshape to (timesteps, codebooks) and transpose back to (1, codebooks, timesteps)
        tokens_tensor = (
            torch.from_numpy(tokens_np.astype(np.int64))
            .reshape(num_timesteps, num_codebooks)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            # Decode discrete audio tokens back to 24kHz PCM
            wav = model.decode([(tokens_tensor, None)])
            audio_out = wav.squeeze().cpu().numpy()

        sys.stderr.write(
            f"[flatline-decoder] Playing audio ({len(audio_out) / SAMPLE_RATE:.2f}s)...\n"
        )
        sd.play(audio_out, samplerate=SAMPLE_RATE)
        sd.wait()

except KeyboardInterrupt:
    sys.stderr.write("\nDecoder stopped.\n")
