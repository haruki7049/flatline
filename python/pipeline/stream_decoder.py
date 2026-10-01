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

sys.stderr.write(f"[flatline-decoder] Ready on [{device}]. Listening for committed utterances...\n")


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

        # Committed utterance as raw PCM (int16, 24kHz, mono)
        pcm_i16 = np.frombuffer(payload, dtype=np.int16)
        audio_f32 = pcm_i16.astype(np.float32) / 32768.0

        audio_tensor = (
            torch.from_numpy(audio_f32).unsqueeze(0).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            # Single continuous encode/decode pass over the whole utterance
            # (no 40ms-chunk re-inference, avoiding boundary artifacts).
            frames = model.encode(audio_tensor)
            wav = model.decode(frames)
            audio_out = wav.squeeze().cpu().numpy()

        sys.stderr.write(
            f"[flatline-decoder] Playing audio ({len(audio_out) / SAMPLE_RATE:.2f}s)...\n"
        )
        sd.play(audio_out, samplerate=SAMPLE_RATE)
        sd.wait()

except KeyboardInterrupt:
    sys.stderr.write("\nDecoder stopped.\n")
