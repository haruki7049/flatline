import queue
import struct
import sys
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000
CHUNK_SAMPLES = 960  # 40ms chunk
MAGIC_NUMBER = 0xAA55
RMS_THRESHOLD = 0.015  # Speech energy threshold

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

with stream:
    sys.stderr.write(
        f"Microphone stream active. Transmitting raw PCM frames at {SAMPLE_RATE}Hz...\n"
    )
    try:
        while True:
            chunk = audio_q.get()

            # Calculate RMS energy for reliable speech boundary detection
            rms = np.sqrt(np.mean(chunk**2))
            is_speech = 1 if rms > RMS_THRESHOLD else 0

            # Raw PCM payload: int16 little-endian, no EnCodec inference here.
            pcm_i16 = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)
            payload = pcm_i16.tobytes()

            # reserved field carries the is_speech flag (0 = silence, 1 = speech)
            header = struct.pack(
                header_fmt,
                MAGIC_NUMBER,
                1,
                is_speech,
                len(payload),
            )

            sys.stdout.buffer.write(header + payload)
            sys.stdout.buffer.flush()

            frame_seq += 1
    except KeyboardInterrupt:
        sys.stderr.write(f"\nStopped. Transmitted {frame_seq} frames.\n")
