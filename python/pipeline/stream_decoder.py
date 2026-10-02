import json
import random
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000
MAGIC_NUMBER = 0xAA55
HEADER_FMT = "<HBBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "responses"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"


def load_wav_mono_f32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def load_response_assets() -> list[dict]:
    """Loads every response WAV from manifest.json into memory up front.

    No Bark/EnCodec inference happens at runtime any more: this is a pure
    token-router that just picks a pre-baked waveform and plays it.
    """
    if not MANIFEST_PATH.exists():
        sys.stderr.write(
            f"[flatline-decoder] {MANIFEST_PATH} not found. Run "
            "python/experiments/build_response_assets.py first.\n"
        )
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    if manifest.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
        raise ValueError(
            f"manifest sample_rate {manifest.get('sample_rate')} != expected {SAMPLE_RATE}"
        )

    assets = []
    for entry in manifest["responses"]:
        audio = load_wav_mono_f32(ASSETS_DIR / entry["wav"])
        assets.append({**entry, "audio": audio})
    return assets


sys.stderr.write(f"[flatline-decoder] Loading response assets from {ASSETS_DIR}...\n")
RESPONSE_ASSETS = load_response_assets()
sys.stderr.write(f"[flatline-decoder] Loaded {len(RESPONSE_ASSETS)} response assets.\n")

sys.stderr.write("[flatline-decoder] Ready. Listening for committed utterances...\n")


def pick_response(category: str | None = None) -> dict:
    candidates = RESPONSE_ASSETS
    if category is not None:
        candidates = [a for a in RESPONSE_ASSETS if a["category"] == category]
        if not candidates:
            raise ValueError(f"no response assets in category {category!r}")
    return random.choice(candidates)


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

        # The committed utterance's raw PCM is only a trigger here: its content is
        # discarded. We are not transcribing it (yet) - see docs/architecture.md.
        payload = read_exact(payload_len)
        if payload is None:
            break

        commit_time = time.perf_counter()

        response = pick_response()
        sys.stderr.write(
            f'[flatline-decoder] Utterance committed. Routing to "{response["text"]}" '
            f'({response["category"]}:{response["id"]})\n'
        )

        sd.play(response["audio"], samplerate=SAMPLE_RATE)
        latency_ms = (time.perf_counter() - commit_time) * 1000.0
        sys.stderr.write(
            f"[flatline-decoder] Playing response ({response['duration_s']:.2f}s). "
            f"Commit-to-playback latency: {latency_ms:.1f}ms\n"
        )
        sd.wait()

except KeyboardInterrupt:
    sys.stderr.write("\nDecoder stopped.\n")
