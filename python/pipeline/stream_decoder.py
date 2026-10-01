import random
import struct
import sys
import numpy as np
import sounddevice as sd
import torch
from encodec import EncodecModel
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig,
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
)

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
FLAT_TEMPERATURE = 0.2

# Cold, affectless acknowledgements. One is picked at random per committed utterance.
RESPONSE_PHRASES = [
    "Acknowledged.",
    "Signal received. Stand by.",
    "Input logged.",
    "Processing complete.",
    "State confirmed.",
    "Noted.",
]

sys.stderr.write(f"[flatline-decoder] Loading EnCodec 24kHz on [{device}]...\n")
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(6.0)
codec.to(device)
codec.eval()

sys.stderr.write("[flatline-decoder] Loading suno/bark-small...\n")
BARK_MODEL_ID = "suno/bark-small"
bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_ID)
bark_model = BarkModel.from_pretrained(BARK_MODEL_ID, torch_dtype=torch.float32).to(device)
bark_model.eval()

# Generation configs are built once at startup (not per-utterance) to keep the
# per-request path allocation-free: only inputs/KV-caches change at generate time.
_sem_dict = dict(bark_model.generation_config.semantic_config)
_sem_dict["temperature"] = FLAT_TEMPERATURE
_sem_dict["do_sample"] = True
SEMANTIC_GEN_CONFIG = BarkSemanticGenerationConfig(**_sem_dict)

_coarse_dict = dict(bark_model.generation_config.coarse_acoustics_config)
_coarse_dict["temperature"] = FLAT_TEMPERATURE
_coarse_dict["do_sample"] = True
COARSE_GEN_CONFIG = BarkCoarseGenerationConfig(**_coarse_dict)

FINE_GEN_CONFIG = BarkFineGenerationConfig(**dict(bark_model.generation_config.fine_acoustics_config))
CODEBOOK_SIZE = bark_model.generation_config.codebook_size

sys.stderr.write(f"[flatline-decoder] Ready on [{device}]. Listening for committed utterances...\n")


def generate_flat_response_tokens(text: str) -> torch.Tensor:
    """Text -> flat/monotone EnCodec RVQ tokens via Bark-small (Semantic -> Coarse -> Fine).

    Returns a tensor shaped [1, 8, T], matching EncodecModel.decode's expected layout.
    """
    inputs = bark_processor(text, voice_preset=None, return_tensors="pt").to(device)

    with torch.no_grad():
        semantic_output = bark_model.semantic.generate(
            inputs["input_ids"],
            semantic_generation_config=SEMANTIC_GEN_CONFIG,
        )

        coarse_output = bark_model.coarse_acoustics.generate(
            semantic_output,
            semantic_generation_config=SEMANTIC_GEN_CONFIG,
            coarse_generation_config=COARSE_GEN_CONFIG,
            codebook_size=CODEBOOK_SIZE,
        )

        fine_output = bark_model.fine_acoustics.generate(
            coarse_output,
            semantic_generation_config=SEMANTIC_GEN_CONFIG,
            coarse_generation_config=COARSE_GEN_CONFIG,
            fine_generation_config=FINE_GEN_CONFIG,
            codebook_size=CODEBOOK_SIZE,
            temperature=FLAT_TEMPERATURE,
        )

    return fine_output


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

        text = random.choice(RESPONSE_PHRASES)
        sys.stderr.write(f'[flatline-decoder] Utterance committed. Responding: "{text}"\n')

        tokens = generate_flat_response_tokens(text)

        with torch.no_grad():
            wav = codec.decode([(tokens, None)])
            audio_out = wav.squeeze().cpu().numpy()

        sys.stderr.write(
            f"[flatline-decoder] Playing response ({len(audio_out) / SAMPLE_RATE:.2f}s)...\n"
        )
        sd.play(audio_out, samplerate=SAMPLE_RATE)
        sd.wait()

except KeyboardInterrupt:
    sys.stderr.write("\nDecoder stopped.\n")
