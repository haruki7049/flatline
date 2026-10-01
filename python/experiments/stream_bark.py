# stream_bark.py
import sys
import numpy as np
import torch
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig,
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
text = sys.argv[1] if len(sys.argv) > 1 else "Hello. I am a machine. I do not have emotions."

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)

inputs = processor(text, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

print(f"Generating for: \"{text}\"", file=sys.stderr)

chunk_frames = 16 # Send in batches of 16 frames (~213ms)

# Extract sub-configs
semantic_config = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
coarse_config = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
fine_config = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

with torch.no_grad():
    # 1. Semantic Generation (Text -> Semantic Tokens)
    semantic_output = model.semantic.generate(
        inputs["input_ids"],
        history_prompt=inputs.get("history_prompt", None),
        semantic_generation_config=semantic_config,
        min_eos_p=0.05,
    )

    # 2. Coarse Generation (Semantic Tokens -> Stage 1 & 2 EnCodec tokens)
    coarse_output = model.coarse_acoustics.generate(
        semantic_output,
        history_prompt=inputs.get("history_prompt", None),
        semantic_generation_config=semantic_config,
        coarse_generation_config=coarse_config,
        codebook_size=model.generation_config.codebook_size,
    )

    # 3. Fine Generation (Stage 1 & 2 -> 8 Stages EnCodec tokens)
    # Generates full [batch, 8, seq_len] tensor
    fine_tokens = model.fine_acoustics.generate(
        coarse_output,
        history_prompt=inputs.get("history_prompt", None),
        semantic_generation_config=semantic_config,
        coarse_generation_config=coarse_config,
        fine_generation_config=fine_config,
        codebook_size=model.generation_config.codebook_size,
    )

    # fine_tokens shape: [1, 8, time] -> transpose to [time, 8]
    tokens_np = fine_tokens.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.uint16)
    total_frames = tokens_np.shape[0]

    # 4. Stream to stdout in chunks
    for start_idx in range(0, total_frames, chunk_frames):
        end_idx = min(start_idx + chunk_frames, total_frames)
        chunk = tokens_np[start_idx:end_idx]
        sys.stdout.buffer.write(chunk.tobytes())
        sys.stdout.buffer.flush()

print("Streaming generation finished.", file=sys.stderr)
