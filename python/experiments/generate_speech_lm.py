import sys
import numpy as np
import torch
from transformers import AutoProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkSemanticGenerationConfig,
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
)

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello. I am a machine. I do not have emotions."
    print(f"Generating flat tokens for text: \"{text}\"")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "suno/bark-small"
    print(f"Loading processor and model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = BarkModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    # 1. Tokenize input text
    inputs = processor(text, voice_preset=None, return_tensors="pt").to(device)

    # 2. Setup generation configs with low temperatures for emotionless flat tone
    sem_dict = dict(model.generation_config.semantic_config)
    sem_dict["temperature"] = 0.2
    sem_dict["do_sample"] = True
    semantic_gen_config = BarkSemanticGenerationConfig(**sem_dict)

    coarse_dict = dict(model.generation_config.coarse_acoustics_config)
    coarse_dict["temperature"] = 0.2
    coarse_dict["do_sample"] = True
    coarse_gen_config = BarkCoarseGenerationConfig(**coarse_dict)

    fine_dict = dict(model.generation_config.fine_acoustics_config)
    fine_gen_config = BarkFineGenerationConfig(**fine_dict)

    print("Generating speech tokens step-by-step (Semantic -> Coarse -> Fine)...")
    with torch.no_grad():
        # Step A: Text -> Semantic tokens
        semantic_output = model.semantic.generate(
            inputs["input_ids"],
            semantic_generation_config=semantic_gen_config,
        )

        # Step B: Semantic tokens -> Coarse acoustics
        coarse_output = model.coarse_acoustics.generate(
            semantic_output,
            semantic_generation_config=semantic_gen_config,
            coarse_generation_config=coarse_gen_config,
            codebook_size=model.generation_config.codebook_size,
        )

        # Step C: Coarse acoustics -> Fine acoustics (EnCodec 8-stage tokens)
        fine_output = model.fine_acoustics.generate(
            coarse_output,
            semantic_generation_config=semantic_gen_config,
            coarse_generation_config=coarse_gen_config,
            fine_generation_config=fine_gen_config,
            codebook_size=model.generation_config.codebook_size,
            temperature=0.2,
        )

    # fine_output shape: [1, 8, T] -> transpose to [T, 8]
    tokens_np = fine_output.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.uint16)
    num_frames, num_stages = tokens_np.shape
    print(f"Generated {num_frames} frames ({num_stages} stages). Duration: {num_frames / 75.0:.2f}s")

    output_path = "tokens_lm.bin"
    tokens_np.tofile(output_path)
    print(f"Saved {tokens_np.nbytes} bytes to {output_path}")

if __name__ == "__main__":
    main()
