# export_clean_text.py
import sys
import numpy as np
import torch
from transformers import AutoProcessor, BarkModel

device = "cuda" if torch.cuda.is_available() else "cpu"
text = sys.argv[1] if len(sys.argv) > 1 else "Hello. I am a machine. I do not have emotions."

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)

inputs = processor(text, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

# Hook into fine_acoustics or codec_decode to grab the exact EnCodec tokens
captured_tokens = []

def hook_fn(module, args, kwargs):
    # args[0] is coarse_output; the output of this forward/generate is fine_tokens
    pass

# We hook codec_decode because it directly takes the fine tokens [batch, 8, time]
original_codec_decode = model.codec_decode
def intercepted_codec_decode(fine_output, output_lengths=None):
    captured_tokens.append(fine_output.detach().cpu())
    return original_codec_decode(fine_output, output_lengths)

model.codec_decode = intercepted_codec_decode

print(f"Generating for: \"{text}\"")
with torch.no_grad():
    _ = model.generate(**inputs, min_eos_p=0.05)

fine_tokens = captured_tokens[0] # [1, 8, T]
tokens_np = fine_tokens.squeeze(0).transpose(0, 1).numpy().astype(np.uint16)
num_frames, num_stages = tokens_np.shape
print(f"Captured {num_frames} frames ({num_stages} stages). Duration: {num_frames / 75.0:.2f}s")

output_path = "tokens_bark.bin"
tokens_np.tofile(output_path)
print(f"Saved {tokens_np.nbytes} bytes to {output_path}")
