import numpy as np
import sys

# 1. Load original tokens
input_path = "tokens_speech.bin"
if len(sys.argv) > 1:
    input_path = sys.argv[1]

raw_data = np.fromfile(input_path, dtype=np.uint16)
num_frames = len(raw_data) // 8
tokens = raw_data.reshape((num_frames, 8)).copy()

print(f"Loaded {num_frames} frames from {input_path}")

# 2. Flatten emotion/prosody:
# - Keep Stage 0 and 1 (phonemes/intelligibility)
# - Clamp or zero out higher stages to remove expressive pitch variance
# For EnCodec, setting higher stages to a fixed neutral centroid removes variance
for stage in range(2, 8):
    tokens[:, stage] = 0 # zero-codebook (neutral base)

# 3. Save flattened tokens
output_path = "tokens_robotic.bin"
tokens.tofile(output_path)
print(f"Saved robotic/emotionless tokens to {output_path}")
