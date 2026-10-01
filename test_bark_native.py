# test_bark_native.py
import torch
import torchaudio
from transformers import AutoProcessor, BarkModel

device = "cuda" if torch.cuda.is_available() else "cpu"
text = "Hello. I am a machine."

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)

# Standard voice preset for stable generation
inputs = processor(text, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)

with torch.no_grad():
    # Use standard sampling parameters (default temperature)
    audio_array = model.generate(**inputs)

audio_tensor = audio_array.cpu()
if audio_tensor.ndim == 1:
    audio_tensor = audio_tensor.unsqueeze(0)

# Normalize audio to prevent clipping
max_val = torch.max(torch.abs(audio_tensor))
if max_val > 0:
    audio_tensor = audio_tensor / max_val * 0.95

torchaudio.save("bark_official.wav", audio_tensor, 24000)
print("Saved stable bark_official.wav")
