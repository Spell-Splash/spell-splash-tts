from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
import numpy as np
import soundfile as sf
from transformers import SpeechT5HifiGan
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load processor from base model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load your fine-tuned model weights
model = SpeechT5ForTextToSpeech.from_pretrained("speechT5_finetuned_ljspeech")

# Dummy speaker embedding (512-dim vector)
speaker_embeddings = torch.tensor(np.zeros((1, 512), dtype=np.float32))


# Text input
text = "Good Morning, how are you doing today? I hope you are having a great day!"

# Tokenize text
inputs = processor(text=text, return_tensors="pt")

# Generate waveform
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings,
        vocoder=None
    )

# --------------------------
# 1. Visualize Mel-Spectrogram (without vocoder)
# --------------------------

# The mel spectrogram was returned when using vocoder=None
# Let's visualize it
mel = speech.squeeze().cpu().numpy()  # shape: [T, 80]

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel.T, sr=16000, hop_length=256, x_axis='time', y_axis='mel')
plt.colorbar(format="%+2.f dB")
plt.title("Mel-Spectrogram (Before Vocoder)")
plt.tight_layout()
plt.savefig("visual/mel_spectrogram.png")
plt.show()


print(speech.shape, speech.dtype)

if speech.ndim > 1:
    speech = speech.squeeze()

# Save to file
sf.write("output_before.wav", speech.numpy(), 16000)
print("Generated audio saved as output_before.wav")


vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
vocoder.eval()

with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings,
        vocoder=vocoder
    )

sf.write("output_after.wav", speech.numpy(), 16000)
print("Generated audio saved as output_after.wav")

# --------------------------
# 2. Visualize Final Waveform (with vocoder)
# --------------------------

# Plot the waveform saved to output_after.wav
waveform, sr = sf.read("output_after.wav")

plt.figure(figsize=(10, 3))
plt.plot(np.linspace(0, len(waveform)/sr, len(waveform)), waveform)
plt.title("Generated Waveform (After Vocoder)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("visual/waveform.png")
plt.show()

print(speech.shape, speech.dtype)