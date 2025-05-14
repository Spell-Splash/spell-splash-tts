from kokoro import KPipeline
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# ----------------------
# 1. Setup
# ----------------------

word = "Cryptography"
output_audio_path = f"audio/{word}_tts.wav"
waveform_plot_path = f"visual/{word}_waveform.png"
spectrogram_plot_path = f"visual/{word}_mel_spectrogram.png"

os.makedirs("audio", exist_ok=True)
os.makedirs("visual", exist_ok=True)

# ----------------------
# 2. Generate Audio from Kokoro
# ----------------------

pipeline = KPipeline(lang_code='a')

print(f"Generating speech for: {word}")
for _, _, audio_tensor in pipeline(word, voice='af_heart'):
    audio_np = audio_tensor.detach().cpu().numpy()
    audio_np = (audio_np * 32767).astype(np.int16)
    write_wav(output_audio_path, 22000, audio_np)
    print(f"Audio saved to: {output_audio_path}")
    break

# ----------------------
# 3. Load audio using librosa
# ----------------------

y, sr = librosa.load(output_audio_path, sr=None)

# ----------------------
# 4. Plot Waveform
# ----------------------

plt.figure(figsize=(10, 3))
plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(waveform_plot_path)
plt.close()
print(f"Waveform saved to: {waveform_plot_path}")

# ----------------------
# 5. Plot Mel Spectrogram
# ----------------------

mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.savefig(spectrogram_plot_path)
plt.close()
print(f"Mel spectrogram saved to: {spectrogram_plot_path}")
