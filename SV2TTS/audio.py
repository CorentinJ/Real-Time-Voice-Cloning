import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

fpath = r"E:\Datasets\LibriSpeech\train-other-500\20\205\20-205-0000.flac"
signal, sampling_rate = librosa.load(fpath, sr=16000)
signal = signal[:int(sampling_rate * 3.5)]
plt.plot(signal)
plt.show()

frames = librosa.feature.melspectrogram(signal, 
                                        sampling_rate,
                                        n_fft=int(sampling_rate * 0.025),
                                        hop_length=int(sampling_rate * 0.01),
                                        n_mels=40)

librosa.display.specshow(
    librosa.power_to_db(frames, ref=np.max),
    hop_length=int(sampling_rate * 0.01),
    y_axis='mel',
    x_axis='time',
    sr=sampling_rate
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()