from vlibs import fileio
from vocoder import inference
import numpy as np
from vocoder.params import model_name
import sounddevice as sd
import torch

model_dir = 'checkpoints'
model_fpath = fileio.join(model_dir, model_name + '.pt')
inference.load_model(model_fpath)
# model = inference._model
# model.a()

target = 8000
overlap = 800
mel = np.load(r"E:\Datasets\Synthesizer\gta\mel-1001-134707-0040_00.npy").T.astype(np.float32)
wav = inference.infer_waveform(mel, normalize=True, target=target, overlap=overlap)
from vocoder.audio import save_wav
save_wav("2target_%d__overlap_%d.wav" % (target, overlap), wav)
sd.play(wav, 16000, blocking=True)

# Replace with dense tensors
# Replace with sparse to see the speedup
# Later, retrain and replace with sparse to see the quality
