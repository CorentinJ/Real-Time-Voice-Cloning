## AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

## Checkout our new project: Unsupervised Speech Decomposition for Rhythm, Pitch, and Timbre Conversion https://github.com/auspicious3000/SpeechSplit

This repository provides a PyTorch implementation of AUTOVC.

AUTOVC is a many-to-many non-parallel voice conversion framework. 

If you find this work useful and use it in your research, please consider citing our paper.

```
@InProceedings{pmlr-v97-qian19c, title = {{A}uto{VC}: Zero-Shot Voice Style Transfer with Only Autoencoder Loss}, author = {Qian, Kaizhi and Zhang, Yang and Chang, Shiyu and Yang, Xuesong and Hasegawa-Johnson, Mark}, pages = {5210--5219}, year = {2019}, editor = {Kamalika Chaudhuri and Ruslan Salakhutdinov}, volume = {97}, series = {Proceedings of Machine Learning Research}, address = {Long Beach, California, USA}, month = {09--15 Jun}, publisher = {PMLR}, pdf = {http://proceedings.mlr.press/v97/qian19c/qian19c.pdf}, url = {http://proceedings.mlr.press/v97/qian19c.html} }
```


### Audio Demo

The audio demo for AUTOVC can be found [here](https://auspicious3000.github.io/autovc-demo/)

### Dependencies
- Python 3
- Numpy
- PyTorch >= v0.4.1
- TensorFlow >= v1.3 (only for tensorboard)
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder

### Pre-trained models

| AUTOVC | Speaker Encoder | WaveNet Vocoder |
|----------------|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV/view?usp=sharing) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Convert Mel-Spectrograms

Download pre-trained AUTOVC model, and run the ```conversion.ipynb``` in the same directory.


### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```vocoder.ipynb``` in the same the directory.

Please note the training metadata and testing metadata have different formats.


### 2.Train model

We have included a small set of training audio files in the wav folder. However, the data is very small and is for code verification purpose only. Please prepare your own dataset for training.

1.Generate spectrogram data from the wav files: ```python make_spect.py```

2.Generate training metadata, including the GE2E speaker embedding (please use one-hot embeddings if you are not doing zero-shot conversion): ```python make_metadata.py```

3.Run the main training script: ```python main.py```

Converges when the reconstruction loss is around 0.0001.



