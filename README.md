# Real-Time Voice Cloning (Development)

This software generates natural-sounding speech to match a user-provided voice sample, in real time.

It implements the three-stage deep learning framework proposed in [1806.04558](https://arxiv.org/pdf/1806.04558.pdf):
1. A speaker encoder which creates a numerical representation of a voice from a few seconds of audio,
2. A modified text-to-speech synthesizer that generates an audio spectrogram in the target voice, and
3. A vocoder that transforms the spectrograms into waveform audio.

<img src="https://user-images.githubusercontent.com/67130644/94659641-73352480-02b9-11eb-9f25-8e3bc09297a3.png" alt="SV2TTS diagram" width="600"/>

## Setup Instructions

This is a developmental version targeted to advanced users. No support is available. If you are new to this software, use this instead: https://github.com/CorentinJ/Real-Time-Voice-Cloning

### Prerequisites

Python 3.6+ is required. A GPU is optional.

### 1. Install requirements 

```
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

### 2. Download pretrained model

https://www.dropbox.com/s/msozsvqfhd31sva/pretrained_pt.zip?dl=1

Extract the files to their corresponding locations in your Real-Time-Voice-Cloning folder.

### 3. Launch toolbox

```
python demo_toolbox.py
```

## Additional Information

| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
