# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. This was my [master's thesis](https://matheo.uliege.be/handle/2268.2/6801).

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

**Video demonstration** (click the picture):

[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |

## News
**28/12/21**: I've done a [major maintenance update](https://github.com/CorentinJ/Real-Time-Voice-Cloning/pull/961). Mostly, I've worked on making setup easier. Find new instructions in the section below.

**14/02/21**: This repo now runs on PyTorch instead of Tensorflow, thanks to the help of @bluefish. If you wish to run the tensorflow version instead, checkout commit `5425557`.

**13/11/19**: I'm now working full time and I will not maintain this repo anymore. To anyone who reads this:
- **If you just want to clone your voice (and not someone else's):** I recommend our free plan on [Resemble.AI](https://www.resemble.ai/). You will get a better voice quality and less prosody errors.
- **If this is not your case:** proceed with this repository, but you might end up being disappointed by the results. If you're planning to work on a serious project, my strong advice: find another TTS repo. Go [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/364) for more info.

**20/08/19:** I'm working on [resemblyzer](https://github.com/resemble-ai/Resemblyzer), an independent package for the voice encoder (inference only). You can use your trained encoder models from this repo with it.


## Setup

#### 1. Install Requirements
1. Both Windows and Linux and supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 or greater is recommended. Python 3.6 should work, python 3.5 might need some tweaking. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 4. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

### 5. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  

depending on whether you downloaded any datasets. If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).
