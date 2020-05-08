# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. Feel free to check [my thesis](https://matheo.uliege.be/handle/2268.2/6801) if you're curious, or if you're looking for info I haven't documented yet. Mostly I would recommend giving a quick look to the figures beyond the introduction.

SV2TTS is a three-stage deep learning framework that allows the creation of a numerical representation of a voice from a few seconds of audio, then use that data to condition a text-to-speech model trained to generate new voices.

**Video demonstration** (click the play button):
[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |


## Get Started
### Requirements
Please use the setup.sh or setup.bat if you're on linux and windows respectively to install the dependancies, and requirements. Currently only python 3.7.x is supported.

* Windows Install Requirements
    * During python installation, make sure python is added to path during installation.
    * During conda installation, make sure you install it 'just for me'.
    * During ms build tools installation, you only need to install the c++ package, which requires around 4.7GB. Upon installation of build tools, you'll need to restart the computer to complete the install process. Rerun the setup.bat to finish the setup process.

#### Install Manually:
You will need [PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1) installed first, then run `pip install -r requirements.txt` to install the necessary packages.

### After install Steps
Next you will need [pretrained models](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models) if you don't plan to train your own.
These models were trained on a cuda device, so they'll produce finicky results for a cpu. New CPU models will need to be produced first. (As of 5/1/20)
Download the models, and uncompress them in this root folder. If done correctly, it should result as `/encoder/saved_models`, `/synthesizer/saved_models`, and `/vocoder/saved_models`.

### Test installation
When you believe you have all the neccesary soup, test the program by running `python demo_cli.py`.
If all tests pass, you're good to go. To use the cpu, use the option `--cpu`.

### Generate Audio from dataset
There are a few preconfigured options for datasets. One in perticular, [`LibriSpeech/train-clean-100`](http://www.openslr.org/resources/12/train-clean-100.tar.gz) is made to work from demo_toolbox.py. When you download this dataset, you can locate the directory anywhere, but creating a folder in this directory named `datasets` is recommended. (All scripts will use this directory as default)

To run the toolbox, use `python demo_toolbox.py` if you followed the recommendation for the datasets directory location. Otherwise, include the full path to the dataset and use the option `-d`.

To set the speaker, you'll need an input audio file. use browse in the toolbox to your personal audio file, or record to set your own voice.

The toolbox supports other datasets, including [dev-train](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets).

If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).

## Contributions & Issues



## Original Author CorentinJ News
**13/11/19**: I'm sorry that I can't maintain this repo as much as I wish I could. I'm working full time as of June 2019 on improving voice cloning techniques and I don't have the time to share my improvements here. Plus this repo relies on a lot of old tensorflow code and it's hard to work with. If you're a researcher, then this repo might be of use to you. **If you just want to clone your voice**, do check our demo on [Resemble.AI](https://www.resemble.ai/) - it will give much better results than this repo and will not require a complex setup.

**20/08/19:** I'm working on [resemblyzer](https://github.com/resemble-ai/Resemblyzer), an independent package for the voice encoder. You can use your trained encoder models from this repo with it.

**06/07/19:** Need to run within a docker container on a remote server? See [here](https://sean.lane.sh/posts/2019/07/Running-the-Real-Time-Voice-Cloning-project-in-Docker/).

**25/06/19:** Experimental support for low-memory GPUs (~2gb) added for the synthesizer. Pass `--low_mem` to `demo_cli.py` or `demo_toolbox.py` to enable it. It adds a big overhead, so it's not recommended if you have enough VRAM.