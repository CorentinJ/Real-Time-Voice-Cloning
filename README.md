# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) with a vocoder that works in real-time.

It is a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.

Video demonstration:

[![Toolbox demo](https://i.imgur.com/Ixy13b7.png)](https://youtu.be/7pjOHW3SCcg)



### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |

### Requirements
You will need the following whether you plan to use the toolbox only or to retrain the models.

**Python 3.7**. Python 3.6 might work too, but I wouldn't go lower because I make extensive use of pathlib.

See **requirements.txt** for the list of packages. You will need both Tensorflow (>=1.10, <=1.14) and PyTorch (>=0.4.1).

A GPU is *highly* recommended (CPU-only is currently not implemented), but you don't necessarily need a high tier GPU if you only want to use the toolbox.

### Pretrained models
Downloadable [here](https://drive.google.com/file/d/1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc/view?usp=sharing) (375mb). Merge the contents of the archive with the contents of the repository.

Encoder: trained 1.5M6 steps (2 months with a single GPU) with a batch size of 64
Synthesizer: trained 256k steps (1 week with 4 GPUs) with a batch size of 144
Encoder: trained 428k steps (4 days with a single GPU) with a batch size of 100


## Datasets, toolbox, preprocessing and training
### Datasets
Ideally, you want to keep all your datasets under a same directory. All prepreprocessing scripts will, by default, output the clean data to a new directory  `SV2TTS` created in your datasets root directory. Inside this directory will be created a directory for each model: the encoder, synthesizer and vocoder.

You will need the following datasets:

For the encoder:
- **[LibriSpeech](http://www.openslr.org/12/):** train-other-500 (extract as `LibriSpeech/train-other-500`)
- **[VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html):** Dev A - D as well as the metadata file (extract as `VoxCeleb1/wav` and `VoxCeleb1/vox1_meta.csv`)
- **[VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html):** Dev A - H (extract as `VoxCeleb1/dev`)

For the synthesizer and the vocoder: 
- **[LibriSpeech](http://www.openslr.org/12/):** train-clean-100, train-clean-360 (extract as `LibriSpeech/train-clean-100` and `LibriSpeech/train-clean-360`)
 
Feel free to adapt the code to your needs. Other interesting datasets that you could use include **[VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)** (used in the SV2TTS paper) or **[M-AILABS](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)**.

### Toolbox
Here's the great thing about this repo: you're expected to run all python scripts in their alphabetical order. Begin with

`python toolbox.py <datasets_root>`

to try the toolbox yourself. `datasets_root` is the directory that contains your LibriSpeech, VoxCeleb or other datasets. It is not mandatory to have datasets in your `datasets_root`. **If you only want to try the toolbox, I recommend downloading `LibriSpeech/train-clean-100` (see above) alone.**

### Preprocessing and training
Pass `-h` to get argument infos for any script. If you want to train models yourself, run the remaining scripts:

`python encoder_preprocess.py <datasets_root>`

`python encoder_train.py my_run <datasets_root>`

The encoder uses visdom. You can disable it, but it's nice to have. Here what the environment looks like:

![Visdom](https://i.imgur.com/rB1xk0b.png)

Then you have two separate scripts to generate the data of the synthesizer. This is convenient in case you want to retrain the encoder, you will then have to regenerate embeddings for the synthesizer.

Begin with the audios and the mel spectrograms:

`python synthesizer_preprocess_audio.py <datasets_root>`

Then the embeddings:
 
`python synthesizer_preprocess_embeds.py <datasets_root>/synthesizer`

You can then train the synthesizer:

`python synthesizer_train.py my_run <datasets_root>/synthesizer`

The synthesizer will output generated audios and spectrograms to its model directory when training. Refer to https://github.com/Rayhane-mamah/Tacotron-2 if you need help.

Use the synthesizer to generate training data for the vocoder:

`python vocoder_preprocess.py <datasets_root>`

And finally, train the vocoder:

`python vocoder_preprocess.py <datasets_root>`

The vocoder also outputs ground truth/generated audios to its model directory.
 
## TODO list and planned features
### Implementation
- [ ] Let the user decide if they want to use speaker embeddings or utterance embeddings for training the synthesizer.
- [ ] Multi-GPU training support for the encoder
- [ ] Move on to a pytorch implementation of the synthesizer?
- [ ] Post-generation cleaning routines for the vocoder?

### Toolbox
- [ ] Handle multiple users in the toolbox
- [ ] Allow for saving generated wavs in the toolbox
- [ ] Use the top left space to draw a legend (rather than doing it inside the plot), and give the possibility to remove speakers from there
- [x] Display vocoder generation

### Code style
- [ ] Object-oriented inference rather than the current static style
- [ ] Setup a config file to allow for default argparse values (mainly datasets_root).
- [ ] Change the structure of the hparams file for each model. I think a namespace is the better solution.
- [ ] Properly document all inference functions

