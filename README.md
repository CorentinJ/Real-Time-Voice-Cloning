# Real-Time Voice Cloning

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[1811.00002](https://arxiv.org/pdf/1802.08435v1.pdf) | WaveRNN | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E | Generalized End-To-End Loss for Speaker Verification | This repo |

### Related papers 
| URL | Designation | Title |
| --- | ----------- | ----- |
|[1808.10128](https://arxiv.org/pdf/1808.10128.pdf) | SST4TTS | Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis |
|[1710.07654](https://arxiv.org/pdf/1710.07654.pdf) | Deep Voice 3 | Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning |
|[1705.08947](https://arxiv.org/pdf/1705.08947.pdf) | Deep Voice 2 | Deep Voice 2: Multi-Speaker Neural Text-to-Speech |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron | Tacotron: Towards End-To-End Speech Synthesis |
|[1702.07825](https://arxiv.org/pdf/1702.07825.pdf) | Deep Voice 1 | Deep Voice: Real-time Neural Text-to-Speech |
|[1609.03499](https://arxiv.org/pdf/1609.03499.pdf) | Wavenet | Wavenet: A Generative Model for Raw Audio |
|[1506.07503](https://arxiv.org/pdf/1506.07503.pdf) | Attention | Attention-Based Models for Speech Recognition |


### Datasets and preprocessing
Ideally, you want to keep all your datasets under a same directory. All prepreprocessing scripts will, by default, output the clean data to a new directory  `SV2TTS` created in your datasets root directory. Inside this directory will be created a directory for each model: the encoder, synthesizer and vocoder.

You will need the following datasets:

For the encoder:
- **[LibriSpeech](http://www.openslr.org/12/):** train-other-500 (extract as `LibriSpeech/train-other-500`)
- **[VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html):** Dev A - D as well as the metadata file (extract as `VoxCeleb1/wav` and `VoxCeleb1/vox1_meta.csv`)
- **[VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html):** Dev A - H (extract as `VoxCeleb1/dev`)

For the synthesizer and the vocoder: 
- **[LibriSpeech](http://www.openslr.org/12/):** train-clean-100, train-clean-360 (extract as `LibriSpeech/train-clean-100` and `LibriSpeech/train-clean-360`)
 
Feel free to adapt the code to your needs. Other interesting datasets that you could use:
- **[VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)**, used in the SV2TTS paper.
- **[M-AILABS](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)**
 
 

