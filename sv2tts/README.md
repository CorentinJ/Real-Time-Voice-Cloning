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
 
 
### Known issues
- There is no noise removal algorithm implemented to clean the data of the synthesizer and the vocoder. I've found the noise removal algorithm from Audacity which uses Fourier analysis to be quite good, but it's too much work to reimplement.
- The hyperparameters for both the encoder and the vocoder do not appear as arguments. I have to think about how I want to do this. 
- I've tried filtering the non-English speakers out of VoxCeleb1 using the metadata file. However, there is no such file for VoxCeleb2. Right now, the non-English speakers of VoxCeleb2 are unfiltered (hopefully, they're still a minority in the dataset). It's hard to tell if this really has a negative impact on the model.
- The sampling rate is forced to be the same between the encoder and the synthesizer/vocoder if you use the toolbox.


### TODO
- I'd like to eventually merge the `audio` module of each package to `utils/audio`.
- I'm looking for a Tacotron framework that is pytorch-based, but that is as good as the tensorflow implementation of Rayhane Mamah
- The toolbox can always be improved
