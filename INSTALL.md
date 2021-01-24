# Setup

## Windows (Partially tested)

 ### 1. Prepare prerequisites

 **Python 3.6+** is needed to run the toolbox.

 * Install [PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1).
 * Install [ffmpeg](https://ffmpeg.org/download.html#get-packages).
 * (Highly recommended, as RTVC uses outdated dependancies) Setup a virtual environment with [`venv`](https://docs.python.org/3/library/venv.html) by running `python -m venv .venv`.
   * Activate the virtual environment with `.venv/Scripts/activate`.
     * Do note you will need to run this command before running anything in the toolbox if you used this to install dependancies, otherwise it will return an error saying it's missing some dependancies.
 * Run `pip install -r requirements.txt` to install the remaining necessary packages.

 ### 2. Download Pretrained Models
 Download the latest [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

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

 ### 6. (Optional) Enable GPU Support
 Note: Enabling GPU support is a lot of work. You will want to set this up if you are going to train your own models. Somebody took the time to make [a better guide](https://poorlydocumented.com/2019/11/installing-corentinjs-real-time-voice-cloning-project-on-windows-10-from-scratch/) on how to install everything. I recommend using it.

 This command installs additional GPU dependencies and recommended packages: `pip install -r requirements_gpu.txt`

 Additionally, you will need to ensure GPU drivers are properly installed and that your CUDA version matches your PyTorch and Tensorflow installations.
 
## Ubuntu 20.04 install instructions (tested)
### 1. Add repositories
```
sudo add-apt-repository universe
sudo add-apt-repository ppa:deadsnakes/ppa
```

### 2. Install software
```
snap install ffmpeg
sudo apt install python3.6 python3.6-dev python3 python3-pip git
pip3 install virtualenv
```

Additional steps are needed to overcome bugs with portaudio and QT:

* Portaudio bugfix: https://stackoverflow.com/a/60824906

```
sudo apt install libasound2-dev
git clone -b alsapatch https://github.com/gglockner/portaudio
cd portaudio
./configure && make
sudo make install
sudo ldconfig
cd ..
```

* QT bugfix: https://askubuntu.com/a/1069502

```
sudo apt install --reinstall libxcb-xinerama0
```

### 3. Make a virtual environment and activate it
```
~/.local/bin/virtualenv --python python36 rtvc
source rtvc/bin/activate
```

### 4. Download RTVC
```
git clone --depth 1 https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
```

### 5. Install requirements
```
cd Real-Time-Voice-Cloning
pip install torch
pip install -r requirements.txt
pip install webrtcvad
```

### 6. Get pretrained models
```
wget https://www.dropbox.com/s/5udq50bkpw2hipy/pretrained.zip?dl=1 -O pretrained.zip
unzip pretrained.zip
```

### 7. Launch toolbox
```
python demo_toolbox.py
```
