I have updated the requirements a little bit. Replace the new txt file in the top repository, and install portaudio:

```
pip install -r requirements.txt;
sudo apt-get install portaudio19-dev python-all-dev python3-all-dev && sudo pip install pyaudio;
brew install portaudio;
```

```speech2speech_ff.py``` supports speech and text both from files.
```speech2speech_fv.py``` supports speech from files, and text from recording.
```speech2speech_vv.py``` is a test version of both voice and text from recording. To run it, we need to replace the ```audio.py``` in the ```encoder``` folder.  It's better to copy whole ```Real-Time-Voice-Cloning``` repository and do it. Because I don't have a GPU and Google Colab doesn't support input audio device, so I can't test recording code.

For file-file code, it could be test by the ```test.ipynb``` here : https://drive.google.com/drive/folders/13hMCz6YzcCuEsM5nx3GkZo0lDkSVjKyK?usp=sharing

To run it locally, just put the speech2speech py files into the top repository as ```demo_cli.py```.   And please remember that the lengh of words for text content needs to be 20+-.
