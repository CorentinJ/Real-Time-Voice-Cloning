from synthesizer.hparams import hparams
from synthesizer.synthesizer import Synthesizer
from synthesizer import audio
from pathlib import Path
from typing import Union
import numpy as np
import librosa
# from tensorflow.train import get_checkpoint_state         # Pick your poison
from tensorflow._api.v1.train import get_checkpoint_state 

_model = None   # type: Synthesizer
sample_rate = hparams.sample_rate


def load_model(checkpoints_dir: Path):
    global _model
    
    if _model is None:
        _model = Synthesizer()
    checkpoint_fpath = get_checkpoint_state(checkpoints_dir).model_checkpoint_path
    _model.load(checkpoint_fpath, hparams)
    
    model_name = checkpoints_dir.parent.name.replace("logs-", "")
    step = int(checkpoint_fpath[checkpoint_fpath.rfind('-') + 1:])
    print("Loaded synthesizer \"%s\" trained to step %d" % (model_name, step))

def is_loaded():
    return _model is not None

def synthesize_spectrogram(text: str, embedding: np.ndarray):
    if not is_loaded():
        raise Exception("Load a model first")
    spec = _model.my_synthesize(embedding[None, ...], text).T
    return spec

def load_preprocess_wav(fpath):
    wav = librosa.load(fpath, hparams.sample_rate)[0]
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    return wav

def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav = load_preprocess_wav(fpath_or_wav)
    else: 
        wav = fpath_or_wav
    
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    return mel_spectrogram
