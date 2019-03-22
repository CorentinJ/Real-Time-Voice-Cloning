from vocoder.model import WaveRNN
from vocoder.params import *
from vocoder import audio
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(rnn_dims=512,
                     fc_dims=512,
                     bits=bits,
                     pad=pad,
                     upsample_factors=(5, 5, 8),
                     feat_dims=80,
                     compute_dims=128,
                     res_out_dims=128,
                     res_blocks=10,
                     hop_length=hop_length,
                     sample_rate=sample_rate).cuda()
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

def infer_waveform(mel, target=11000, overlap=550):
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    wav = _model.generate(mel, True, target, overlap)
    if use_mu_law:
        wav = audio.expand_signal(wav)
    return wav
