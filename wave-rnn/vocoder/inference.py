from vocoder.model import WaveRNN
from vocoder.params import *
from vocoder import audio
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=rnn_dims, 
        fc_dims=fc_dims, 
        bits=bits,
        pad=pad,
        upsample_factors=upsample_factors, 
        feat_dims=feat_dims,
        compute_dims=compute_dims, 
        res_out_dims=res_out_dims, 
        res_blocks=res_blocks,
        hop_length=hop_length,
        sample_rate=sample_rate
    ).cuda()
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

def infer_waveform(mel, normalize=True, batched=True, target=8000, overlap=800):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = audio.normalize_mel(mel)
    wav = _model.generate(mel, batched, target, overlap)
    if use_mu_law:
        wav = audio.expand_signal(wav)
    return wav
