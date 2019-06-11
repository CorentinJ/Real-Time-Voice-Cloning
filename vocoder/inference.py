from vocoder.models.fatchord_version import WaveRNN
from vocoder.utils import audio
from vocoder import hparams as hp
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    ).cuda()
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()

def is_loaded():
    return _model is not None

def infer_waveform(mel, normalize=True, batched=True, target=8000, overlap=800, no_pre=True):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, no_pre)
    return wav


if __name__ == '__main__':
    fpath = "saved_models/gen_s_mel_raw/gen_s_mel_raw.pt"
    load_model(fpath)
    
    mel_root = r"E:\Datasets\SV2TTS\vocoder\mels_gta"
    import numpy as np
    from pathlib import Path
    import sounddevice as sd
    import random
    
    mel_fpaths = list(Path(mel_root).glob("*.npy"))
    random.shuffle(mel_fpaths)
    for _ in range(50):
        mel = None
        for _ in range(5):
            mel_fpath = mel_fpaths.pop(0)
            sub_mel = np.load(mel_fpath).T.astype(np.float32)
            mel = sub_mel if mel is None else np.concatenate((mel, sub_mel), axis=1)
            
        wav = infer_waveform(mel, no_pre=False)
        
        sd.wait()
        sd.play(wav, 16000)

        # import matplotlib.pyplot as plt
        # _, axs = plt.subplots(2)
        # axs[0].imshow(mel)
        # axs[1].plot(wav)
        # plt.show()
        # sd.stop()