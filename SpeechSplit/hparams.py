# Yen
# Added to import tfcompat even when called from other directory
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
###

from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    # model   
    freq = 8,
    dim_neck = 64, # 8 Content
    freq_2 = 8,
    dim_neck_2 = 6, # 1 Rhytm
    freq_3 = 8,
    dim_neck_3 = 64, # 16 Pitch
    
    dim_enc = 512,
    dim_enc_2 = 128,
    dim_enc_3 = 256,
    
    dim_freq = 80,
    dim_spk_emb = 256,
    dim_f0 = 257,
    dim_dec = 512,
    len_raw = 128,
    chs_grp = 16,
    
    # interp
    min_len_seg = 19,
    max_len_seg = 32,
    min_len_seq = 128, # 64
    max_len_seq = 256, # 128
    max_len_pad = 320, # 192
    
    # data loader
    root_dir = 'assets/spmel',
    feat_dir = 'assets/raptf0',
    batch_size = 16,
    mode = 'train',
    shuffle = True,
    num_workers = 0,
    samplier = 2,
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
