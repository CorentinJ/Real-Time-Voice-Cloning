# demo conversion
import os 
import torch
import pickle
import numpy as np
# from utils import pad_seq_to_2
# from utils import quantize_f0_numpy
# from model import Generator_3 as Generator
# from model import Generator_6 as F0_Converter

# from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy
# from model import InterpLnr

import argparse
from torch.backends import cudnn

# from solver import Solver
# from data_loader import get_loader
# from hparams import hparams, hparams_debug_string

# spectrogram to waveform
import soundfile
import pickle
from synthesis import build_model
from synthesis import wavegen

import matplotlib.pyplot as plt

# min_len_seq = hparams.min_len_seq
# max_len_seq = hparams.max_len_seq
# max_len_pad = hparams.max_len_pad

root_dir = 'assets/spmel'
feat_dir = 'assets/raptf0'

torch.set_default_dtype(torch.float32)
device = 'cuda:0'
# Interp = InterpLnr(hparams)

# G = Generator(hparams).eval().to(device)
# g_checkpoint = torch.load('assets/265000-G.ckpt', map_location=lambda storage, loc: storage)
# G.load_state_dict(g_checkpoint['model'])

# P = F0_Converter(hparams).eval().to(device)
# p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
# P.load_state_dict(p_checkpoint['model'])

model = build_model().to(device)
checkpoint = torch.load("../SpeechSplit/assets/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

exit()