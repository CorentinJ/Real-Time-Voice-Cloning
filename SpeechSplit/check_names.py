# demo conversion
import os 
import torch
import pickle
import numpy as np
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy
from model import InterpLnr

import argparse
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from hparams import hparams, hparams_debug_string

# spectrogram to waveform
import torch
import soundfile
import pickle
import os
from synthesis import build_model
from synthesis import wavegen

import matplotlib.pyplot as plt

min_len_seq = hparams.min_len_seq
max_len_seq = hparams.max_len_seq
max_len_pad = hparams.max_len_pad

root_dir = 'assets/spmel'
feat_dir = 'assets/raptf0'

torch.set_default_dtype(torch.float32)
device = 'cuda:0'
Interp = InterpLnr(hparams)

G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/265000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

P = F0_Converter(hparams).eval().to(device)
p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])

### Load dataset
# For fast training.
cudnn.benchmark = True

torch.set_default_dtype(torch.float32)
# Data loader.
vcc_loader = get_loader(hparams)


############################3
# Pick First Voice For now (Todo: choose?)

metaname = os.path.join(root_dir, "train.pkl")
meta = pickle.load(open(metaname, "rb"))

# # Pick first voice
for i in range(5):
    submmeta = meta[3+i]
    speaker_id_name = submmeta[0]
    emb_org_val = submmeta[1][0]
    # for speaker_save in sbmt[2]:
    speaker_save = submmeta[2][0]
    print(speaker_save[4:])

    sp_tmp = np.load(os.path.join(root_dir, speaker_save + ".npy"))
    f0_tmp = np.load(os.path.join(feat_dir, speaker_save + ".npy"))

    x_real_pad = sp_tmp[0:, :]
    f0_org_val = f0_tmp[0:]
    len_org_val = np.array([max_len_pad -1])

    print(x_real_pad.shape)
    print(f0_org_val.shape)

    a = x_real_pad[0:len_org_val[0], :]
    c = f0_org_val[0:len_org_val[0]]

    print(a.shape)
    print(c.shape)

    a = np.clip(a, 0, 1)


    x_real_pad = np.pad(a, ((0,max_len_pad-a.shape[0]),(0,0)), 'constant')
    f0_org_val = np.pad(c[:,np.newaxis], ((0,max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)

    print(x_real_pad.shape)
    print(f0_org_val.shape)


    # data_loader_samp = vcc_loader[2]
    # data_iter_samp = iter(data_loader_samp)
    # speaker_id_name, x_real_pad, emb_org_val, f0_org_val, len_org_val = next(data_iter_samp)


    x_real_pad = torch.from_numpy(np.stack(x_real_pad, axis=0))
    emb_org_val = torch.from_numpy(np.stack(emb_org_val, axis=0))
    f0_org_val = torch.from_numpy(np.stack(f0_org_val, axis=0))
    len_org_val = torch.from_numpy(np.stack(len_org_val, axis=0))

    x_real_pad =  torch.unsqueeze(x_real_pad.to(device)  , 0)
    emb_org_val = torch.unsqueeze( emb_org_val.to(device), 0)
    len_org_val = torch.unsqueeze( len_org_val.to(device), 0)
    f0_org_val =  torch.unsqueeze(f0_org_val.to(device), 0)

    # x_real_pad = torch.unsqueeze(x_real_pad, 0)
    emb_org_val_empty = torch.zeros_like(emb_org_val)
    x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
    x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
    x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)

    print(x_f0.shape)
    print(len_org_val.shape)

    x_f0_intrp = Interp(x_f0, len_org_val) 
    f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
    x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)

    x_f0_F_intrp = Interp(x_f0_F, len_org_val) 
    f0_F_org_intrp = quantize_f0_torch(x_f0_F_intrp[:,:,-1])[0]
    x_f0_F_intrp_org = torch.cat((x_f0_F_intrp[:,:,:-1], f0_F_org_intrp), dim=-1)

    x_f0_C_intrp = Interp(x_f0_C, len_org_val) 
    f0_C_org_intrp = quantize_f0_torch(x_f0_C_intrp[:,:,-1])[0]
    x_f0_C_intrp_org = torch.cat((x_f0_C_intrp[:,:,:-1], f0_C_org_intrp), dim=-1)
                            
    # x_identic_val = G(x_f0_intrp_org, x_real_pad, emb_org_val)
    # x_identic_woF = G(x_f0_F_intrp_org, x_real_pad, emb_org_val)
    # x_identic_woR = G(x_f0_intrp_org, torch.zeros_like(x_real_pad), emb_org_val)
    # x_identic_woC = G(x_f0_C_intrp_org, x_real_pad, emb_org_val)

    conditions = ['N', 'F', 'R', 'C']
    spect_vc = []

    with torch.no_grad():
        # for condition in conditions:
        # if condition == 'N':
        x_identic_val = G(x_f0_intrp_org, x_real_pad, emb_org_val)
        # if condition == 'F':
        x_identic_woF = G(x_f0_intrp_org, x_real_pad, emb_org_val_empty)
        # if condition == 'R':
        x_identic_woR = G(x_f0_intrp_org, torch.zeros_like(x_real_pad), emb_org_val)
        # if condition == 'C':
        x_identic_woC = G(x_f0_C_intrp_org, x_real_pad, emb_org_val)
            
        melsp_gd_pad = x_real_pad[0].cpu().numpy().T
        melsp_out = x_identic_val[0].cpu().numpy().T
        melsp_woF = x_identic_woF[0].cpu().numpy().T
        melsp_woR = x_identic_woR[0].cpu().numpy().T
        melsp_woC = x_identic_woC[0].cpu().numpy().T
        
        min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
        max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
        
        fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
        im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
        im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
        im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
        im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
        im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
        plt.show()
        plt.close()
    
    exit()

  
    if not os.path.exists('results'):
        os.makedirs('results')

    model = build_model().to(device)
    checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth")
    model.load_state_dict(checkpoint["state_dict"])

    print(len(spect_vc))
    for spect in spect_vc:
        name = spect[0]
        name = name.split('/')[1]   
        print(name)

        c = spect[1]
        print(len(c))
        waveform = wavegen(model, c=c)   
        soundfile.write('results/'+name+'_0.wav', waveform, samplerate=16000)