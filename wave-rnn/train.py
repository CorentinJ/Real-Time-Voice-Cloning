# ## Alternative Model (Training)
# I've found WaveRNN quite slow to train so here's an alternative that utilises the optimised rnn 
# kernels in Pytorch. The model below is much much faster to train, it will converge in 48hrs when 
# training on 22.5kHz samples (or 24hrs using 16kHz samples) on a single GTX1080. It also works 
# quite well with predicted GTA features. 
# The model is simply two residual GRUs in sequence and then three dense layers with a 512 softmax 
# output. This is supplemented with an upsampling network.
# Since the Pytorch rnn kernels are 'closed', the options for conditioning sites are greatly 
# reduced. Here's the strategy I went with given that restriction:  
# 1 - Upsampling: Nearest neighbour upsampling followed by 2d convolutions with 'horizontal' kernels
# to interpolate. Split up into two or three layers depending on the stft hop length.
# 2 - A 1d resnet with a 5 wide conv input and 1x1 res blocks. Not sure if this is necessary, but 
# the thinking behind it is: the upsampled features give a local view of the conditioning - why not
# supplement that with a much wider view of conditioning features, including a peek at the future. 
# One thing to note is that the resnet is computed only once and in parallel, so it shouldn't slow 
# down training/generation much. 
# Train this model to ~500k steps for 8/9bit linear samples or ~1M steps for 10bit linear or 9+bit 
# mu_law. 

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from vocoder.vocoder_dataset import VocoderDataset
from vocoder.model import WaveRNN
from vlibs import fileio
from vocoder.params import *
import time
import numpy as np


model_dir = 'checkpoints'
fileio.ensure_dir(model_dir)
model_fpath = fileio.join(model_dir, model_name + '.pt')

data_path = "../data/Synthesizer"
# data_path = "E:/Datasets/Synthesizer"
gen_path = 'model_outputs'
fileio.ensure_dir(gen_path)

def collate(batch) :
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hop_length for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + seq_len + 1] for i, x in enumerate(batch)]
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    
    x_input = 2 * coarse[:, :seq_len].float() / (2 ** bits - 1.) - 1.
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse

if __name__ == '__main__':
    print_params()
    
    dataset = VocoderDataset(data_path)
    model = WaveRNN(
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
    )
    model = model.cuda()
    
    global step
    if os.path.exists(model_fpath):
        checkpoint = torch.load(model_fpath)
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state'])
        print("Loaded from step %d." % step)
    else:
        step = 0
        print("Starting from scratch.")
    
    def train(model, optimiser, epochs, batch_size, classes, seq_len, step, lr=1e-4):
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        
        for e in range(epochs):
            trn_loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size, 
                                    num_workers=2, shuffle=True, pin_memory=True)
            start = time.time()
            running_loss = 0.
    
            iters = len(trn_loader)
    
            for i, (x, m, y) in enumerate(trn_loader):
                x, m, y = x.cuda(), m.cuda(), y.cuda()
    
                y_hat = model(x, m)
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                y = y.unsqueeze(-1)
                loss = criterion(y_hat, y)
                
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                
                speed = (i + 1) / (time.time() - start)
                avg_loss = running_loss / (i + 1)
                
                step += 1
                k = step // 1000
                print('\rEpoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- %.2f steps/sec -- '
                       'Step: %ik' % (e + 1, epochs, i + 1, iters, avg_loss, speed, k), end='')
                
                if (i + 1) % 1000 == 0:
                    torch.save({'step': step, 'model_state': model.state_dict()}, model_fpath)
            
            torch.save({'step': step, 'model_state': model.state_dict()}, model_fpath)
            print('<saved>')
            
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, epochs=100, batch_size=100, classes=2 ** bits,
          seq_len=seq_len, step=step, lr=1e-4)
    