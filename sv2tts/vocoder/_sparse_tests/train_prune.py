from vocoder.vocoder_dataset import VocoderDataset
from torch.utils.data import DataLoader
from vocoder.params import *
from vocoder.model import WaveRNN
from vlibs import fileio
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import time
import os

class PruneMask:
    def __init__(self, layer, prune_rnn_input):
        # TODO: replace by bool tensor?
        self.mask = []
        self.p_idx = [0]
        self.total_params = 0
        self.n_pruned_params = 0
        self.split_size = 0
        self.init_mask(layer, prune_rnn_input)
    
    def init_mask(self, layer, prune_rnn_input):
        # Determine the layer type and num matrix splits if rnn 
        layer_type = str(layer).split('(')[0]
        splits = {'Linear': 1, 'GRU': 3, 'LSTM': 4}
        
        # Organise the num and indices of layer parameters
        # Dense will have one index and rnns two (if pruning input)
        if layer_type != 'Linear':
            self.p_idx = [0, 1] if prune_rnn_input else [1]
        
        # Get list of parameters from layers
        params = self.get_params(layer)
        
        # For each param matrix in this layer, create a mask
        for W in params:
            self.mask += [torch.ones_like(W)]
            self.total_params += W.size(0) * W.size(1)
        
        # Need a split size for mask_from_matrix() later on
        self.split_size = self.mask[0].size(0) // splits[layer_type]
    
    def get_params(self, layer):
        params = []
        for idx in self.p_idx:
            params += [list(layer.parameters())[idx].data]
        return params
        # return [list(layer.parameters())[idx].data for idx in self.p_idx]
    
    def update_mask(self, layer, z):
        params = self.get_params(layer)
        for i, W in enumerate(params):
            self.mask[i] = self.mask_from_matrix(W, z)
        self.n_pruned_params = sum(int((1 - M).sum().item()) for M in self.mask)
    
    def apply_mask(self, layer):
        params = self.get_params(layer)
        for M, W in zip(self.mask, params): 
            W *= M
    
    def mask_from_matrix(self, W, z):
        # Split into gate matrices (or not)
        W_split = torch.split(W, self.split_size)
        
        M = []
        # Loop through splits 
        for W in W_split:
            # Sort the magnitudes
            W_abs = torch.abs(W)
            sorted_abs, _ = torch.sort(W_abs.view(-1))
            
            # Pick k (num weights to zero) 
            k = int(W.size(0) * W.size(1) * z)
            threshold = sorted_abs[k]
            
            # Create the mask
            M += [(W_abs >= threshold).float()]
        
        return torch.cat(M)
    
class Pruner:
    def __init__(self, layers, start_prune, prune_steps, target_sparsity, prune_every,
                 prune_rnn_input=True):
        self.z = 0  # Objects sparsity @ time t
        self.t_0 = start_prune
        self.S = prune_steps
        self.Z = target_sparsity
        self.prune_every = prune_every
        self.num_pruned = 0
        self.masks = [PruneMask(layer, prune_rnn_input) for layer in layers]
        self.total_params = sum(m.total_params for m in self.masks)
    
    def update_sparsity(self, t):
        z = self.Z * (1 - (1 - (t - self.t_0) / self.S) ** 3)
        self.z = max(0, min(self.Z, z))
    
    def prune(self, layers, t):
        self.update_sparsity(t)
        
        for (l, m) in zip(layers, self.masks):
            if t % self.prune_every == 0 and t > self.t_0: 
                m.update_mask(l, self.z)
            if t >= self.t_0: 
                m.apply_mask(l)

        self.update_n_pruned()
    
    def restart(self, layers, t):
        # In case training is stopped
        self.update_sparsity(t)
        for (l, m) in zip(layers, self.masks):
            m.update_mask(l, self.z)
        self.update_n_pruned()
    
    def update_n_pruned(self):
        self.num_pruned = sum(m.n_pruned_params for m in self.masks)
    

model_dir = 'checkpoints'
fileio.ensure_dir(model_dir)
model_fpath = fileio.join(model_dir, model_name + '.pt')

# data_path = "../data/Synthesizer"
data_path = "E:/Datasets/Synthesizer"
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
        step = 1
        print("Starting from scratch.")

    start_prune = 0
    prune_steps = 2000
    prune_every = 50
    sparsity_target = 0.9
    layers2prune = [model.I, model.rnn1, model.rnn2, model.fc1, model.fc2, model.fc3]
    pruner = Pruner(layers2prune, start_prune, prune_steps, sparsity_target, prune_every)
    pruner.restart(layers2prune, step)

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

                pruner.prune(layers2prune, step)
                if step % 10 == 0:
                    print("Step: %d\nSparsity: %.2f\nPruned params: %d\nTotal params: "
                          "%d\nLoss: %.3f\n" % 
                          (step, pruner.z, pruner.num_pruned, pruner.total_params, avg_loss))

                k = step // 1000
                # print('\rEpoch: %i/%i -- Batch: %i/%i -- Loss: %.3f -- %.2f steps/sec -- '
                #        'Step: %ik' % (e + 1, epochs, i + 1, iters, avg_loss, speed, k), end='')
                
                if step % 100 == 0:
                    torch.save({'step': step, 'model_state': model.state_dict()}, model_fpath)
            
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, epochs=10000, batch_size=32, classes=2 ** bits,
          seq_len=seq_len, step=step, lr=1e-4)
    