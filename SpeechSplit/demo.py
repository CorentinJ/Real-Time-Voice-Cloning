# demo conversion
import os 
import torch
import pickle
import numpy as np

from model import Generator_3_Encode as Generator
from model import Generator_6 as F0_Converter

from utils import quantize_f0_torch
from model import InterpLnr

from torch.backends import cudnn

from data_loader import get_loader
from hparams import hparams, hparams_debug_string

import sys
import umap

from sklearn.decomposition import PCA

sys.path.append('../Real-Time-Voice-Cloning/')
# Real Time VOice Cloning encoder
from encoder import inference as encoder

import random
import matplotlib.pyplot as plt
import seaborn as sns

# Progress Bar
from tqdm import tqdm
# Path
from pathlib import Path

min_len_seq = hparams.min_len_seq
max_len_seq = hparams.max_len_seq
max_len_pad = hparams.max_len_pad

rootDir = '../LibriSpeech/Test_50-50'
dirName, subdirList, _ = next(os.walk(rootDir))

root_dir = '/home/yen/RTVC Dropbox/Yen S/SpeechSPlit/spmel'
feat_dir = '/home/yen/RTVC Dropbox/Yen S/SpeechSPlit/raptf0'

torch.set_default_dtype(torch.float32)
device = 'cuda:0'
Interp = InterpLnr(hparams)

G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/20000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

# P = F0_Converter(hparams).eval().to(device)
# p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
# P.load_state_dict(p_checkpoint['model'])

### Load dataset
# For fast training.
cudnn.benchmark = True

torch.set_default_dtype(torch.float32)
# Data loader.
# vcc_loader = get_loader(hparams)


############################3
# Pick First Voice For now (Todo: choose?)

np.random.seed(43)

metaname = os.path.join(root_dir, "train.pkl")
meta = pickle.load(open(metaname, "rb"))

print(len(meta))
indices_meta = np.arange(len(meta))
np.random.shuffle(indices_meta)


k = 0
# Loop through different outputs:

# trim to 20 speakers
# indices_meta = indices_meta[:20]
#encoder_outputs_conc
outputs_name = ["out_emb", "content_emb", "rhytm_emb", "freq_emb", "orig_emb"]

encoder_path = Path("../Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
encoder.load_model(encoder_path)

# encoder_outputs_conc, content_emb, rhytm_emb, freq_emb, orig_emb
for output_index in range(len(outputs_name)):
    k = 0
    encoder_output = []

    speaker_all = []
    # colors = []
    for index_speaker in tqdm(indices_meta):

        submmeta = meta[index_speaker]
        # Pick first voice
        speaker_id_name = submmeta[0]
        # emb_org_val = submmeta[1]

        if(not speaker_id_name in subdirList):
            continue
        # # exit()

        indices_voice = np.arange(len(submmeta[2]))
        # np.random.shuffle(indices_voice)


        # emb_org_val = torch.from_numpy(np.stack(emb_org_val, axis=0))
        # emb_org_val = torch.unsqueeze( emb_org_val.to(device), 0)

        j = 0
        for i in indices_voice:
            # G.load_state_dict(g_checkpoint['model'])
            if j > 6:
                break
            j +=1

            speaker_save = submmeta[2][i]
            if(speaker_id_name in ['19', '31', '39', '40', '83', '89', '103', '125', '150', '198']):
                speaker_all.append("Woman")
            else:
                speaker_all.append("Man")
            # speaker_all.append(speaker_save)
            # if(outputs_name[output_index] == "orig_emb"):
            # if(True):
            # Create path to speaker
            dirVoice = Path(os.path.join(dirName,speaker_id_name))
            # Get all wav files from speaker
            fileList = list(dirVoice.glob("**/*.flac"))
            fileList= [str(x) for x in fileList]
            speaker_save_check = speaker_save.split("/")[-1]
            
            path_wav = next((s for s in fileList if speaker_save_check in s), None)
            if(path_wav == None):
                print("Path wav not found")
                exit(1)
            preprocessed_wav = encoder.preprocess_wav(path_wav)
            emb_org_val = encoder.embed_utterance_old(preprocessed_wav)
            emb_org_val = torch.from_numpy(np.stack(emb_org_val, axis=0))
            emb_org_val = torch.unsqueeze( emb_org_val.to(device), 0)
                # encoder_output.append(embed)
                # continue

            sp_tmp = np.load(os.path.join(root_dir, speaker_save + ".npy"))
            f0_tmp = np.load(os.path.join(feat_dir, speaker_save + ".npy"))

            x_real_pad = sp_tmp[0:, :]
            f0_org_val = f0_tmp[0:]
            len_org_val = np.array([max_len_pad -1])

            a = x_real_pad[0:len_org_val[0], :]
            c = f0_org_val[0:len_org_val[0]]

            a = np.clip(a, 0, 1)

            x_real_pad = np.pad(a, ((0,max_len_pad-a.shape[0]),(0,0)), 'constant')
            f0_org_val = np.pad(c[:,np.newaxis], ((0,max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)
                        

            # data_loader_samp = vcc_loader[2]
            # data_iter_samp = iter(data_loader_samp)
            # speaker_id_name, x_real_pad, emb_org_val, f0_org_val, len_org_val = next(data_iter_samp)


            x_real_pad = torch.from_numpy(np.stack(x_real_pad, axis=0))
            
            f0_org_val = torch.from_numpy(np.stack(f0_org_val, axis=0))
            len_org_val = torch.from_numpy(np.stack(len_org_val, axis=0))

            x_real_pad =  torch.unsqueeze(x_real_pad.to(device)  , 0)
            len_org_val = torch.unsqueeze( len_org_val.to(device), 0)
            f0_org_val =  torch.unsqueeze(f0_org_val.to(device), 0)

            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)

            x_f0_intrp = Interp(x_f0, len_org_val)
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)
            
            # exit()
            # Get: Concatenated encoder output, Content, Rhytm, Freq, Original
            output= G(x_f0_intrp_org, x_real_pad, emb_org_val)
            print(output[0].shape)
            exit()
            encoder_output.append(output[output_index].cpu().detach().numpy().flatten())


    reducer = umap.UMAP()
    projected_encoder_output = reducer.fit_transform(np.array(encoder_output))

    fig = plt.figure()
    sns.scatterplot(x=projected_encoder_output[:, 0], y=projected_encoder_output[:, 1], hue=speaker_all, legend=False)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    # plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection " + outputs_name[output_index])
    plt.savefig("Projection_" + outputs_name[output_index], bbox_inches='tight')
    plt.clf()
    print("Saved Plot: " + outputs_name[output_index])
    exit()
    # exit()