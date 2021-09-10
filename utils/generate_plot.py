# demo conversion
import os 
import torch
import pickle
import numpy as np

from torch.backends import cudnn

import sys
import umap
import io

import random
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist

# Progress Bar
from tqdm import tqdm

# Path
from pathlib import Path

sys.path.append('../')
from SpeechSplit.utils import quantize_f0_torch
from SpeechSplit.model import InterpLnr
from SpeechSplit.hparams import hparams
from SpeechSplit.model import Generator_3_Encode as Generator_speechsplit
sys.path.append('../Real-Time-Voice-Cloning/')
# Real Time VOice Cloning encoder
from encoder import inference as encoder

import torch
from autovc.model_vc import Generator as Generator_autovc

class PlotGenerator(object):
    """generates plot with male/female distribution"""
    
    def __init__(self, generator, writer, freq, it, abs_dir, hparams=None, speechsplit=False, autovc=False):
        self.G = generator.eval()
        self.writer = writer
        self.freq = freq
        self.iteration = it

        self.speechsplit = speechsplit
        self.autovc = autovc
        
        self.root_dir = abs_dir
        if(speechsplit):
            self.root_dir = abs_dir[0]
            self.feat_dir = abs_dir[1]
            
        
        self.hparams = hparams
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
    
    def plot_to_tensorboard(self, fig, title):
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=128)
        io_buf.seek(0)

        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        
        self.writer.add_image(title, img_arr, self.iteration, dataformats='HWC')

    def speechsplit_prep(self, x_real_pad, f0_org_val):
        max_len_pad = self.hparams.max_len_pad
        Interp = InterpLnr(self.hparams)

        len_org_val = np.array([max_len_pad -1])

        a = x_real_pad[0:len_org_val[0], :]
        c = f0_org_val[0:len_org_val[0]]

        a = np.clip(a, 0, 1)

        x_real_pad = np.pad(a, ((0,max_len_pad-a.shape[0]),(0,0)), 'constant')
        f0_org_val = np.pad(c[:,np.newaxis], ((0,max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)
        
        x_real_pad = torch.from_numpy(np.stack(x_real_pad, axis=0))
        
        f0_org_val = torch.from_numpy(np.stack(f0_org_val, axis=0))
        len_org_val = torch.from_numpy(np.stack(len_org_val, axis=0))

        x_real_pad =  torch.unsqueeze(x_real_pad.to(self.device)  , 0)
        len_org_val = torch.unsqueeze( len_org_val.to(self.device), 0)
        f0_org_val =  torch.unsqueeze(f0_org_val.to(self.device), 0)

        x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)

        x_f0_intrp = Interp(x_f0, len_org_val)
        f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
        x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)

        return x_real_pad, x_f0_intrp_org

    def calculate_distance(self, projected_encoder_output, speaker_gender, prefix=''):
        projected_encoder_output_Man = projected_encoder_output[np.where(np.array(speaker_gender) == "Man")]
        projected_encoder_output_Woman = projected_encoder_output[np.where(np.array(speaker_gender) == "Woman")]
        
        Y = cdist(projected_encoder_output_Man, projected_encoder_output_Woman, 'euclidean')
        Y = np.mean(np.sum(Y, axis=1))
        title = "Emb-Dist/" + prefix + 'Distance Male|Female embedding'
        
        if self.writer != None:
            self.writer.add_scalar(title, Y, self.iteration)
        else:
            print(Y)

    def plot_image(self):
        # Dataset
        rootDir = '../LibriSpeech/Test_50-50'
        dirName, subdirList, _ = next(os.walk(rootDir))
              
        ### Load dataset
        cudnn.benchmark = True
        torch.set_default_dtype(torch.float32)
        
        # Load data
        metaname = os.path.join(self.root_dir, "train.pkl")

        meta = np.array(pickle.load(open(metaname, "rb")), dtype=object)
        meta = np.array(sorted(meta, key=lambda x: x[0]), dtype=object)
        
        # Get indices of voices from Test_50_50
        indices_librispeech = np.searchsorted(meta[:,0], np.array(subdirList))
        meta = meta[indices_librispeech]

        indices_meta = np.arange(len(meta))
        
        # Loop through different outputs:
        encoder_path = Path("../Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
        encoder.load_model(encoder_path)

        encoder_output = []
        if self.speechsplit:
            encoder_output = [[],[],[],[]]
            prefix = ["encoder_outputs/", "cont_enc/", "rhytm_enc/", "freq_enc/"]
        speaker_gender = []
        speaker_all = []
        
        plot_specto = True
        for index_speaker in tqdm(indices_meta):

            submmeta = meta[index_speaker]
            # Pick first voice
            speaker_id_name = submmeta[0]
            
            indices_voice = np.arange(len(submmeta[2]))
            
            j = 0
            for i in indices_voice:
                
                if j > 6:
                    break
                j +=1

                speaker_save = submmeta[2][i]
                
                
                # Create path to speaker
                dirVoice = Path(os.path.join(dirName,speaker_id_name))

                # Get all wav files from speaker
                fileList = list(dirVoice.glob("**/*.flac"))
                fileList= [str(x) for x in fileList]
                # get base name from current speaker
                speaker_save_check = speaker_save.split("/")[-1]
                # get full path
                path_wav = next((s for s in fileList if speaker_save_check in s), None)

                if(path_wav == None):
                    print("Path flac not found")
                    exit(1)
                
                preprocessed_wav = encoder.preprocess_wav(path_wav)
                emb_org_val = encoder.embed_utterance_old(preprocessed_wav)

                x_real_pad = np.load(os.path.join(self.root_dir, speaker_save + ".npy"))
                emb_org_val = torch.from_numpy(np.stack(emb_org_val, axis=0))
                x_real_pad = torch.from_numpy(np.stack(x_real_pad, axis=0))
                
                # Check if spectogram is long enough
                if(x_real_pad.shape[0] <= 512):
                    j-=1
                    continue             
                
                if(self.speechsplit):
                    # Load Spectogram and F0 spectogram
                    f0_tmp = np.load(os.path.join(self.feat_dir, speaker_save + ".npy"))
                    x_real_pad, x_f0_intrp_org = self.speechsplit_prep(x_real_pad, f0_tmp)

                    # Get output from encoder
                    # We are interested in the frequency embedding
                    output = self.G(x_f0_intrp_org, x_real_pad, emb_org_val)
                    for i in range(len(encoder_output)):
                        encoder_output[i].append(output[i].cpu().detach().numpy().flatten())
                
                elif(self.autovc):
                    # Crop embedding                    
                    x_real_pad = x_real_pad[:512, :]
                    
                    emb_org_val = torch.unsqueeze( emb_org_val.to(self.device), 0)
                    x_real_pad =torch.unsqueeze( x_real_pad.to(self.device), 0)
                    
                    # Get output from encoder
                    output = self.G(x_real_pad, emb_org_val, None)
                    encoder_output.append(output.cpu().detach().numpy().flatten())


                speaker_all.append(speaker_id_name)
                if(speaker_id_name in ['19', '31', '39', '40', '83', '89', '103', '125', '150', '198']):
                    speaker_gender.append("Woman")
                else:
                    speaker_gender.append("Man")

                if(plot_specto and self.autovc):                 
                    x_identic_val = self.G(x_real_pad, emb_org_val, emb_org_val)[0]
                    melsp_gd_pad = x_real_pad[0].cpu().detach().numpy().T
                    melsp_out = x_identic_val[0].cpu().detach().numpy().T
                    
                    min_value = np.min(np.hstack([melsp_gd_pad, melsp_out]))
                    max_value = np.max(np.hstack([melsp_gd_pad, melsp_out]))
                    
                    fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.set_title('Original Specto')
                    im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                    ax1.set_title('Generated Specto')
                    im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                    
                    fig.set_dpi(128)
                    if(self.writer != None):
                        self.plot_to_tensorboard(fig, 'Spectogram Projection')
                    else:
                        plt.show()
                    plt.clf()
                    plot_specto = False

        reducer = umap.UMAP()
        if self.speechsplit:
            for i in range(len(encoder_output)):
                projected_encoder_output = reducer.fit_transform(np.array(encoder_output[i]))            
                self.calculate_distance(projected_encoder_output, speaker_gender, prefix=prefix[i])
            
            encoder_output = encoder_output[0]
        
        projected_encoder_output = reducer.fit_transform(np.array(encoder_output))
        
        self.calculate_distance(projected_encoder_output, speaker_gender)
        
        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5, 5))
        ax1.set_title('Male/Female Distribution')
        ax1.axis("off")
        sns.scatterplot(x=projected_encoder_output[:, 0], y=projected_encoder_output[:, 1], hue=speaker_gender, legend=False, ax=ax1)
        ax2.set_title('Distribution per Speaker')
        sns.scatterplot(x=projected_encoder_output[:, 0], y=projected_encoder_output[:, 1], hue=speaker_all, legend=False, ax=ax2)
        
        fig.set_dpi(128)
        plt.title("UMAP projection")
        plt.axis("off")
        if(self.writer != None):
            self.plot_to_tensorboard(fig, 'UMAP projection')
        else:
            plt.show()
        
        plt.clf()
        print("Saved Plot to Tensorboard")

# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if use_cuda else 'cpu')
# # # G = Generator_autovc(16, 256, 512, 16).eval().to(device)
# # # g_checkpoint = torch.load('../autovc/run/models/427000-G.ckpt', map_location=device)
# # # G.load_state_dict(g_checkpoint['model'])

# G = Generator_speechsplit(hparams).eval().to(device)
# g_checkpoint = torch.load('../SpeechSplit/assets/20000-G.ckpt', map_location=lambda storage, loc: storage)
# G.load_state_dict(g_checkpoint['model'])

# gen = PlotGenerator(G, None, None, 1, ["/home/yen/RTVC Dropbox/Yen S/SpeechSPlit/spmel", "/home/yen/RTVC Dropbox/Yen S/SpeechSPlit/raptf0"], hparams=hparams, speechsplit=True)
# gen.plot_image()