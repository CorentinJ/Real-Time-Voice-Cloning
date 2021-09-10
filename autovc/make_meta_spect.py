"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('../Real-Time-Voice-Cloning/')
# Real Time VOice Cloning encoder
from encoder import inference as encoder

## Old embedding -> Replace with GE2E

# C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
# c_checkpoint = torch.load('3000000-BL.ckpt')
# new_state_dict = OrderedDict()
# for key, val in c_checkpoint['model_b'].items():
#     new_key = key[7:]
#     new_state_dict[new_key] = val
# C.load_state_dict(new_state_dict)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


num_uttrs = 10
len_crop = 256

# Directory containing mel-spectrograms
rootDir = '../LibriSpeech/train-clean-100'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# spectrogram directory
targetDir = './spmel'

# Load encoder model (hardcoded for now)
encoder_path = Path("../Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
encoder.load_model(encoder_path)

speakers = []


# Loop over all speakers
for speaker in tqdm(sorted(subdirList)):
    # make sure speaker is a number (Librispeech)
    try: 
        int(speaker)
    except ValueError: 
        continue

    
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, subdir, _ = next(os.walk(os.path.join(dirName,speaker)))
   
    embs = []
    filename_list = []

    subdir = subdir[0]
    prng = RandomState(int(subdir))
    
    dirVoice = Path(os.path.join(dirName, speaker))
    fileList = list(dirVoice.glob("**/*.flac"))
    
    if not os.path.exists(os.path.join(targetDir, speaker)):
        os.makedirs(os.path.join(targetDir, speaker))
    
    for i, fileName in enumerate(sorted(fileList)):
        
        subdir = str(os.path.basename(os.path.dirname(fileName)))        
        fileName_base = str(os.path.basename(fileName))

        # Read audio file
        x, fs = sf.read(os.path.join(fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    
         

        tmp = S
        #check if audio fragments is long enough
        if(tmp.shape[0] <= len_crop):
            continue
        
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        
         # preproces and generate embedding
        preprocessed_wav = encoder.preprocess_wav(fileName)
        embed = encoder.embed_utterance_old(preprocessed_wav)
        embs.append(embed) 
        filename_list.append(os.path.join(speaker,fileName_base[:-5]))

        # save spect    
        np.save(os.path.join(targetDir, speaker, fileName_base[:-5]),
                S.astype(np.float32), allow_pickle=False)
       
    utterances.append(np.mean(embs, axis=0))
    # free memory
    embs.clear()

    utterances.append(filename_list)
    speakers.append(utterances)
    
with open(os.path.join(targetDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

