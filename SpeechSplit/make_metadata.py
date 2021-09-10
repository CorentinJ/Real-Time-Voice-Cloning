import sys
sys.path.append('../Real-Time-Voice-Cloning/')

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Real Time Voice Cloning encoder
from encoder import inference as encoder

# make spect f0 imports
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT


def prep_wav(wav_file):
    # read audio file
    x, fs = sf.read(wav_file)
    assert fs == 16000

    # make sure both f0 and spectogram get same length by padding
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    # add some random noise for robustness        
    wav = x * 0.96 + (prng.rand(x.shape[0])-0.5)*1e-06
    
    return wav, fs

def create_specto(wav, mel_basis, min_level):
    
    # compute spectrogram
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100

    return S

def extract_f0(wav, fs):
    # min/max fundamental frequency
    lo, hi = 50, 500

    # extract f0
    f0_rapt = sptk.rapt(wav.astype(np.float32), fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    return f0_rapt, f0_norm


np.random.seed = 42

# Root directory voices
rootDir = '../LibriSpeech/train-clean-100'

# Target directory mel and f0
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# Settings spect/f0
#####
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

#####

# Load encoder model (hardcoded for now)
encoder_path = Path("../Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
encoder.load_model(encoder_path)

speakers = []
# Access all directories in rootDir, where name of directory is speaker
for speaker in tqdm(sorted(subdirList)):
    utterances = []
    utterances.append(speaker)
   
    # Create path to speaker
    dirVoice = Path(os.path.join(dirName,speaker))
    # Get all wav files from speaker
    fileList = list(dirVoice.glob("**/*.flac"))
    
    # Check if list contains files
    if(len(fileList) == 0):
        continue

    
    embs = []
    fileNameSaves = []

    prng = RandomState(int(os.path.basename(os.path.dirname(fileList[0]))[1:]))
    for i in range(len(fileList)):
        fileName = str(os.path.basename(fileList[i]))

        # Create Directories if they do not exist
        if not os.path.exists(os.path.join(targetDir, speaker)):
            os.makedirs(os.path.join(targetDir, speaker))
        if not os.path.exists(os.path.join(targetDir_f0, speaker)):
            os.makedirs(os.path.join(targetDir_f0, speaker)) 

        # If rewrite is off skip already written files
        # if os.path.exists(os.path.join(targetDir, speaker, fileName[:-5])):
        #     continue
        
        # Speaker embedding using GE2E encoder
        # preproces and generate embedding
        preprocessed_wav = encoder.preprocess_wav(fileList[i])
        embed = encoder.embed_utterance_old(preprocessed_wav)
        embs.append(embed)        

        wav, fs = prep_wav(fileList[i])
        S = create_specto(wav, mel_basis, min_level)
        
        f0_rapt, f0_norm = extract_f0(wav, fs)
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, speaker, fileName[:-5]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, speaker, fileName[:-5]),
                f0_norm.astype(np.float32), allow_pickle=False)
        fileNameSaves.append(os.path.join(speaker,fileName[:-5]))
    utterances.append(embs)

    
    # create file list
    # for filePath in sorted(fileList):
    #     fileName = str(os.path.basename(filePath))
    utterances.append(fileNameSaves)
    speakers.append(utterances)
    
with open(os.path.join(targetDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)