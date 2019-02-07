
import numpy as np
from datasets.audio import *
import os
from hparams import hparams
import sounddevice

n_sample = 0
mel_folder = 'logs-Tacotron/mel-spectrograms'
mel_file = 'mel-prediction-step-{}.npy'.format(n_sample)
out_dir = 'wav_out'

os.makedirs(out_dir, exist_ok=True)

#mel_file = os.path.join(mel_folder, mel_file)

from vlibs import fileio

# fnames = fileio.listdir('logs-two_outputs/mel-spectrograms/')
fnames = fileio.listdir('tacotron_output/eval/')
for i in range(1, len(fnames)):
    # mel_file = 'logs-two_outputs/mel-spectrograms/mel-prediction-step-110000.npy'
    mel_file = fileio.join('tacotron_output/eval/', fnames[i])
    mel_spectro = np.load(mel_file) #.transpose()
    wav = inv_mel_spectrogram(mel_spectro.T, hparams) 
    sounddevice.wait()
    print(fnames[i])
    sounddevice.play(wav, 22050)
sounddevice.wait()
quit()

save_wav(wav, os.path.join(out_dir, 'test_mel_{}.wav'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
        sr=hparams.sample_rate)


# In[3]:


from tacotron.utils.plot import *

plot_spectrogram(mel_spectro, path=os.path.join(out_dir, 'test_mel_{}.png'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))))


# In[4]:


lin_file = 'training_data/linear/linear-LJ001-0005.npy'
lin_spectro = np.load(lin_file)
lin_spectro.shape


# In[5]:


wav = inv_linear_spectrogram(lin_spectro.T, hparams)
save_wav(wav, os.path.join(out_dir, 'test_linear_{}.wav'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
        sr=hparams.sample_rate)


# In[6]:


plot_spectrogram(lin_spectro, path=os.path.join(out_dir, 'test_linear_{}.png'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
                auto_aspect=True)

