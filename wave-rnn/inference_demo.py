from vlibs import fileio
from vocoder.vocoder_dataset import VocoderDataset
from vocoder import inference
from vocoder import audio
import numpy as np
from vocoder.params import print_params, model_name, use_mu_law
import torch

print_params()


model_dir = 'checkpoints'
model_fpath = fileio.join(model_dir, model_name + '.pt')
inference.load_model(model_fpath)

data_path = 'E:\\Datasets\\Synthesizer'
dataset = VocoderDataset(data_path)

gen_path = 'model_outputs'
fileio.ensure_dir(gen_path)

# Generate Samples
n_samples = 5
print('Generating...')
for i in sorted(np.random.choice(len(dataset), n_samples)):
    mel, wav_gt = dataset[i]
    
    out_gt_fpath = fileio.join(gen_path, "%s_%d_gt.wav" % (model_name, i))
    out_pred_fpath = fileio.join(gen_path, "%s_%d_pred.wav" % (model_name, i))
    
    wav_gt = audio.restore_signal(wav_gt)
    if use_mu_law:
        wav_gt = audio.expand_signal(wav_gt)
    wav_pred = inference.infer_waveform(mel, normalize=False)   # The dataloader already normalizes

    audio.save_wav(out_pred_fpath, wav_pred)
    audio.save_wav(out_gt_fpath, wav_gt)
    print('')
    