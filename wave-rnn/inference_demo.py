from vlibs import fileio
from vocoder.vocoder_dataset import VocoderDataset
from vocoder import inference
from vocoder import audio
import numpy as np

run_name = 'mu_law'
model_dir = 'checkpoints'
model_fpath = fileio.join(model_dir, run_name + '.pt')
inference.load_model(model_fpath)

data_path = 'E:\\Datasets\\Synthesizer'
dataset = VocoderDataset(data_path)

gen_path = 'model_outputs'
fileio.ensure_dir(gen_path)

# Generate Samples
n_samples = 10
print('Generating...')
for i in sorted(np.random.choice(len(dataset), n_samples)):
    mel, wav_gt = dataset[i]
    
    out_gt_fpath = fileio.join(gen_path, "%s_%d_gt.wav" % (run_name, i))
    out_pred_fpath = fileio.join(gen_path, "%s_%d_pred.wav" % (run_name, i))
    
    wav_gt = audio.restore_signal(wav_gt)
    wav_pred = inference.infer_waveform(mel)

    audio.save_wav(out_pred_fpath, wav_pred)
    audio.save_wav(out_gt_fpath, wav_gt)
    print('')