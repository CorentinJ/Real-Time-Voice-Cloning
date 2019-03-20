import torch
from vlibs import fileio
from models.wavernn import WaveRNN
from utils.vocoder_dataset import VocoderDataset
from params import *
from utils import audio

run_name = 'from_synth'
model_dir = 'checkpoints'
model_fpath = fileio.join(model_dir, run_name + '.pt')

model = WaveRNN(rnn_dims=512,
              fc_dims=512,
              bits=bits,
              pad=pad,
              upsample_factors=(5, 5, 8),
              feat_dims=80,
              compute_dims=128,
              res_out_dims=128,
              res_blocks=10,
              hop_length=hop_length,
              sample_rate=sample_rate).cuda()

checkpoint = torch.load(model_fpath)
step = checkpoint['step']
model.load_state_dict(checkpoint['model_state'])

data_path = 'E:\\Datasets\\Synthesizer'
gen_path = 'model_outputs'
fileio.ensure_dir(gen_path)

dataset = VocoderDataset(data_path)

# Generate Samples
target = 11000
overlap = 550
k = step // 1000
for i, (mel, wav_gt) in enumerate(dataset):
    print('Generating...')
    out_gt_fpath = fileio.join(gen_path, "%s_%dk_steps_%d_gt.wav" % (run_name, k, i))
    out_pred_fpath = fileio.join(gen_path, "%s_%dk_steps_%d_pred.wav" % (run_name, k, i))
    
    wav_gt = audio.restore_signal(wav_gt)
    wav_pred = model.generate(mel, True, target, overlap)

    audio.save_wav(out_pred_fpath, wav_pred)
    audio.save_wav(out_gt_fpath, wav_gt)


