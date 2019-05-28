from vocoder.vocoder_dataset import VocoderDataset
from vocoder.params import print_params, model_name, use_mu_law
from vocoder import inference
from vocoder import audio
from pathlib import Path
import sounddevice as sd
import numpy as np
import torch

print_params()


model_fpath = "vocoder/saved_models/pretrained/pretrained.pt"
inference.load_model(model_fpath)

syn_dir = Path("E:\\Datasets\\SV2TTS\\Synthesizer")
voc_dir = Path("E:\\Datasets\\SV2TTS\\Vocoder")
wav_dir = syn_dir.joinpath("audio")
gta_dir = voc_dir.joinpath("mels_gta")
metadata_fpath = voc_dir.joinpath("synthesized.txt")
dataset = VocoderDataset(metadata_fpath, gta_dir, wav_dir)

# Generate Samples
n_samples = 5
print('Generating...')
for i in sorted(np.random.choice(len(dataset), n_samples)):
    mel, wav_gt = dataset[i]
    
    # out_gt_fpath = fileio.join(gen_path, "%s_%d_gt.wav" % (model_name, i))
    # out_pred_fpath = fileio.join(gen_path, "%s_%d_pred.wav" % (model_name, i))
    
    wav_gt = audio.unquantize_signal(wav_gt)
    if use_mu_law:
        wav_gt = audio.expand_signal(wav_gt)
    sd.wait()
    sd.play(wav_gt, 16000)
    
    wav_pred = inference.infer_waveform(mel, normalize=False)   # The dataloader already normalizes
    sd.wait()
    sd.play(wav_pred, 16000)



    # audio.save_wav(out_pred_fpath, wav_pred)
    # audio.save_wav(out_gt_fpath, wav_gt)
    print('')
    