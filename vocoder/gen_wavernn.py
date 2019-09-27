from vocoder.models.fatchord_version import  WaveRNN
from vocoder.audio import *
import wandb
import numpy as np


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):
    k = model.get_step() // 1000

    audios = []
    gen_audios = []
    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)
        audios.append(wandb.Audio(x.astype(np.float32), caption="%dk_steps_%d_target"%(k, i), sample_rate=hp.sample_rate))
        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, hp.mu_law)
        gen_audios.append(wandb.Audio(wav.astype(np.float32), caption="%dk_steps_%d_%s"%(k, i, batch_str), sample_rate=hp.sample_rate))
        save_wav(wav, save_str)
    wandb.log({"audio": audios}, commit=False)
    wandb.log({"generated_audio": gen_audios}, commit=False)

