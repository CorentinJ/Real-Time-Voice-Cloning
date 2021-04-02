import numpy as np
from synthesizer.hparams import hparams


def duration_warp(real_d, int_d):
    total_diff = sum(real_d) - sum(int_d)
    drop_diffs = np.array(real_d) - np.array(int_d)
    drop_order = np.argsort(-drop_diffs)
    for i in range(int(total_diff)):
        index = drop_order[i]
        int_d[index] += 1

    return int_d


def get_alignment(file):
    sil_phones = ['sil', 'sp', 'spn']
    phones = []
    durations_real = []
    durations_int = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for line in file:
        try:
            s, e, p = line.strip().split('\t')
            s, e = float(s), float(e)
        except ValueError:
            s, e = line.strip().split('\t')
            s, e = float(s), float(e)
            # p = ''

       # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)

        d = e*hparams.sample_rate/hparams.hop_size - \
            s*hparams.sample_rate/hparams.hop_size
        durations_real.append(d)
        durations_int.append(int(d))

    # Trimming tailing silences
    durations_real = durations_real[:end_idx]
    durations_int = durations_int[:end_idx]
    phones = phones[:end_idx]
    durations = duration_warp(durations_real, durations_int)
    return phones, durations, start_time, end_time
