## Audio parameters - WARNING: these should match those of the synthesizer!
sample_rate = 16000
hop_length = 200
# Maximum absolute value found in mels generated from the synthesizer
mel_max_abs_value = 4
# Whether or not to apply a mu-law to the audio before quantization (and after restoration of the
# quantized signal). This results in a greater audio quality but also requires more steps to 
# reach convergence.
use_mu_law = True
# Minimum number of mel frames below which samples are discarded for training
min_n_frames = 10

## Model parameters
# model_name = 'mu_law_prune'
model_name = 'mu_law'
# Number of bits for the encoding. Higher means higher quality output but longer training time 
# and training memory required.
bits = 9
pad = 2
seq_len = hop_length * 5
mel_win = seq_len // hop_length + 2 * pad
rnn_dims = 512
fc_dims = 512
upsample_factors = (5, 5, 8)
feat_dims = 80
compute_dims = 128
res_out_dims = 128
res_blocks = 10


def print_params():
    for param_name in sorted(globals()):
        if param_name.startswith('__') or param_name == 'print_params':
            continue
        print("%s: %s" % (param_name, str(globals()[param_name])))