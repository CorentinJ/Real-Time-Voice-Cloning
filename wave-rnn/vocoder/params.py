## Audio parameters - WARNING: these should match those of the synthesizer!
sample_rate = 16000
n_fft = 800
fft_bins = 513
num_mels = 80
hop_length = 200
win_length = 800
fmin = 55
min_level_db = -100
ref_level_db = 20
# Maximum absolute value found in mels generated from the synthesizer
mel_max_abs_value = 4
# Whether or not to apply a mu-law to the audio before quantization (and after restoration of the
# quantized signal). This results in a greater audio quality but also requires more steps to 
# reach convergence.
use_mu_law = True
# Minimum number of mel frames below which samples are discarded for training
min_n_frames = 10

# ## Model parameters
# model_name = 'mu_law'
# # Number of bits for the encoding. Higher means higher quality output but longer training time 
# # and training memory required.
# bits = 9
# pad = 2
# seq_len = hop_length * 5
# mel_win = seq_len // hop_length + 2 * pad
# rnn_dims = 512
# fc_dims = 512
# upsample_factors = (5, 5, 8)
# feat_dims = 80
# compute_dims = 128
# res_out_dims = 128
# res_blocks = 10

## Model parameters
model_name = 'mu_law_big'
# Number of bits for the encoding. Higher means higher quality output but longer training time 
# and training memory required.
bits = 9
pad = 2
seq_len = hop_length * 5
mel_win = seq_len // hop_length + 2 * pad
rnn_dims = 768
fc_dims = 768
upsample_factors = (5, 5, 8)
feat_dims = 80
compute_dims = 196
res_out_dims = 196
res_blocks = 10


def print_params():
    for param_name in sorted(globals()):
        if param_name.startswith('__') or param_name == 'print_params':
            continue
        print("%s: %s" % (param_name, str(globals()[param_name])))