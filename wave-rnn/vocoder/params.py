
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

## Model parameters
# Number of bits for the encoding. Higher means higher quality output but longer training time 
# and training memory required.
bits = 9
pad = 2
seq_len = hop_length * 5
mel_win = seq_len // hop_length + 2 * pad
# Minimum number of mel frames below which samples are discarded for training
min_n_frames = 10



