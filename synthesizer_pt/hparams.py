# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 16000
n_fft = 800
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 200                    # For 16000 Hz, 200 = 12.5ms - in line with Tacotron 2 paper
win_length = 800                    # For 16000 Hz, 800 = 50ms - same reason as above
fmin = 50
symmetric_mels = True               # If true, data is rescaled to be [-max, max]. If false, [0, max].
max_abs_value = 4.                  # Gradient explodes if too big, premature convergence if too small.
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']
tts_stop_threshold = -3.4           # Value below which audio generation ends.
                                    # For example, for a range of [-4, 4], this
                                    # will terminate the sequence at the first
                                    # frame that has all values < -3.4

# Training
tts_schedule = [(7,  1e-3,  10_000,  20),   # progressive training schedule
                (5,  1e-4, 100_000,  20),   # (r, lr, step, batch_size)
                (2,  1e-4, 180_000,  16),
                (2,  1e-4, 350_000,  8)]

tts_max_mel_len = 900               # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_eval_interval = 2000            # evaluates the model every X steps:
                                    #     if X = 0, evaluates every epoch
                                    #     if X < 0, never evaluates
tts_eval_num_samples = 1            # makes this number of samples

# Data Preprocessing
tts_rescale = True
tts_rescaling_max = 0.9
preemphasize=True                  
preemphasis=0.97                    # filter coefficient to use if preemphasize is True

# Mel visualization and Griffin-Lim
signal_normalization=True           
allow_clipping_in_normalization=True# Only relevant if mel_normalization = True
power = 1.5
griffin_lim_iters=60
# ------------------------------------------------------------------------------------------------------------------#

### SV2TTS
tts_speaker_embedding_size = 256    # embedding dimension for the speaker embedding
tts_silence_min_duration_split=0.4, # Duration in seconds of a silence for an utterance to be split
tts_utterance_min_duration=1.6,     # Duration in seconds below which utterances are discarded
