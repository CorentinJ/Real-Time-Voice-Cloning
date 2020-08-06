
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = '/path/to/wav_files/'
data_path = 'data/'

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
voc_model_id = 'ljspeech_mol'
tts_model_id = 'ljspeech_lsa_smooth_attention'

# set this to True if you are only interested in WaveRNN
ignore_tts = False


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


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 32
voc_lr = 1e-4
voc_checkpoint_every = 25_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 1_000_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length
voc_clip_grad_norm = 4              # set to None if no gradient clipping needed

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


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
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes
tts_eval_interval = 0               # evaluates the model every X steps:
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
