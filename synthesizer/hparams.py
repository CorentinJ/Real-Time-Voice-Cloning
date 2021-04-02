import ast
import pprint
from synthesizer.utils.symbols import symbols


class HParams(object):
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self


hparams = HParams(
    # Signal Processing (used in both synthesizer and vocoder)
    sample_rate=24000,
    n_fft=2048,
    num_mels=128,
    # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    hop_size=300,
    # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    win_size=1200,
    fmin=20,
    min_level_db=-100,
    ref_level_db=20,
    # Gradient explodes if too big, premature convergence if too small.
    max_abs_value=4.,
    # Filter coefficient to use if preemphasize is True
    preemphasis=0.97,
    preemphasize=False,

    # Tacotron Text-to-Speech (TTS)
    # Embedding dimension for the graphemes/phoneme inputs
    tts_embed_dims=256,
    tts_encoder_dims=128,
    tts_decoder_dims=256,
    tts_postnet_dims=256,
    tts_encoder_K=16,
    tts_lstm_dims=512,
    tts_postnet_K=8,
    tts_num_highways=4,
    tts_dropout=0.5,
    tts_cleaner_names=["english_cleaners"],
    # Value below which audio generation ends.
    tts_stop_threshold=-3.4,
    # For example, for a range of [-4, 4], this
    # will terminate the sequence at the first
    # frame that has all values < -3.4

    n_symbols=len(symbols),
    symbols_embedding_dim=512,

    # Encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,

    # Decoder parameters
    n_frames_per_step=1,  # currently only 1 is supported
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=2000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # Attention parameters
    attention_rnn_dim=1024,
    attention_dim=128,
    decoder_lstm_dim=1024,

    # Location Layer parameters
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,

    pos_enc_dim=32,

    # Tacotron Training
    tts_schedule=[(1,  1e-3,  20_000,  12),   # Progressive training schedule
                  (1,  5e-4,  40_000,  12),   # (r, lr, step, batch_size)
                  (1,  2e-4,  80_000,  12),   #
                  # r = reduction factor (# of mel frames
                  (1,  1e-4, 160_000,  12),
                  # synthesized for each decoder iteration)
                  (1,  3e-5, 320_000,  12),
                  (1,  1e-5, 640_000,  12)],  # lr = learning rate

    # clips the gradient norm to prevent explosion - set to None if not needed
    tts_clip_grad_norm=1.0,
    # Number of steps between model evaluation (sample generation)
    tts_eval_interval=400,
    # Set to -1 to generate after completing epoch, or 0 to disable

    tts_eval_num_samples=3,                   # Makes this number of samples

    # Data Preprocessing
    max_mel_frames=900,
    rescale=False,
    rescaling_max=0.9,
    # For vocoder preprocessing and inference.
    synthesis_batch_size=16,

    # Mel Visualization and Griffin-Lim
    signal_normalization=False,
    power=1.5,
    griffin_lim_iters=60,

    # Audio processing options
    # Should not exceed (sample_rate // 2)
    fmax=12000,
    # Used when signal_normalization = True
    allow_clipping_in_normalization=True,
    # If true, discards samples exceeding max_mel_frames
    clip_mels_length=True,
    # "Fast spectrogram phase recovery using local weighted sums"
    use_lws=False,
    # Sets mel range to [-max_abs_value, max_abs_value] if True,
    symmetric_mels=True,
    #               and [0, max_abs_value] if False

    # SV2TTS
    speaker_embedding_size=256,               # Dimension for the speaker embedding
    # Duration in seconds of a silence for an utterance to be split
    silence_min_duration_split=0.4,
    # Duration in seconds below which utterances are discarded
    utterance_min_duration=1.6,


    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate=False,
    # learning_rate=1e-10,
    learning_rate=1e-3,
    weight_decay=1e-6,
    grad_clip_thresh=1.0,
    eps=1e-6,
    batch_size=2,
    mask_padding=True,  # set model's padded outputs to padded values
    lambda_dur=2,
    n_warm_up_step=4000,

    ###############################
    # Experiment Parameters        #
    ################################
    epochs=500,
    iters_per_checkpoint=1000,
    seed=1234,
    dynamic_loss_scaling=True,
    fp16_run=False,
    distributed_run=False,
    dist_backend="nccl",
    dist_url="tcp://localhost:54321",
    cudnn_enabled=True,
    cudnn_benchmark=False,
    cudnn_deterministic=True,
    ignore_layers=['embedding.weight'],
)


def hparams_debug_string():
    return str(hparams)
