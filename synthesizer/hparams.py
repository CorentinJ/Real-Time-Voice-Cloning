from tensorflow.contrib.training import HParams

# Default hyperparameters
hparams = HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
    cleaners="english_cleaners",
    
    # If you only have 1 GPU or want to use only one GPU, please set num_gpus=0 and specify the 
    # GPU idx on run. example:
    # expample 1 GPU of index 2 (train on "/gpu2" only): CUDA_VISIBLE_DEVICES=2 python train.py 
    # --model="Tacotron" --hparams="tacotron_gpu_start_idx=2"
    # If you want to train on multiple GPUs, simply specify the number of GPUs available, 
    # and the idx of the first GPU to use. example:
    # example 4 GPUs starting from index 0 (train on "/gpu0"->"/gpu3"): python train.py 
    # --model="Tacotron" --hparams="tacotron_num_gpus=4, tacotron_gpu_start_idx=0"
    # The hparams arguments can be directly modified on this hparams.py file instead of being 
    # specified on run if preferred!
    
    # If one wants to train both Tacotron and WaveNet in parallel (provided WaveNet will be 
    # trained on True mel spectrograms), one needs to specify different GPU idxes.
    # example Tacotron+WaveNet on a machine with 4 or plus GPUs. Two GPUs for each model: 
    # CUDA_VISIBLE_DEVICES=0,1 python train.py --model="Tacotron" 
	# --hparams="tacotron_gpu_start_idx=0, tacotron_num_gpus=2"
    # Cuda_VISIBLE_DEVICES=2,3 python train.py --model="WaveNet" 
	# --hparams="wavenet_gpu_start_idx=2; wavenet_num_gpus=2"
    
    # IMPORTANT NOTE: If using N GPUs, please multiply the tacotron_batch_size by N below in the 
    # hparams! (tacotron_batch_size = 32 * N)
    # Never use lower batch size than 32 on a single GPU!
    # Same applies for Wavenet: wavenet_batch_size = 8 * N (wavenet_batch_size can be smaller than
    #  8 if GPU is having OOM, minimum 2)
    # Please also apply the synthesis batch size modification likewise. (if N GPUs are used for 
    # synthesis, minimal batch size must be N, minimum of 1 sample per GPU)
    # We did not add an automatic multi-GPU batch size computation to avoid confusion in the 
    # user"s mind and to provide more control to the user for
    # resources related decisions.
    
    # Acknowledgement:
    #	Many thanks to @MlWoo for his awesome work on multi-GPU Tacotron which showed to work a 
	# little faster than the original
    #	pipeline for a single GPU as well. Great work!
    
    # Hardware setup: Default supposes user has only one GPU: "/gpu:0" (Tacotron only for now! 
    # WaveNet does not support multi GPU yet, WIP)
    # Synthesis also uses the following hardware parameters for multi-GPU parallel synthesis.
    tacotron_gpu_start_idx=0,  # idx of the first GPU to be used for Tacotron training.
    tacotron_num_gpus=1,  # Determines the number of gpus in use for Tacotron training.
    split_on_cpu=True,
    # Determines whether to split data on CPU or on first GPU. This is automatically True when 
	# more than 1 GPU is used.
    ###########################################################################################################################################
    
    # Audio
    # Audio parameters are the most important parameters to tune when using this work on your 
    # personal data. Below are the beginner steps to adapt
    # this work to your personal data:
    #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how 
	# many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
    #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, 
	# so there are plenty of examples to refer to)
    #	2- set sample_rate parameter to your data correct sample rate
    #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms 
	# window_size, and 12.5ms frame_shift(hop_size))
    #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
    #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto 
	# example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
    #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
    #		a- n_fft can be either equal to win_size or the first power of 2 that comes after 
	# win_size. I usually recommend using the latter
    #			to be more consistent with signal processing friends. No big difference to be seen
	#  however. For the tuto example: n_fft = 2048 = 2**11
    #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 
	# 1 = 1025.
    #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto 
	# example: upsample_scales=[15, 20] where 15 * 20 = 300
    #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only 
	# keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
    #			so the training segments should be long enough (2.8~3x upsample_scales[0] * 
	# hop_size or longer) so that the first kernel size can see the middle 
    #			of the samples efficiently. The length of WaveNet training segments is under the 
	# parameter "max_time_steps".
    #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying 
	# preprocessing (or part of it, ctrl-C to stop), then use the
    #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That 
	# will first give you some idea about your above parameters, and
    #		it will also give you an idea about trimming. If silences persist, try reducing 
	# trim_top_db slowly. If samples are trimmed mid words, try increasing it.
    #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are 
	# showing black silent regions on top), then restart from step 2.
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value
    # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    # train samples of lengths between 3sec and 14sec are more than enough to make a model capable
    # of good parallelization.
    clip_mels_length=True,
    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
	# also consider clipping your samples to smaller chunks)
    max_mel_frames=900,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
	#  and still getting OOM errors.
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    # Only used to set as True if using WaveNet, no difference in performance is observed in 
    # either cases.
    silence_threshold=2,  # silence threshold used for sound trimming for wavenet preprocessing
    
    # Mel spectrogram  
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # M-AILABS (and other datasets) trim params (these parameters are usually correct for any 
	# data, but definitely must be tuned for specific speakers)
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    normalize_for_wavenet=True,
    # whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet=True,
    # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    ###########################################################################################################################################
    
    # Tacotron
    outputs_per_step=2, # Was 1
    # number of frames to generate at each decoding step (increase to speed up computation and 
    # allows for higher batch size, decreases G&L audio quality)
    stop_at_any=True,
    # Determines whether the decoder should stop when predicting <stop> to any frame or to all of 
    # them (True works pretty well)
    
    embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)
    
    # Encoder parameters
    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=512,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=256,  # number of lstm units for each direction (forward and backward)
    
    # Attention mechanism
    smoothing=False,  # Whether to smooth the attention normalization function
    attention_dim=128,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31,),  # kernel size of attention convolution
    cumulative_weights=True,
    # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (
    # Recommended: True)
    
    # Decoder
    prenet_layers=[256, 256],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=1024,  # number of decoder lstm units on each layer
    max_iters=2000,
    # Max decoder steps during inference (Just for safety from infinite loop cases)
    
    # Residual postnet
    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=512,  # number of postnet convolution filters for each layer
    
    # CBHG mel->linear postnet
    cbhg_kernels=8,
    # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act
    #  as "K-grams"
    cbhg_conv_channels=128,  # Channels of the convolution bank
    cbhg_pool_size=2,  # pooling size of the CBHG
    cbhg_projection=256,
    # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
    cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
    cbhg_highwaynet_layers=4,  # Number of HighwayNet layers
    cbhg_highway_units=128,  # Number of units used in HighwayNet fully connected layers
    cbhg_rnn_units=128,
    # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in 
    # shape
    
    # Loss params
    mask_encoder=True,
    # whether to mask encoder padding while computing attention. Set to True for better prosody 
    # but slower convergence.
    mask_decoder=False,
    # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not
    #  be weighted, else recommended pos_weight = 20)
    cross_entropy_pos_weight=20,
    # Use class weights to reduce the stop token classes imbalance (by adding more penalty on 
    # False Negatives (FN)) (1 = disabled)
    predict_linear=False,
    # Whether to add a post-processing network to the Tacotron to predict linear spectrograms (
	# True mode Not tested!!)
    ###########################################################################################################################################

    # Tacotron Training
    # Reproduction seeds
    tacotron_random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    tacotron_data_random_state=1234,  # random state for train test split repeatability
    
    # performance parameters
    tacotron_swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause 
    # major slowdowns! Only use when critical!)
    
    # train/test split ratios, mini-batches sizes
    tacotron_batch_size=36,  # number of training samples on each training steps (was 32)
    # Tacotron Batch synthesis supports ~16x the training batch size (no gradients during 
    # testing). 
    # Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times
    #  different from training. We thus recommend masking the encoder.
    tacotron_synthesis_batch_size=128,
    # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN"T TRAIN TACOTRON WITH "mask_encoder=True"!!
    tacotron_test_size=0.05,
    # % of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is 
	# enough to have a good idea about overfit)
    tacotron_test_batches=None,  # number of test batches.
    
    # Learning rate schedule
    tacotron_decay_learning_rate=True,
    # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=50000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate
    
    # Optimization parameters
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter
    
    # Regularization parameters
    tacotron_reg_weight=1e-7,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=False,
    # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is
    #  high and biasing the model)
    tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
    tacotron_clip_gradients=True,  # whether to clip gradients
    
    # Evaluation parameters
    natural_eval=False,
    # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same
	#  teacher-forcing ratio as in training (just for overfit)
    
    # Decoder RNN learning can take be done in one of two ways:
    #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode="constant"
    #	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is 
    # function of global step. (teacher forcing ratio decay) mode="scheduled"
    # The second approach is inspired by:
    # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    tacotron_teacher_forcing_mode="constant",
    # Can be ("constant" or "scheduled"). "scheduled" mode applies a cosine teacher forcing ratio 
    # decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1.,
    # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder 
	# inputs, Only relevant if mode="constant"
    tacotron_teacher_forcing_init_ratio=1.,
    # initial teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_final_ratio=0.,
    # final teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_alpha=0.,
    # teacher forcing ratio decay rate. Relevant if mode="scheduled"
    ###########################################################################################################################################
 
    # Tacotron-2 integration parameters
    train_with_GTA=False,
    # Whether to use GTA mels to train WaveNet instead of ground truth mels.
    ###########################################################################################################################################
    
    # Eval sentences (if no eval text file was specified during synthesis, these sentences are 
	# used for eval)
    sentences=[
        # From July 8, 2017 New York Times:
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There\"s a way to measure the acute emotional intelligence that has never gone out of "
		"style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "The Senate\"s bill to repeal and replace the Affordable Care Act is now imperiled.",
        # From Google"s Tacotron example page:
        "Generative adversarial network or variational auto-encoder.",
        "Basilar membrane and otolaryngology are not auto-correlations.",
        "He has read the whole thing.",
        "He reads books.",
        "He thought it was time to present the present.",
        "Thisss isrealy awhsome.",
        "Punctuation sensitivity, is working.",
        "Punctuation sensitivity is working.",
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",
        # From The web (random long utterance)
        "Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
        This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
        the adopted architecture is able to perform this task with wild success.",
        "Thank you so much for your support!",
    ],
    
    
    ### SV2TTS ###
    speaker_embedding_size=256,
    silence_min_duration_split=0.4, # Duration in seconds of a silence for an utterance to be split
    utterance_min_duration=1.6,     # Duration in seconds below which utterances are discarded
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
