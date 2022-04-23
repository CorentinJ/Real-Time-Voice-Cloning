import os
import sys
import time
from pathlib import Path

from nni.experiment import Experiment

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

params = {
    'tts_embed_dims': 512,
    'tts_encoder_dims': 256,
    'tts_decoder_dims': 128,
    'tts_postnet_dims': 512,
    'tts_lstm_dims': 1024,
}

search_space = {
    'tts_embed_dims': {'_type': 'choice', '_value': [256, 512, 768]},
    'tts_encoder_dims': {'_type': 'choice', '_value': [128, 256, 512]},
    'tts_decoder_dims': {'_type': 'choice', '_value': [64, 128, 256, 512]},
    'tts_postnet_dims': {'_type': 'choice', '_value': [256, 512, 768]},
    'tts_lstm_dims': {'_type': 'choice', '_value': [512, 768, 1024, 1280]},  # 1152
}
experiment = Experiment('local')

experiment.config.trial_code_directory = '.'
experiment.config.trial_command = "C:/tools/miniconda3/envs/torch310/python.exe synthesizer_train.py rusmodeltweaked " \
                                  "D:/projects/github/RUSLAN/SV2TTS/synthesizer --use_tweaked --perf_limit --n_epoch " \
                                  "2 --save_every 0 "
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 30
experiment.config.trial_gpu_number = 1
experiment.config.trial_concurrency = 1
experiment.config.training_service.use_active_gpu = True
experiment.run(8080)

input('Press enter to quit')
experiment.stop()
