import argparse
import os

import numpy as np
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tqdm import tqdm
from wavenet_vocoder.synthesizer import Synthesizer


def run_synthesis(args, checkpoint_path, output_dir, hparams):
	log_dir = os.path.join(output_dir, 'plots')
	wav_dir = os.path.join(output_dir, 'wavs')

	#We suppose user will provide correct folder depending on training method
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	if args.model == 'Tacotron-2':
		#If running all Tacotron-2, synthesize audio from evaluated mels
		metadata_filename = os.path.join(args.mels_dir, 'map.txt')
		with open(metadata_filename, encoding='utf-8') as f:
			metadata = np.array([line.strip().split('|') for line in f])

		speaker_ids = metadata[:, 2]
		mel_files = metadata[:, 1]
		texts = metadata[:, 0]

		speaker_ids = None if (speaker_ids == '<no_g>').all() else speaker_ids
	else:
		#else Get all npy files in input_dir (supposing they are mels)
		mel_files  = sorted([os.path.join(args.mels_dir, f) for f in os.listdir(args.mels_dir) if f.split('.')[-1] == 'npy'])
		speaker_ids = None if args.speaker_id is None else args.speaker_id.replace(' ', '').split(',')
		if speaker_ids is not None:
			assert len(speaker_ids) == len(mel_files)

		texts = None

	log('Starting synthesis! (this will take a while..)')
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)

	mel_files = [mel_files[i: i+hparams.wavenet_synthesis_batch_size] for i in range(0, len(mel_files), hparams.wavenet_synthesis_batch_size)]
	speaker_ids = None if speaker_ids is None else [speaker_ids[i: i+hparams.wavenet_synthesis_batch_size] for i in range(0, len(speaker_ids), hparams.wavenet_synthesis_batch_size)]
	texts = None if texts is None else [texts[i: i+hparams.wavenet_synthesis_batch_size] for i in range(0, len(texts), hparams.wavenet_synthesis_batch_size)]

	with open(os.path.join(wav_dir, 'map.txt'), 'w') as file:
		for i, mel_batch in enumerate(tqdm(mel_files)):
			mel_spectros = [np.load(mel_file) for mel_file in mel_batch]

			basenames = [os.path.basename(mel_file).replace('.npy', '') for mel_file in mel_batch]
			speaker_id_batch = None if speaker_ids is None else speaker_ids[i]
			audio_files = synth.synthesize(mel_spectros, speaker_id_batch, basenames, wav_dir, log_dir)

			speaker_logs = ['<no_g>'] * len(mel_batch) if speaker_id_batch is None else speaker_id_batch

			for j, mel_file in enumerate(mel_batch):
				if texts is None:
					file.write('{}|{}\n'.format(mel_file, audio_files[j], speaker_logs[j]))
				else:
					file.write('{}|{}|{}\n'.format(texts[i][j], mel_file, audio_files[j], speaker_logs[j]))

	log('synthesized audio waveforms at {}'.format(wav_dir))



def wavenet_synthesize(args, hparams, checkpoint):
	output_dir = 'wavenet_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	run_synthesis(args, checkpoint_path, output_dir, hparams)
