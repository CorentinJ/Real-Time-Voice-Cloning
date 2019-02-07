import os
import wave
from datetime import datetime

import numpy as np
import pyaudio
import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence


class Synthesizer:
	def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
		log('Constructing model: %s' % model_name)
		#Force the batch size to be known in order to use attention masking in batch synthesis
		inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
		input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
		targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
		split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
		with tf.variable_scope('Tacotron_model') as scope:
			self.model = create_model(model_name, hparams)
			if gta:
				self.model.initialize(inputs, input_lengths, targets, gta=gta, split_infos=split_infos)
			else:
				self.model.initialize(inputs, input_lengths, split_infos=split_infos)

			self.mel_outputs = self.model.tower_mel_outputs
			self.linear_outputs = self.model.tower_linear_outputs if (hparams.predict_linear and not gta) else None
			self.alignments = self.model.tower_alignments
			self.stop_token_prediction = self.model.tower_stop_token_prediction
			self.targets = targets

		self.gta = gta
		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.

		self.inputs = inputs
		self.input_lengths = input_lengths
		self.targets = targets
		self.split_infos = split_infos

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPUs as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

		#Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
		while len(texts) % hparams.tacotron_synthesis_batch_size != 0:
			texts.append(texts[-1])
			basenames.append(basenames[-1])
			if mel_filenames is not None:
				mel_filenames.append(mel_filenames[-1])

		assert 0 == len(texts) % self._hparams.tacotron_num_gpus
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]

		size_per_device = len(seqs) // self._hparams.tacotron_num_gpus

		#Pad inputs according to each GPU max length
		input_seqs = None
		split_infos = []
		for i in range(self._hparams.tacotron_num_gpus):
			device_input = seqs[size_per_device*i: size_per_device*(i+1)]
			device_input, max_seq_len = self._prepare_inputs(device_input)
			input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
			split_infos.append([max_seq_len, 0, 0, 0])

		feed_dict = {
			self.inputs: input_seqs,
			self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		if self.gta:
			np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
			target_lengths = [len(np_target) for np_target in np_targets]

			#pad targets according to each GPU max length
			target_seqs = None
			for i in range(self._hparams.tacotron_num_gpus):
				device_target = np_targets[size_per_device*i: size_per_device*(i+1)]
				device_target, max_target_len = self._prepare_targets(device_target, self._hparams.outputs_per_step)
				target_seqs = np.concatenate((target_seqs, device_target), axis=1) if target_seqs is not None else device_target
				split_infos[i][1] = max_target_len #Not really used but setting it in case for future development maybe?

			feed_dict[self.targets] = target_seqs
			assert len(np_targets) == len(texts)

		feed_dict[self.split_infos] = np.asarray(split_infos, dtype=np.int32)

		if self.gta or not hparams.predict_linear:
			mels, alignments, stop_tokens = self.session.run([self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
			#Linearize outputs (1D arrays)
			mels = [mel for gpu_mels in mels for mel in gpu_mels]
			alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
			stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

			if not self.gta:
				#Natural batch synthesis
				#Get Mel lengths for the entire batch from stop_tokens predictions
				target_lengths = self._get_output_lengths(stop_tokens)

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			assert len(mels) == len(texts)

		else:
			linears, mels, alignments, stop_tokens = self.session.run([self.linear_outputs, self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)
			#Linearize outputs (1D arrays)
			linears = [linear for gpu_linear in linears for linear in gpu_linear]
			mels = [mel for gpu_mels in mels for mel in gpu_mels]
			alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
			stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

			#Natural batch synthesis
			#Get Mel/Linear lengths for the entire batch from stop_tokens predictions
			# target_lengths = self._get_output_lengths(stop_tokens)
			target_lengths = [9999]

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
			assert len(mels) == len(linears) == len(texts)


		if basenames is None:
			#Generate wav and read it
			wav = audio.inv_mel_spectrogram(mels.T, hparams)
			audio.save_wav(wav, 'temp.wav', sr=hparams.sample_rate) #Find a better way

			chunk = 512
			f = wave.open('temp.wav', 'rb')
			p = pyaudio.PyAudio()
			stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
				channels=f.getnchannels(),
				rate=f.getframerate(),
				output=True)
			data = f.readframes(chunk)
			while data:
				stream.write(data)
				data=f.readframes(chunk)

			stream.stop_stream()
			stream.close()

			p.terminate()
			return


		saved_mels_paths = []
		speaker_ids = []
		for i, mel in enumerate(mels):
			#Get speaker id for global conditioning (only used with GTA generally)
			if hparams.gin_channels > 0:
				raise RuntimeError('Please set the speaker_id rule in line 99 of tacotron/synthesizer.py to allow for global condition usage later.')
				speaker_id = '<no_g>' #set the rule to determine speaker id. By using the file basename maybe? (basenames are inside "basenames" variable)
				speaker_ids.append(speaker_id) #finish by appending the speaker id. (allows for different speakers per batch if your model is multispeaker)
			else:
				speaker_id = '<no_g>'
				speaker_ids.append(speaker_id)

			# Write the spectrogram to disk
			# Note: outputs mel-spectrogram files and target ones have same names, just different folders
			mel_filename = os.path.join(out_dir, 'mel-{}.npy'.format(basenames[i]))
			np.save(mel_filename, mel, allow_pickle=False)
			saved_mels_paths.append(mel_filename)

			if log_dir is not None:
				#save wav (mel -> wav)
				wav = audio.inv_mel_spectrogram(mel.T, hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])), sr=hparams.sample_rate)

				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}.png'.format(basenames[i])),
					title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

				#save mel spectrogram plot
				plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
					title='{}'.format(texts[i]), split_title=True)

				if hparams.predict_linear:
					#save wav (linear -> wav)
					wav = audio.inv_linear_spectrogram(linears[i].T, hparams)
					audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-linear.wav'.format(basenames[i])), sr=hparams.sample_rate)

					#save linear spectrogram plot
					plot.plot_spectrogram(linears[i], os.path.join(log_dir, 'plots/linear-{}.png'.format(basenames[i])),
						title='{}'.format(texts[i]), split_title=True, auto_aspect=True)



		return saved_mels_paths, speaker_ids

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) for row in np.round(stop_tokens).tolist()]
		return output_lengths
