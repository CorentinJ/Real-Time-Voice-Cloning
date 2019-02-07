import os
import threading
import time

import numpy as np
import tensorflow as tf
from datasets import audio
from infolog import log
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from .util import is_mulaw_quantize, is_scalar_input



_batches_per_group = 32


class Feeder:
	"""
		Feeds batches of data into queue in a background thread.
	"""
	def __init__(self, coordinator, metadata_filename, base_dir, hparams):
		super(Feeder, self).__init__()

		self._coord = coordinator
		self._hparams = hparams
		self._train_offset = 0
		self._test_offset = 0

		if hparams.symmetric_mels:
			self._spec_pad = -hparams.max_abs_value
		else:
			self._spec_pad = 0.

		#Base directory of the project (to map files from different locations)
		self._base_dir = base_dir

		#Load metadata
		self._data_dir = os.path.dirname(metadata_filename)
		with open(metadata_filename, 'r') as f:
			self._metadata = [line.strip().split('|') for line in f]

		#Train test split
		if hparams.wavenet_test_size is None:
			assert hparams.wavenet_test_batches is not None

		test_size = (hparams.wavenet_test_size if hparams.wavenet_test_size is not None
			else hparams.wavenet_test_batches * hparams.wavenet_batch_size)
		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.wavenet_data_random_state)

		#Make sure test size is a multiple of batch size else round up
		len_test_indices = _round_down(len(test_indices), hparams.wavenet_batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		self.test_steps = len(self._test_meta) // hparams.wavenet_batch_size

		if hparams.wavenet_test_size is None:
			assert hparams.wavenet_test_batches == self.test_steps

		#Get conditioning status
		self.local_condition, self.global_condition = self._check_conditions()

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			if is_scalar_input(hparams.input_type):
				input_placeholder = tf.placeholder(tf.float32, shape=(None, 1, None), name='audio_inputs')
				target_placeholder = tf.placeholder(tf.float32, shape=(None, None, 1), name='audio_targets')
				target_type = tf.float32
			else:
				input_placeholder = tf.placeholder(tf.float32, shape=(None, hparams.quantize_channels, None), name='audio_inputs')
				target_placeholder = tf.placeholder(tf.int32, shape=(None, None, 1), name='audio_targets')
				target_type = tf.int32

			self._placeholders = [
			input_placeholder,
			target_placeholder,
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			]

			queue_types = [tf.float32, target_type, tf.int32]

			if self.local_condition:
				self._placeholders.append(tf.placeholder(tf.float32, shape=(None, hparams.num_mels, None), name='local_condition_features'))
				queue_types.append(tf.float32)
			if self.global_condition:
				self._placeholders.append(tf.placeholder(tf.int32, shape=(None, 1), name='global_condition_features'))
				queue_types.append(tf.int32)

			# Create queue for buffering data
			queue = tf.FIFOQueue(8, queue_types, name='intput_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			variables = queue.dequeue()

			self.inputs = variables[0]
			self.inputs.set_shape(self._placeholders[0].shape)
			self.targets = variables[1]
			self.targets.set_shape(self._placeholders[1].shape)
			self.input_lengths = variables[2]
			self.input_lengths.set_shape(self._placeholders[2].shape)

			#If local conditioning disabled override c inputs with None
			if hparams.cin_channels < 0:
				self.local_condition_features = None
			else:
				self.local_condition_features = variables[3]
				self.local_condition_features.set_shape(self._placeholders[3].shape)

			#If global conditioning disabled override g inputs with None
			if hparams.gin_channels < 0:
				self.global_condition_features = None
			else:
				self.global_condition_features = variables[4]
				self.global_condition_features.set_shape(self._placeholders[4].shape)


			# Create queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, queue_types, name='eval_queue')
			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
			eval_variables = eval_queue.dequeue()

			self.eval_inputs = eval_variables[0]
			self.eval_inputs.set_shape(self._placeholders[0].shape)
			self.eval_targets = eval_variables[1]
			self.eval_targets.set_shape(self._placeholders[1].shape)
			self.eval_input_lengths = eval_variables[2]
			self.eval_input_lengths.set_shape(self._placeholders[2].shape)

			#If local conditioning disabled override c inputs with None
			if hparams.cin_channels < 0:
				self.eval_local_condition_features = None
			else:
				self.eval_local_condition_features = eval_variables[3]
				self.eval_local_condition_features.set_shape(self._placeholders[3].shape)

			#If global conditioning disabled override g inputs with None
			if hparams.gin_channels < 0:
				self.eval_global_condition_features = None
			else:
				self.eval_global_condition_features = eval_variables[4]
				self.eval_global_condition_features.set_shape(self._placeholders[4].shape)



	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self):
		meta = self._test_meta[self._test_offset]
		self._test_offset += 1

		if self._hparams.train_with_GTA:
			mel_file = meta[2]
		else:
			mel_file = meta[1]
		audio_file = meta[0]

		input_data = np.load(os.path.join(self._base_dir, audio_file))

		if self.local_condition:
			local_condition_features = np.load(os.path.join(self._base_dir, mel_file))
		else:
			local_condition_features = None

		if self.global_condition:
			global_condition_features = meta[3]
			if global_condition_features == '<no_g>':
				raise RuntimeError('Please redo the wavenet preprocessing (or GTA synthesis) to assign global condition features!')
		else:
			global_condition_features = None

		return (input_data, local_condition_features, global_condition_features, len(input_data))

	def make_test_batches(self):
		start = time.time()

		#Read one example for evaluation
		n = 1

		#Test on entire test set (one sample at an evaluation step)
		examples = [self._get_test_groups() for i in range(len(self._test_meta))]
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)

		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			n = self._hparams.wavenet_batch_size
			examples = [self._get_next_example() for i in range(n * _batches_per_group)]

			# Bucket examples base on similiar output length for efficiency
			examples.sort(key=lambda x: x[-1])
			batches = [examples[i: i+n] for i in range(0, len(examples), n)]
			np.random.shuffle(batches)

			log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		test_batches = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		'''Get a single example (input, output, len_output) from disk
		'''
		if self._train_offset >= len(self._train_meta):
			self._train_offset = 0
			np.random.shuffle(self._train_meta)
		meta = self._train_meta[self._train_offset]
		self._train_offset += 1

		if self._hparams.train_with_GTA:
			mel_file = meta[2]
			if 'linear' in mel_file:
				raise RuntimeError('Linear spectrogram files selected instead of GTA mels, did you specify the wrong metadata?')
		else:
			mel_file = meta[1]
		audio_file = meta[0]

		input_data = np.load(os.path.join(self._base_dir, audio_file))

		if self.local_condition:
			local_condition_features = np.load(os.path.join(self._base_dir, mel_file))
		else:
			local_condition_features = None

		if self.global_condition:
			global_condition_features = meta[3]
			if global_condition_features == '<no_g>':
				raise RuntimeError('Please redo the wavenet preprocessing (or GTA synthesis) to assign global condition features!')
		else:
			global_condition_features = None

		return (input_data, local_condition_features, global_condition_features, len(input_data))


	def _prepare_batch(self, batch):
		np.random.shuffle(batch)

		#Limit time steps to save GPU Memory usage
		max_time_steps = self._limit_time()
		#Adjust time resolution for upsampling
		batch = self._adjust_time_resolution(batch, self.local_condition, max_time_steps)

		#time lengths
		input_lengths = [len(x[0]) for x in batch]
		max_input_length = max(input_lengths)

		inputs = self._prepare_inputs([x[0] for x in batch], max_input_length)
		targets = self._prepare_targets([x[0] for x in batch], max_input_length)
		local_condition_features = self._prepare_local_conditions(self.local_condition, [x[1] for x in batch])
		global_condition_features = self._prepare_global_conditions(self.global_condition, [x[2] for x in batch])

		new_batch = (inputs, targets, input_lengths)
		if local_condition_features is not None:
			new_batch += (local_condition_features, )
		if global_condition_features is not None:
			new_batch += (global_condition_features, )

		return new_batch

	def _prepare_inputs(self, inputs, maxlen):
		if is_mulaw_quantize(self._hparams.input_type):
			#[batch_size, time_steps, quantize_channels]
			x_batch = np.stack([_pad_inputs(np_utils.to_categorical(
				x, num_classes=self._hparams.quantize_channels), maxlen) for x in inputs]).astype(np.float32)
		else:
			#[batch_size, time_steps, 1]
			x_batch = np.stack([_pad_inputs(x.reshape(-1, 1), maxlen) for x in inputs]).astype(np.float32)
		assert len(x_batch.shape) == 3
		#Convert to channels first [batch_size, quantize_channels (or 1), time_steps]
		x_batch = np.transpose(x_batch, (0, 2, 1))
		return x_batch

	def _prepare_targets(self, targets, maxlen):
		#[batch_size, time_steps]
		if is_mulaw_quantize(self._hparams.input_type):
			y_batch = np.stack([_pad_targets(x, maxlen) for x in targets]).astype(np.int32)
		else:
			y_batch = np.stack([_pad_targets(x, maxlen) for x in targets]).astype(np.float32)
		assert len(y_batch.shape) == 2
		#Add extra axis (make 3 dimension)
		y_batch = np.expand_dims(y_batch, axis=-1)
		return y_batch

	def _prepare_local_conditions(self, local_condition, c_features):
		if local_condition:

			maxlen = max([len(x) for x in c_features])
			#[-max, max] or [0,max]
			T2_output_range = (-self._hparams.max_abs_value, self._hparams.max_abs_value) if self._hparams.symmetric_mels else (0, self._hparams.max_abs_value)

			if self._hparams.clip_for_wavenet:
				c_features = [np.clip(x, T2_output_range[0], T2_output_range[1]) for x in c_features]
				
			c_batch = np.stack([_pad_inputs(x, maxlen, _pad=T2_output_range[0]) for x in c_features]).astype(np.float32)
			assert len(c_batch.shape) == 3
			#[batch_size, c_channels, time_steps]
			c_batch = np.transpose(c_batch, (0, 2, 1))

			if self._hparams.normalize_for_wavenet:
				#rerange to [0, 1]
				c_batch = np.interp(c_batch, T2_output_range, (0, 1))
		else:
			c_batch = None
		return c_batch

	def _prepare_global_conditions(self, global_condition, g_features):
		if global_condition:
			g_batch = np.array(g_features).astype(np.int32).reshape(-1, 1)
		else:
			g_batch = None
		return g_batch

	def _check_conditions(self):
		local_condition = self._hparams.cin_channels > 0
		global_condition = self._hparams.gin_channels > 0
		return local_condition, global_condition

	def _limit_time(self):
		'''Limit time resolution to save GPU memory.
		'''
		if self._hparams.max_time_sec is not None:
			return int(self._hparams.max_time_sec * self._hparams.sample_rate)
		elif self._hparams.max_time_steps is not None:
			return self._hparams.max_time_steps
		else:
			return None

	def _adjust_time_resolution(self, batch, local_condition, max_time_steps):
		'''Adjust time resolution between audio and local condition
		'''
		if local_condition:
			new_batch = []
			for b in batch:
				x, c, g, l = b
				self._assert_ready_for_upsample(x, c)
				if max_time_steps is not None:
					max_steps = _ensure_divisible(max_time_steps, audio.get_hop_size(self._hparams), True)
					if len(x) > max_time_steps:
						max_time_frames = max_steps // audio.get_hop_size(self._hparams)
						start = np.random.randint(0, len(c) - max_time_frames)
						time_start = start * audio.get_hop_size(self._hparams)
						x = x[time_start: time_start + max_time_frames * audio.get_hop_size(self._hparams)]
						c = c[start: start + max_time_frames, :]
						self._assert_ready_for_upsample(x, c)

				new_batch.append((x, c, g, l))
			return new_batch
		else:
			new_batch = []
			for b in batch:
				x, c, g, l = b
				x = audio.trim_silence(x, hparams)
				if max_time_steps is not None and len(x) > max_time_steps:
					start = np.random.randint(0, len(c) - max_time_steps)
					x = x[start: start + max_time_steps]
				new_batch.append((x, c, g, l))
			return new_batch

	def _assert_ready_for_upsample(self, x, c):
		assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size(self._hparams)

def _pad_inputs(x, maxlen, _pad=0):
	return np.pad(x, [(0, maxlen - len(x)), (0, 0)], mode='constant', constant_values=_pad)

def _pad_targets(x, maxlen, _pad=0):
	return np.pad(x, (0, maxlen - len(x)), mode='constant', constant_values=_pad)

def _round_up(x, multiple):
	remainder = x % multiple
	return x if remainder == 0 else x + multiple - remainder

def _round_down(x, multiple):
	remainder = x % multiple
	return x if remainder == 0 else x - remainder

def _ensure_divisible(length, divisible_by=256, lower=True):
	if length % divisible_by == 0:
		return length
	if lower:
		return length - length % divisible_by
	else:
		return length + (divisible_by - length % divisible_by)
