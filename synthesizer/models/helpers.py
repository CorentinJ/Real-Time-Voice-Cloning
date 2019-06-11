import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


class TacoTestHelper(Helper):
	def __init__(self, batch_size, hparams):
		with tf.name_scope("TacoTestHelper"):
			self._batch_size = batch_size
			self._output_dim = hparams.num_mels
			self._reduction_factor = hparams.outputs_per_step
			self.stop_at_any = hparams.stop_at_any

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		"""Stop on EOS. Otherwise, pass the last output as the next input and pass through state."""
		with tf.name_scope("TacoTestHelper"):
			#A sequence is finished when the output probability is > 0.5
			finished = tf.cast(tf.round(stop_token_prediction), tf.bool)

			#Since we are predicting r frames at each step, two modes are
			#then possible:
			#	Stop when the model outputs a p > 0.5 for any frame between r frames (Recommended)
			#	Stop when the model outputs a p > 0.5 for all r frames (Safer)
			#Note:
			#	With enough training steps, the model should be able to predict when to stop correctly
			#	and the use of stop_at_any = True would be recommended. If however the model didn"t
			#	learn to stop correctly yet, (stops too soon) one could choose to use the safer option
			#	to get a correct synthesis
			if self.stop_at_any:
				finished = tf.reduce_any(tf.reduce_all(finished, axis=0)) #Recommended
			else:
				finished = tf.reduce_all(tf.reduce_all(finished, axis=0)) #Safer option

			# Feed last output frame as next input. outputs is [N, output_dim * r]
			next_inputs = outputs[:, -self._output_dim:]
			next_state = state
			return (finished, next_inputs, next_state)


class TacoTrainingHelper(Helper):
	def __init__(self, batch_size, targets, hparams, gta, evaluating, global_step):
		# inputs is [N, T_in], targets is [N, T_out, D]
		with tf.name_scope("TacoTrainingHelper"):
			self._batch_size = batch_size
			self._output_dim = hparams.num_mels
			self._reduction_factor = hparams.outputs_per_step
			self._ratio = tf.convert_to_tensor(hparams.tacotron_teacher_forcing_ratio)
			self.gta = gta
			self.eval = evaluating
			self._hparams = hparams
			self.global_step = global_step

			r = self._reduction_factor
			# Feed every r-th target frame as input
			self._targets = targets[:, r-1::r, :]

			#Maximal sequence length
			self._lengths = tf.tile([tf.shape(self._targets)[1]], [self._batch_size])

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		#Compute teacher forcing ratio for this global step.
		#In GTA mode, override teacher forcing scheme to work with full teacher forcing
		if self.gta:
			self._ratio = tf.convert_to_tensor(1.) #Force GTA model to always feed ground-truth
		elif self.eval and self._hparams.natural_eval:
			self._ratio = tf.convert_to_tensor(0.) #Force eval model to always feed predictions
		else:
			if self._hparams.tacotron_teacher_forcing_mode == "scheduled":
				self._ratio = _teacher_forcing_ratio_decay(self._hparams.tacotron_teacher_forcing_init_ratio,
					self.global_step, self._hparams)

		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		with tf.name_scope(name or "TacoTrainingHelper"):
			#synthesis stop (we let the model see paddings as we mask them when computing loss functions)
			finished = (time + 1 >= self._lengths)

			#Pick previous outputs randomly with respect to teacher forcing ratio
			next_inputs = tf.cond(
				tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
				lambda: self._targets[:, time, :], #Teacher-forcing: return true frame
				lambda: outputs[:,-self._output_dim:])

			#Pass on state
			next_state = state
			return (finished, next_inputs, next_state)


def _go_frames(batch_size, output_dim):
	"""Returns all-zero <GO> frames for a given batch size and output dimension"""
	return tf.tile([[0.0]], [batch_size, output_dim])

def _teacher_forcing_ratio_decay(init_tfr, global_step, hparams):
		#################################################################
		# Narrow Cosine Decay:

		# Phase 1: tfr = 1
		# We only start learning rate decay after 10k steps

		# Phase 2: tfr in ]0, 1[
		# decay reach minimal value at step ~280k

		# Phase 3: tfr = 0
		# clip by minimal teacher forcing ratio value (step >~ 280k)
		#################################################################
		#Compute natural cosine decay
		tfr = tf.train.cosine_decay(init_tfr,
			global_step=global_step - hparams.tacotron_teacher_forcing_start_decay, #tfr = 1 at step 10k
			decay_steps=hparams.tacotron_teacher_forcing_decay_steps, #tfr = 0 at step ~280k
			alpha=hparams.tacotron_teacher_forcing_decay_alpha, #tfr = 0% of init_tfr as final value
			name="tfr_cosine_decay")

		#force teacher forcing ratio to take initial value when global step < start decay step.
		narrow_tfr = tf.cond(
			tf.less(global_step, tf.convert_to_tensor(hparams.tacotron_teacher_forcing_start_decay)),
			lambda: tf.convert_to_tensor(init_tfr),
			lambda: tfr)

		return narrow_tfr