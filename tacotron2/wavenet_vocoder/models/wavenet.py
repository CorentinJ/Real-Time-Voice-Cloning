import numpy as np
import tensorflow as tf
from datasets import audio
from infolog import log
from wavenet_vocoder import util
from wavenet_vocoder.util import *

from .gaussian import sample_from_gaussian
from .mixture import sample_from_discretized_mix_logistic
from .modules import (Conv1D1x1, ConvTranspose2D, ConvTranspose1D, DiscretizedMixtureLogisticLoss, Embedding, GaussianMaximumLikelihoodEstimation,
						LeakyReluActivation, MaskedCrossEntropyLoss, ReluActivation, ResidualConv1DGLU, WeightNorm)


def _expand_global_features(batch_size, time_length, global_features, data_format='BCT'):
	"""Expand global conditioning features to all time steps

	Args:
		batch_size: int
		time_length: int
		global_features: Tensor of shape [batch_size, channels] or [batch_size, channels, 1]
		data_format: string, 'BCT' to get output of shape [batch_size, channels, time_length]
			or 'BTC' to get output of shape [batch_size, time_length, channels]

	Returns:
		None or Tensor of shape [batch_size, channels, time_length] or [batch_size, time_length, channels]
	"""
	accepted_formats = ['BCT', 'BTC']
	if not (data_format in accepted_formats):
		raise ValueError('{} is an unknow data format, accepted formats are "BCT" and "BTC"'.format(data_format))

	if global_features is None:
		return None

	#[batch_size, channels] ==> [batch_size, channels, 1]
	# g = tf.cond(tf.equal(tf.rank(global_features), 2),
	# 	lambda: tf.expand_dims(global_features, axis=-1),
	# 	lambda: global_features)
	g = tf.reshape(global_features, [tf.shape(global_features)[0], tf.shape(global_features)[1], 1])
	g_shape = tf.shape(g)

	#[batch_size, channels, 1] ==> [batch_size, channels, time_length]
	# ones = tf.ones([g_shape[0], g_shape[1], time_length], tf.int32)
	# g = g * ones
	g = tf.tile(g, [1, 1, time_length])

	if data_format == 'BCT':
		return g

	else:
		#[batch_size, channels, time_length] ==> [batch_size, time_length, channels]
		return tf.transpose(g, [0, 2, 1])


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda x: 2**x):
	"""Compute receptive field size.

	Args:
		total_layers; int
		num_cycles: int
		kernel_size: int
		dilation: callable, function used to compute dilation factor.
			use "lambda x: 1" to disable dilated convolutions.

	Returns:
		int: receptive field size in sample.
	"""
	assert total_layers % num_cycles == 0

	layers_per_cycle = total_layers // num_cycles
	dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
	return (kernel_size - 1) * sum(dilations) + 1

def maybe_Normalize_weights(layer, weight_normalization=True, init=False, init_scale=1.):
	"""Maybe Wraps layer with Weight Normalization wrapper.

	Args;
		layer: tf layers instance, the layer candidate for normalization
		weight_normalization: Boolean, determines whether to normalize the layer
		init: Boolean, determines if the current run is the data dependent initialization run
		init_scale: Float, Initialisation scale of the data dependent initialization. Usually 1.
	"""
	if weight_normalization:
		return WeightNorm(layer, init, init_scale)
	return layer

class WaveNet():
	"""Tacotron-2 Wavenet Vocoder model.
	"""
	def __init__(self, hparams, init):
		#Get hparams
		self._hparams = hparams

		if self.local_conditioning_enabled():
			assert hparams.num_mels == hparams.cin_channels

		#Initialize model architecture
		assert hparams.layers % hparams.stacks == 0
		layers_per_stack = hparams.layers // hparams.stacks

		self.scalar_input = is_scalar_input(hparams.input_type)

		#first (embedding) convolution
		with tf.variable_scope('input_convolution'):
			if self.scalar_input:
				self.first_conv = Conv1D1x1(hparams.residual_channels, 
					weight_normalization=hparams.wavenet_weight_normalization, 
					weight_normalization_init=init, 
					weight_normalization_init_scale=hparams.wavenet_init_scale,
					name='input_convolution')
			else:
				self.first_conv = Conv1D1x1(hparams.residual_channels, 
					weight_normalization=hparams.wavenet_weight_normalization, 
					weight_normalization_init=init, 
					weight_normalization_init_scale=hparams.wavenet_init_scale,
					name='input_convolution')

		#Residual Blocks
		self.residual_layers = []
		for layer in range(hparams.layers):
			self.residual_layers.append(ResidualConv1DGLU(
			hparams.residual_channels, hparams.gate_channels,
			kernel_size=hparams.kernel_size,
			skip_out_channels=hparams.skip_out_channels,
			use_bias=hparams.use_bias,
			dilation_rate=2**(layer % layers_per_stack),
			dropout=hparams.wavenet_dropout,
			cin_channels=hparams.cin_channels,
			gin_channels=hparams.gin_channels,
			weight_normalization=hparams.wavenet_weight_normalization, 
			init=init, 
			init_scale=hparams.wavenet_init_scale,
			name='ResidualConv1DGLU_{}'.format(layer)))

		#Final (skip) convolutions
		with tf.variable_scope('skip_convolutions'):
			self.last_conv_layers = [
			ReluActivation(name='final_conv_relu1'),
			Conv1D1x1(hparams.skip_out_channels, 
				weight_normalization=hparams.wavenet_weight_normalization, 
				weight_normalization_init=init, 
				weight_normalization_init_scale=hparams.wavenet_init_scale,
				name='final_convolution_1'), 
			ReluActivation(name='final_conv_relu2'),
			Conv1D1x1(hparams.out_channels, 
				weight_normalization=hparams.wavenet_weight_normalization, 
				weight_normalization_init=init, 
				weight_normalization_init_scale=hparams.wavenet_init_scale,
				name='final_convolution_2'),]

		#Global conditionning embedding
		if hparams.gin_channels > 0 and hparams.use_speaker_embedding:
			assert hparams.n_speakers is not None
			self.embed_speakers = Embedding(
				hparams.n_speakers, hparams.gin_channels, std=0.1, name='gc_embedding')
		else:
			self.embed_speakers = None

		self.all_convs = [self.first_conv] + self.residual_layers + self.last_conv_layers

		#Upsample conv net
		if hparams.upsample_conditional_features:
			self.upsample_conv = []
			for i, s in enumerate(hparams.upsample_scales):
				with tf.variable_scope('local_conditioning_upsampling_{}'.format(i+1)):
					if hparams.upsample_type == '2D':
						convt = ConvTranspose2D(1, (hparams.freq_axis_kernel_size, 2*s),
							padding='same', strides=(1, s))
					else:
						assert hparams.upsample_type == '1D'
						convt = ConvTranspose1D(hparams.cin_channels, (2*s, ),
							padding='same', strides=(s, ))

					self.upsample_conv.append(maybe_Normalize_weights(convt, 
						hparams.wavenet_weight_normalization, init, hparams.wavenet_init_scale))

					if hparams.upsample_activation == 'LeakyRelu':
						self.upsample_conv.append(LeakyReluActivation(alpha=hparams.leaky_alpha,
							name='upsample_leaky_relu_{}'.format(i+1)))
					elif hparams.upsample_activation == 'Relu':
						self.upsample_conv.append(ReluActivation(name='upsample_relu_{}'.format(i+1)))
					else:
						assert hparams.upsample_activation == None

			self.all_convs += self.upsample_conv
		else:
			self.upsample_conv = None

		self.receptive_field = receptive_field_size(hparams.layers,
			hparams.stacks, hparams.kernel_size)


	def set_mode(self, is_training):
		for conv in self.all_convs:
			try:
				conv.set_mode(is_training)
			except AttributeError:
				pass

	def initialize(self, y, c, g, input_lengths, x=None, synthesis_length=None):
		'''Initialize wavenet graph for train, eval and test cases.
		'''
		hparams = self._hparams
		self.is_training = x is not None
		self.is_evaluating = not self.is_training and y is not None
		#Set all convolutions to corresponding mode
		self.set_mode(self.is_training)

		log('Initializing Wavenet model.  Dimensions (? = dynamic shape): ')
		log('  Train mode:                {}'.format(self.is_training))
		log('  Eval mode:                 {}'.format(self.is_evaluating))
		log('  Synthesis mode:            {}'.format(not (self.is_training or self.is_evaluating)))
		with tf.variable_scope('inference') as scope:
			#Training
			if self.is_training:
				batch_size = tf.shape(x)[0]
				#[batch_size, time_length, 1]
				self.mask = self.get_mask(input_lengths, maxlen=tf.shape(x)[-1]) #To be used in loss computation
				#[batch_size, channels, time_length]
				y_hat = self.step(x, c, g, softmax=False) #softmax is automatically computed inside softmax_cross_entropy if needed

				if is_mulaw_quantize(hparams.input_type):
					#[batch_size, time_length, channels]
					self.y_hat_q = tf.transpose(y_hat, [0, 2, 1])

				self.y_hat = y_hat
				self.y = y
				self.input_lengths = input_lengths

				#Add mean and scale stats if using Guassian distribution output (there would be too many logistics if using MoL)
				if self._hparams.out_channels == 2:
					self.means = self.y_hat[:, 0, :]
					self.log_scales = self.y_hat[:, 1, :]
				else:
					self.means = None

				#Graph extension for log saving
				#[batch_size, time_length]
				shape_control = (batch_size, tf.shape(x)[-1], 1)
				with tf.control_dependencies([tf.assert_equal(tf.shape(y), shape_control)]):
					y_log = tf.squeeze(y, [-1])
					if is_mulaw_quantize(hparams.input_type):
						self.y = y_log

				y_hat_log = tf.cond(tf.equal(tf.rank(y_hat), 4),
					lambda: tf.squeeze(y_hat, [-1]),
					lambda: y_hat)
				y_hat_log = tf.reshape(y_hat_log, [batch_size, hparams.out_channels, -1])

				if is_mulaw_quantize(hparams.input_type):
					#[batch_size, time_length]
					y_hat_log = tf.argmax(tf.nn.softmax(y_hat_log, axis=1), 1)

					y_hat_log = util.inv_mulaw_quantize(y_hat_log, hparams.quantize_channels)
					y_log = util.inv_mulaw_quantize(y_log, hparams.quantize_channels)

				else:
					#[batch_size, time_length]
					if hparams.out_channels == 2:
						y_hat_log = sample_from_gaussian(
							y_hat_log, log_scale_min_gauss=hparams.log_scale_min_gauss)
					else:
						y_hat_log = sample_from_discretized_mix_logistic(
							y_hat_log, log_scale_min=hparams.log_scale_min)

					if is_mulaw(hparams.input_type):
						y_hat_log = util.inv_mulaw(y_hat_log, hparams.quantize_channels)
						y_log = util.inv_mulaw(y_log, hparams.quantize_channels)

				self.y_hat_log = y_hat_log
				self.y_log = y_log

				log('  inputs:                    {}'.format(x.shape))
				if self.local_conditioning_enabled():
					log('  local_condition:           {}'.format(c.shape))
				if self.has_speaker_embedding():
					log('  global_condition:          {}'.format(g.shape))
				log('  targets:                   {}'.format(y_log.shape))
				log('  outputs:                   {}'.format(y_hat_log.shape))


			#evaluating
			elif self.is_evaluating:
				#[time_length, ]
				idx = 0
				length = input_lengths[idx]
				y_target = tf.reshape(y[idx], [-1])[:length]

				if c is not None:
					c = tf.expand_dims(c[idx, :, :length], axis=0)
					with tf.control_dependencies([tf.assert_equal(tf.rank(c), 3)]):
						c = tf.identity(c, name='eval_assert_c_rank_op')
				if g is not None:
					g = tf.expand_dims(g[idx], axis=0)

				batch_size = tf.shape(c)[0]

				#Start silence frame
				if is_mulaw_quantize(hparams.input_type):
					initial_value = mulaw_quantize(0, hparams.quantize_channels)
				elif is_mulaw(hparams.input_type):
					initial_value = mulaw(0.0, hparams.quantize_channels)
				else:
					initial_value = 0.0

				#[channels, ]
				if is_mulaw_quantize(hparams.input_type):
					initial_input = tf.one_hot(indices=initial_value, depth=hparams.quantize_channels, dtype=tf.float32)
					initial_input = tf.tile(tf.reshape(initial_input, [1, 1, hparams.quantize_channels]), [batch_size, 1, 1])
				else:
					initial_input = tf.ones([batch_size, 1, 1], tf.float32) * initial_value

				#Fast eval
				y_hat = self.incremental(initial_input, c=c, g=g, time_length=length,
					softmax=False, quantize=True, log_scale_min=hparams.log_scale_min, log_scale_min_gauss=hparams.log_scale_min_gauss)

				#Save targets and length for eval loss computation
				if is_mulaw_quantize(hparams.input_type):
					self.y_eval = tf.reshape(y[idx], [1, -1])[:, :length]
				else:
					self.y_eval = tf.expand_dims(y[idx], axis=0)[:, :length, :]
				self.eval_length = length

				if is_mulaw_quantize(hparams.input_type):
					y_hat = tf.reshape(tf.argmax(y_hat, axis=1), [-1])
					y_hat = inv_mulaw_quantize(y_hat, hparams.quantize_channels)
					y_target = inv_mulaw_quantize(y_target, hparams.quantize_channels)
				elif is_mulaw(hparams.input_type):
					y_hat = inv_mulaw(tf.reshape(y_hat, [-1]), hparams.quantize_channels)
					y_target = inv_mulaw(y_target, hparams.quantize_channels)
				else:
					y_hat = tf.reshape(y_hat, [-1])

				self.y_hat = y_hat
				self.y_target = y_target

				if self.local_conditioning_enabled():
					log('  local_condition:           {}'.format(c.shape))
				if self.has_speaker_embedding():
					log('  global_condition:          {}'.format(g.shape))
				log('  targets:                   {}'.format(y_target.shape))
				log('  outputs:                   {}'.format(y_hat.shape))

			#synthesizing
			else:
				batch_size = tf.shape(c)[0]
				if c is None:
					assert synthesis_length is not None
				else:
					#[batch_size, local_condition_time, local_condition_dimension(num_mels)]
					message = ('Expected 3 dimension shape [batch_size(1), time_length, {}] for local condition features but found {}'.format(
							hparams.cin_channels, c.shape))
					with tf.control_dependencies([tf.assert_equal(tf.rank(c), 3, message=message)]):
						c = tf.identity(c, name='synthesis_assert_c_rank_op')

					Tc = tf.shape(c)[1]
					upsample_factor = audio.get_hop_size(self._hparams)

					#Overwrite length with respect to local condition features
					synthesis_length = Tc * upsample_factor

					#[batch_size, local_condition_dimension, local_condition_time]
					#time_length will be corrected using the upsample network
					c = tf.transpose(c, [0, 2, 1])

				if g is not None:
					assert g.shape == (batch_size, 1)

				#Start silence frame
				if is_mulaw_quantize(hparams.input_type):
					initial_value = mulaw_quantize(0, hparams.quantize_channels)
				elif is_mulaw(hparams.input_type):
					initial_value = mulaw(0.0, hparams.quantize_channels)
				else:
					initial_value = 0.0

				if is_mulaw_quantize(hparams.input_type):
					assert initial_value >= 0 and initial_value < hparams.quantize_channels
					initial_input = tf.one_hot(indices=initial_value, depth=hparams.quantize_channels, dtype=tf.float32)
					initial_input = tf.tile(tf.reshape(initial_input, [1, 1, hparams.quantize_channels]), [batch_size, 1, 1])
				else:
					initial_input = tf.ones([batch_size, 1, 1], tf.float32) * initial_value

				y_hat = self.incremental(initial_input, c=c, g=g, time_length=synthesis_length,
					softmax=False, quantize=True, log_scale_min=hparams.log_scale_min, log_scale_min_gauss=hparams.log_scale_min_gauss)

				if is_mulaw_quantize(hparams.input_type):
					y_hat = tf.reshape(tf.argmax(y_hat, axis=1), [batch_size, -1])
					y_hat = util.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
				elif is_mulaw(hparams.input_type):
					y_hat = util.inv_mulaw(tf.reshape(y_hat, [batch_size, -1]), hparams.quantize_channels)
				else:
					y_hat = tf.reshape(y_hat, [batch_size, -1])

				self.y_hat = y_hat

				if self.local_conditioning_enabled():
					log('  local_condition:           {}'.format(c.shape))
				if self.has_speaker_embedding():
					log('  global_condition:          {}'.format(g.shape))
				log('  outputs:                   {}'.format(y_hat.shape))

		self.variables = tf.trainable_variables()
		n_vars = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
		log('  Receptive Field:           ({} samples / {:.1f} ms)'.format(self.receptive_field, self.receptive_field / hparams.sample_rate * 1000.))

		#1_000_000 is causing syntax problems for some people?! Python please :)
		log('  WaveNet Parameters:        {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.variables]) / 1000000))

		self.ema = tf.train.ExponentialMovingAverage(decay=hparams.wavenet_ema_decay)


	def add_loss(self):
		'''Adds loss computation to the graph. Supposes that initialize function has already been called.
		'''
		with tf.variable_scope('loss') as scope:
			if self.is_training:
				if is_mulaw_quantize(self._hparams.input_type):
					self.loss = MaskedCrossEntropyLoss(self.y_hat_q[:, :-1, :], self.y[:, 1:], mask=self.mask)
				else:
					if self._hparams.out_channels == 2:
						self.loss = GaussianMaximumLikelihoodEstimation(self.y_hat[:, :, :-1], self.y[:, 1:, :], hparams=self._hparams, mask=self.mask)
					else:
						self.loss = DiscretizedMixtureLogisticLoss(self.y_hat[:, :, :-1], self.y[:, 1:, :], hparams=self._hparams, mask=self.mask)
			elif self.is_evaluating:
				if is_mulaw_quantize(self._hparams.input_type):
					self.eval_loss = MaskedCrossEntropyLoss(self.y_hat_eval, self.y_eval, lengths=[self.eval_length])
				else:
					if self._hparams.out_channels == 2:
						self.eval_loss = GaussianMaximumLikelihoodEstimation(self.y_hat_eval, self.y_eval, hparams=self._hparams, lengths=[self.eval_length])
					else:
						self.eval_loss = DiscretizedMixtureLogisticLoss(self.y_hat_eval, self.y_eval, hparams=self._hparams, lengths=[self.eval_length])


	def add_optimizer(self, global_step):
		'''Adds optimizer to the graph. Supposes that initialize function has already been called.
		'''
		with tf.variable_scope('optimizer'):
			hp = self._hparams

			#Create lr schedule
			if hp.wavenet_lr_schedule == 'noam':
				learning_rate = self._noam_learning_rate_decay(hp.wavenet_learning_rate, 
					global_step,
					warmup_steps=hp.wavenet_warmup)
			else:
				assert hp.wavenet_lr_schedule == 'exponential'
				learning_rate = self._exponential_learning_rate_decay(hp.wavenet_learning_rate,
					global_step,
					hp.wavenet_decay_rate,
					hp.wavenet_decay_steps)

			#Adam optimization
			self.learning_rate = learning_rate
			optimizer = tf.train.AdamOptimizer(learning_rate, hp.wavenet_adam_beta1,
				hp.wavenet_adam_beta2, hp.wavenet_adam_epsilon)

			gradients, variables = zip(*optimizer.compute_gradients(self.loss))
			self.gradients = gradients

			#Gradients clipping
			if hp.wavenet_clip_gradients:
				clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
			else:
				clipped_gradients = gradients

			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
					global_step=global_step)

		#Add exponential moving average
		#https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
		#Use adam optimization process as a dependency
		with tf.control_dependencies([adam_optimize]):
			#Create the shadow variables and add ops to maintain moving averages
			#Also updates moving averages after each update step
			#This is the optimize call instead of traditional adam_optimize one.
			assert tuple(self.variables) == variables #Verify all trainable variables are being averaged
			self.optimize = self.ema.apply(variables)

	def _noam_learning_rate_decay(self, init_lr, global_step, warmup_steps=4000.0):
		# Noam scheme from tensor2tensor:
		step = tf.cast(global_step + 1, dtype=tf.float32)
		return tf.maximum(init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5), 1e-4)

	def _exponential_learning_rate_decay(self, init_lr, global_step,
							 decay_rate=0.5,
							 decay_steps=300000):
		#Compute natural exponential decay
		lr = tf.train.exponential_decay(init_lr,
			global_step,
			decay_steps,
			decay_rate,
			name='wavenet_lr_exponential_decay')
		return lr


	def get_mask(self, input_lengths, maxlen=None):
		expand = not is_mulaw_quantize(self._hparams.input_type)
		mask = sequence_mask(input_lengths, max_len=maxlen, expand=expand)

		if is_mulaw_quantize(self._hparams.input_type):
			return mask[:, 1:]
		return mask[:, 1:, :]

	#Sanity check functions
	def has_speaker_embedding(self):
		return self.embed_speakers is not None

	def local_conditioning_enabled(self):
		return self._hparams.cin_channels > 0

	def step(self, x, c=None, g=None, softmax=False):
		"""Forward step

		Args:
			x: Tensor of shape [batch_size, channels, time_length], One-hot encoded audio signal.
			c: Tensor of shape [batch_size, cin_channels, time_length], Local conditioning features.
			g: Tensor of shape [batch_size, gin_channels, 1] or Ids of shape [batch_size, 1],
				Global conditioning features.
				Note: set hparams.use_speaker_embedding to False to disable embedding layer and
				use extrnal One-hot encoded features.
			softmax: Boolean, Whether to apply softmax.

		Returns:
			a Tensor of shape [batch_size, out_channels, time_length]
		"""
		#[batch_size, channels, time_length] -> [batch_size, time_length, channels]
		batch_size = tf.shape(x)[0]
		time_length = tf.shape(x)[-1]

		if g is not None:
			if self.embed_speakers is not None:
				#[batch_size, 1] ==> [batch_size, 1, gin_channels]
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, gin_channels, 1]
				with tf.control_dependencies([tf.assert_equal(tf.rank(g), 3)]):
					g = tf.transpose(g, [0, 2, 1])

		#Expand global conditioning features to all time steps
		g_bct = _expand_global_features(batch_size, time_length, g, data_format='BCT')

		if c is not None and self.upsample_conv is not None:
			if self._hparams.upsample_type == '2D':
				#[batch_size, 1, cin_channels, time_length]
				expand_dim = 1
			else:
				assert self._hparams.upsample_type == '1D'
				#[batch_size, cin_channels, 1, time_length]
				expand_dim = 2

			c = tf.expand_dims(c, axis=expand_dim)

			for transposed_conv in self.upsample_conv:
				c = transposed_conv(c)

			#[batch_size, cin_channels, time_length]
			c = tf.squeeze(c, [expand_dim])
			with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], tf.shape(x)[-1])]):
				c = tf.identity(c, name='control_c_and_x_shape')

		#Feed data to network
		x = self.first_conv(x)
		skips = None
		for conv in self.residual_layers:
			x, h = conv(x, c, g_bct)
			if skips is None:
				skips = h
			else:
				skips = skips + h
		x = skips

		for conv in self.last_conv_layers:
			x = conv(x)

		return tf.nn.softmax(x, axis=1) if softmax else x


	def incremental(self, initial_input, c=None, g=None,
		time_length=100, test_inputs=None,
		softmax=True, quantize=True, log_scale_min=-7.0, log_scale_min_gauss=-7.0):
		"""Inceremental forward step

		Inputs of shape [batch_size, channels, time_length] are reshaped to [batch_size, time_length, channels]
		Input of each time step is of shape [batch_size, 1, channels]

		Args:
			Initial input: Tensor of shape [batch_size, channels, 1], initial recurrence input.
			c: Tensor of shape [batch_size, cin_channels, time_length], Local conditioning features
			g: Tensor of shape [batch_size, gin_channels, time_length] or [batch_size, gin_channels, 1]
				global conditioning features
			T: int, number of timesteps to generate
			test_inputs: Tensor, teacher forcing inputs (debug)
			softmax: Boolean, whether to apply softmax activation
			quantize: Whether to quantize softmax output before feeding to
				next time step input
			log_scale_min: float, log scale minimum value.

		Returns:
			Tensor of shape [batch_size, channels, time_length] or [batch_size, channels, 1]
				Generated one_hot encoded samples
		"""
		batch_size = tf.shape(initial_input)[0]

		#Note: should reshape to [batch_size, time_length, channels]
		#not [batch_size, channels, time_length]
		if test_inputs is not None:
			if self.scalar_input:
				if tf.shape(test_inputs)[1] == 1:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])
			else:
				if tf.shape(test_inputs)[1] == self._hparams.out_channels:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])

			batch_size = tf.shape(test_inputs)[0]
			if time_length is None:
				time_length = tf.shape(test_inputs)[1]
			else:
				time_length = tf.maximum(time_length, tf.shape(test_inputs)[1])

		#Global conditioning
		if g is not None:
			if self.embed_speakers is not None:
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, channels, 1]
				with tf.control_dependencies([tf.assert_equal(tf.rank(g), 3)]):
					g = tf.transpose(g, [0, 2, 1])

		self.g_btc = _expand_global_features(batch_size, time_length, g, data_format='BTC')

		#Local conditioning
		if c is not None and self.upsample_conv is not None:
			if self._hparams.upsample_type == '2D':
				#[batch_size, 1, cin_channels, time_length]
				expand_dim = 1
			else:
				assert self._hparams.upsample_type == '1D'
				#[batch_size, cin_channels, 1, time_length]
				expand_dim = 2

			c = tf.expand_dims(c, axis=expand_dim)

			for upsample_conv in self.upsample_conv:
				c = upsample_conv(c)

			#[batch_size, channels, time_length]
			c = tf.squeeze(c, [expand_dim])
			with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], time_length)]):
				self.c = tf.transpose(c, [0, 2, 1])

		#Initialize loop variables
		if initial_input.shape[1] == self._hparams.out_channels:
			initial_input = tf.transpose(initial_input, [0, 2, 1])

		initial_time = tf.constant(0, dtype=tf.int32)
		if test_inputs is not None:
			initial_input = tf.expand_dims(test_inputs[:, 0, :], axis=1)
		initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		initial_loss_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		#Only use convolutions queues for Residual Blocks main convolutions (only ones with kernel size 3 and dilations, all others are 1x1)
		initial_queues = [tf.zeros((batch_size, res_conv.layer.kw + (res_conv.layer.kw - 1) * (res_conv.layer.dilation_rate[0] - 1), self._hparams.residual_channels),
			name='convolution_queue_{}'.format(i+1)) for i, res_conv in enumerate(self.residual_layers)]

		def condition(time, unused_outputs_ta, unused_current_input, unused_loss_outputs_ta, unused_queues):
			return tf.less(time, time_length)

		def body(time, outputs_ta, current_input, loss_outputs_ta, queues):
			#conditioning features for single time step
			ct = None if self.c is None else tf.expand_dims(self.c[:, time, :], axis=1)
			gt = None if self.g_btc is None else tf.expand_dims(self.g_btc[:, time, :], axis=1)

			x = self.first_conv.incremental_step(current_input)

			skips = None
			new_queues = []
			for conv, queue in zip(self.residual_layers, queues):
				x, h, new_queue = conv.incremental_step(x, ct, gt, queue=queue)
				skips = h if skips is None else (skips + h)
				new_queues.append(new_queue)
			x = skips

			for conv in self.last_conv_layers:
				try:
					x = conv.incremental_step(x)
				except AttributeError: #When calling Relu activation
					x = conv(x)

			#Save x for eval loss computation
			loss_outputs_ta = loss_outputs_ta.write(time, tf.squeeze(x, [1])) #squeeze time_length dimension (=1)

			#Generate next input by sampling
			if self.scalar_input:
				if self._hparams.out_channels == 2:
					x = sample_from_gaussian(
						tf.reshape(x, [batch_size, -1, 1]),
						log_scale_min_gauss=log_scale_min_gauss)
				else:
					x = sample_from_discretized_mix_logistic(
						tf.reshape(x, [batch_size, -1, 1]), log_scale_min=log_scale_min)

				next_input = tf.expand_dims(x, axis=-1) #Expand on the channels dimension
			else:
				x = tf.nn.softmax(tf.reshape(x, [batch_size, -1]), axis=1) if softmax \
					else tf.reshape(x, [batch_size, -1])
				if quantize:
					#[batch_size, 1]
					sample = tf.multinomial(x, 1) #Pick a sample using x as probability (one for each batche)
					#[batch_size, 1, quantize_channels] (time dimension extended by default)
					x = tf.one_hot(sample, depth=self._hparams.quantize_channels)

				next_input = x

			if len(x.shape) == 3:
				x = tf.squeeze(x, [1])

			outputs_ta = outputs_ta.write(time, x)
			time = tf.Print(time + 1, [time+1, time_length])
			#output = x (maybe next input)
			if test_inputs is not None:
				#override next_input with ground truth
				next_input = tf.expand_dims(test_inputs[:, time, :], axis=1)

			return (time, outputs_ta, next_input, loss_outputs_ta, new_queues)

		res = tf.while_loop(
			condition,
			body,
			loop_vars=[
				initial_time, initial_outputs_ta, initial_input, initial_loss_outputs_ta, initial_queues
			],
			parallel_iterations=32,
			swap_memory=self._hparams.wavenet_swap_with_cpu)

		outputs_ta = res[1]
		#[time_length, batch_size, channels]
		outputs = outputs_ta.stack()

		#Save eval prediction for eval loss computation
		eval_outputs = res[3].stack()

		if is_mulaw_quantize(self._hparams.input_type):
			self.y_hat_eval = tf.transpose(eval_outputs, [1, 0, 2])
		else:
			self.y_hat_eval = tf.transpose(eval_outputs, [1, 2, 0])

		#[batch_size, channels, time_length]
		return tf.transpose(outputs, [1, 2, 0])

	def clear_queue(self):
		self.first_conv.clear_queue()
		for f in self.conv_layers:
			f.clear_queue()
		for f in self.last_conv_layers:
			try:
				f.clear_queue()
			except AttributeError:
				pass


