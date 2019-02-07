import numpy as np
import tensorflow as tf
from wavenet_vocoder.util import sequence_mask

from .gaussian import gaussian_maximum_likelihood_estimation_loss
from .mixture import discretized_mix_logistic_loss


class Embedding:
	"""Embedding class for global conditions.
	"""
	def __init__(self, num_embeddings, embedding_dim, std=0.1, name='gc_embedding'):
		#Create embedding table
		self.embedding_table = tf.get_variable(name,
			[num_embeddings, embedding_dim], dtype=tf.float32,
			initializer=tf.truncated_normal_initializer(mean=0., stddev=std))

	def __call__(self, inputs):
		#Do the actual embedding
		return tf.nn.embedding_lookup(self.embedding_table, inputs)

class ReluActivation:
	"""Simple class to wrap relu activation function in class for later call.
	"""
	def __init__(self, name=None):
		self.name = name

	def __call__(self, inputs):
		return tf.nn.relu(inputs, name=self.name)


class LeakyReluActivation:
	'''Simple class to wrap leaky relu activation function in class for later call.
	'''
	def __init__(self, alpha=0.3, name=None):
		self.alpha = alpha
		self.name = name

	def __call__(self, inputs):
		return tf.nn.leaky_relu(inputs, alpha=self.alpha, name=self.name)


class WeightNorm(tf.keras.layers.Wrapper):
	""" This wrapper reparameterizes a layer by decoupling the weight's
	magnitude and direction. This speeds up convergence by improving the
	conditioning of the optimization problem.
	Weight Normalization: A Simple Reparameterization to Accelerate
	Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
	Tim Salimans, Diederik P. Kingma (2016)
	WeightNorm wrapper works for tf layers.
	```python
		normed_layer = WeightNorm(tf.layers.Conv2D(2, 2, activation='relu'),
						input_shape=(32, 32, 3), data_init=True)

		output = normed_layer(input)
	```
	Arguments:
		layer: a layer instance.
		data_init: If `True` use data dependant variable initialization (Requires an initialization forward pass or behavior will be wrong)
	Raises:
		ValueError: If not initialized with a `Layer` instance.
		ValueError: If `Layer` does not contain a `kernel` of weights
		NotImplementedError: If `data_init` is True and running graph execution
	"""
	def __init__(self, layer, init=False, init_scale=1., **kwargs):
		if not isinstance(layer, tf.layers.Layer):
			raise ValueError(
					'Please initialize `WeightNorm` layer with a '
					'`Layer` instance. You passed: {input}'.format(input=layer))

		self.init = init
		self.init_scale = init_scale

		if hasattr(layer, 'kw'):
			self.kw = layer.kw

		if hasattr(layer, 'dilation_rate'):
			self.dilation_rate = layer.dilation_rate

		if hasattr(layer, 'filters'):
			self.filters = layer.filters

		if hasattr(layer, 'kernel_size'):
			self.kernel_size = layer.kernel_size

		if hasattr(layer, 'use_bias'):
			self.use_bias = layer.use_bias

		super(WeightNorm, self).__init__(layer, **kwargs)
		self._track_checkpointable(layer, name='layer')

	def set_mode(self, is_training):
		self.layer.set_mode(is_training)

	def _compute_weights(self):
		"""Generate weights by combining the direction of weight vector
		 with it's norm """
		with tf.variable_scope('compute_weights'):
			self.layer.kernel = tf.nn.l2_normalize(
					self.layer.v, axis=self.norm_axes) * self.layer.g

	def _init_norm(self, weights):
		"""Set the norm of the weight vector"""
		with tf.variable_scope('init_norm'):
			flat = tf.reshape(weights, [-1, self.layer_depth])
			return tf.reshape(tf.norm(flat, axis=0), (self.layer_depth,))

	def _data_dep_init(self, inputs):
		"""Data dependent initialization (Done by Calling a feedforward pass at step 0 of training)"""
		with tf.variable_scope('data_dep_init'):
			# Generate data dependant init values
			activation = self.layer.activation
			self.layer.activation = None
			x_init = self.layer.call(inputs)
			m_init, v_init = tf.nn.moments(x_init, self.norm_axes)
			scale_init = self.init_scale / tf.sqrt(v_init + 1e-10)

		# Assign data dependant init values and return x_init
		self.layer.g = self.layer.g * scale_init
		self.layer.bias = (-m_init * scale_init)
		self.layer.activation = activation
		self.initialized = True

		return x_init


	def build(self, input_shape):
		"""Build `Layer`"""
		input_shape = tf.TensorShape(input_shape).as_list()
		self.input_spec = tf.layers.InputSpec(shape=input_shape)

		if not self.layer.built:
			if hasattr(self, 'data_format'):
				self.layer.data_format = self.data_format

			self.layer.build(input_shape)
			self.layer.built = False

			if not hasattr(self.layer, 'kernel'):
				raise ValueError(
						'`WeightNorm` must wrap a layer that'
						' contains a `kernel` for weights'
				)

			# The kernel's filter or unit dimension is -1
			self.layer_depth = int(self.layer.kernel.shape[-1])
			self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

			self.kernel = self.layer.kernel
			self.bias = self.layer.bias

			self.layer.v = self.layer.kernel
			self.layer.g = self.layer.add_variable(
					name="g",
					shape=(self.layer_depth,),
					initializer=tf.constant_initializer(1.),
					dtype=self.layer.kernel.dtype,
					trainable=True)

			with tf.control_dependencies([self.layer.g.assign(
					self._init_norm(self.layer.v))]):
				self._compute_weights()

			self.layer.built = True

		super(WeightNorm, self).build()
		self.built = True

	def call(self, inputs):
		"""Call `Layer`"""
		if self.init:
			return self._data_dep_init(inputs)
		else:
			return self.layer.call(inputs)

	# def incremental_step(self, inputs, convolution_queue=None):
	# 	"""Call wrapped layer"""
	# 	return self.layer.incremental_step(inputs, convolution_queue)


class CausalConv1D(tf.keras.layers.Wrapper):
	def __init__(self, filters,
				 kernel_size,
				 strides=1,
				 data_format='channels_first',
				 dilation_rate=1,
				 activation=None,
				 use_bias=True,
				 weight_normalization = True,
				 weight_normalization_init = True,
				 weight_normalization_init_scale = 1.,
				 kernel_initializer=None,
				 bias_initializer=tf.zeros_initializer(),
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 trainable=True,
				 name=None,
				 **kwargs):

		layer = tf.layers.Conv1D(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='valid',
			data_format=data_format,
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			trainable=trainable,
			name=name, **kwargs
		)

		if weight_normalization:
			layer = WeightNorm(layer, weight_normalization_init, weight_normalization_init_scale)

		super(CausalConv1D, self).__init__(layer, **kwargs)
		self._track_checkpointable(layer, name='layer')
		self.kw = kernel_size
		self.dilation_rate = self.layer.dilation_rate

	def set_mode(self, is_training):
		self.training = is_training

	def _get_linearized_weight(self, in_channels):
		#layers.Conv1D
		if tf.shape(self.layer.kernel) == (self.layer.filters, in_channels, self.kw):
			#[filters, in, kw]
			weight = tf.transpose(self.layer.kernel, [2, 1, 0])
		else:
			#[kw, in, filters]
			weight = self.layer.kernel

		#[kw, in, filters]
		assert weight.shape == (self.kw, in_channels, self.layer.filters)
		self.in_channels = in_channels

		return tf.cast(tf.reshape(weight, [-1, self.layer.filters]), dtype=tf.float32)

	def build(self, input_shape):
		"""Build `Layer`"""
		input_shape = tf.TensorShape(input_shape).as_list()
		self.input_spec = tf.layers.InputSpec(shape=input_shape)

		self.layer.data_format = 'channels_first' if self.training else 'channels_last'
		in_channels = input_shape[1] if self.layer.data_format == 'channels_first' else input_shape[-1]

		#Build layer
		self.layer.build(input_shape)
		self.built = False

		#Create Linearized weights
		self.linearized_weights = self._get_linearized_weight(in_channels)
		super(CausalConv1D, self).build()
		self.built = True
		 
	def call(self, inputs, incremental=False, convolution_queue=None):
		"""Call 'Layer'"""
		if incremental:
			#Incremental run
			#input [batch_size, time_length, channels]
			if self.training:
				raise RuntimeError('incremental step only supported during synthesis')

			batch_size = tf.shape(inputs)[0]

			#Fast dilation
			#Similar to using tf FIFOQueue to schedule states of dilated convolutions
			if self.kw > 1:
				#shift queue (remove first element for following append)
				convolution_queue = convolution_queue[:, 1:, :]

				#append next input
				convolution_queue = tf.concat([convolution_queue, tf.expand_dims(inputs[:, -1, :], axis=1)], axis=1)

				inputs = convolution_queue
				if self.dilation_rate[0] > 1:
					inputs = inputs[:, 0::self.dilation_rate[0], :]

			#Compute step prediction
			output = tf.matmul(tf.reshape(inputs, [batch_size, -1]), self.linearized_weights)
			if self.layer.use_bias:
				output = tf.nn.bias_add(output, self.layer.bias)

			#[batch_size, 1(time_step), channels(filters)]
			if convolution_queue is None:
				return tf.reshape(output, [batch_size, 1, self.layer.filters])
			else:
				return [tf.reshape(output, [batch_size, 1, self.layer.filters]), convolution_queue]

		#Normal run
		#Causal convolutions are only padded on the left side
		assert self.layer.kernel_size[0] == self.kw
		padding = (self.kw - 1) * self.dilation_rate[0]

		#Pad depending on data format
		if self.layer.data_format == 'channels_first':
			time_dim = -1
			inputs_ = tf.pad(inputs, tf.constant([(0, 0), (0, 0), (padding, 0)]))
		else:
			assert self.layer.data_format == 'channels_last'
			time_dim = 1
			inputs_ = tf.pad(inputs, tf.constant([(0, 0), (padding, 0), (0, 0)]))

		#Compute convolution
		outputs = self.layer.call(inputs_)

		#Assert time step consistency
		with tf.control_dependencies([tf.assert_equal(tf.shape(outputs)[time_dim], tf.shape(inputs)[time_dim])]):
			outputs = tf.identity(outputs, name='time_dimension_check')
		return outputs

	def incremental_step(self, inputs, convolution_queue=None):
		'''At sequential inference times:
		we adopt fast wavenet convolution queues approach by saving pre-computed states for faster generation

		inputs: [batch_size, time_length, channels] ('NWC')! Channels last!
		'''
		return self(inputs, incremental=True, convolution_queue=convolution_queue)


class Conv1D1x1(CausalConv1D):
	"""Conv1D 1x1 is literally a causal convolution with kernel_size = 1"""
	def __init__(self, filters,
				 strides=1,
				 data_format='channels_first',
				 dilation_rate=1,
				 activation=None,
				 use_bias=True,
				 weight_normalization = True,
				 weight_normalization_init = True,
				 weight_normalization_init_scale = 1.,
				 kernel_initializer=None,
				 bias_initializer=tf.zeros_initializer(),
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 trainable=True,
				 name=None,
				 **kwargs):
		super(Conv1D1x1, self).__init__(
			filters=filters,
			kernel_size=1,
			strides=strides,
			data_format=data_format,
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			weight_normalization = weight_normalization,
			weight_normalization_init = weight_normalization_init,
			weight_normalization_init_scale = weight_normalization_init_scale,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			trainable=trainable,
			name=name, **kwargs
		)

	def call(self, inputs, incremental=False, convolution_queue=None):
		#Call parent class call function
		return super(Conv1D1x1, self).call(inputs, incremental=incremental, convolution_queue=convolution_queue)

	def incremental_step(self, inputs, unused_queue=None):
		#Call parent class incremental function
		output = self(inputs, incremental=True, convolution_queue=unused_queue) #Drop unused queue
		return output


class ResidualConv1DGLU(tf.keras.layers.Wrapper):
	'''Dilated conv1d + Gated Linear Unit + condition convolutions + residual and skip convolutions

	Dilated convolution is considered to be the most important part of the block so we use it as main layer
	'''
	def __init__(self, residual_channels, gate_channels, kernel_size,
			skip_out_channels=None, cin_channels=-1, gin_channels=-1,
			dropout=1 - .95, dilation_rate=1, use_bias=True,
			weight_normalization=True, init=False, init_scale=1., name='ResidualConv1DGLU', **kwargs):
		self.dropout = dropout
		self.scope = name

		if skip_out_channels is None:
			skip_out_channels = residual_channels

		conv = CausalConv1D(gate_channels, kernel_size,
			dilation_rate=dilation_rate, use_bias=use_bias, 
			weight_normalization=weight_normalization, 
			weight_normalization_init=init, 
			weight_normalization_init_scale=init_scale,
			name='residual_block_conv_{}'.format(name))


		#Local conditioning
		if cin_channels > 0:
			self.conv1x1c = Conv1D1x1(gate_channels, use_bias=use_bias,
				weight_normalization=weight_normalization, 
				weight_normalization_init=init, 
				weight_normalization_init_scale=init_scale, 
				name='residual_block_cin_conv_{}'.format(name))

		else:
			self.conv1x1c = None

		#Global conditioning
		if gin_channels > 0:
			self.conv1x1g = Conv1D1x1(gate_channels, use_bias=use_bias,
				weight_normalization=weight_normalization, 
				weight_normalization_init=init, 
				weight_normalization_init_scale=init_scale,
				name='residual_block_gin_conv_{}'.format(name))

		else:
			self.conv1x1g = None


		gate_out_channels = gate_channels // 2

		self.conv1x1_out = Conv1D1x1(residual_channels, use_bias=use_bias, 
			weight_normalization=weight_normalization, 
			weight_normalization_init=init, 
			weight_normalization_init_scale=init_scale,
			name='residual_block_out_conv_{}'.format(name))

		self.conv1x1_skip = Conv1D1x1(skip_out_channels, use_bias=use_bias, 
			weight_normalization=weight_normalization, 
			weight_normalization_init=init, 
			weight_normalization_init_scale=init_scale,
			name='residual_block_skip_conv_{}'.format(name))

		super(ResidualConv1DGLU, self).__init__(conv, **kwargs)

	def set_mode(self, is_training):
		for conv in [self.layer, self.conv1x1c, self.conv1x1g, self.conv1x1_out, self.conv1x1_skip]:
			try:
				conv.set_mode(is_training)
			except AttributeError:
				pass


	def call(self, x, c=None, g=None):
		x, s, _ = self.step(x, c=c, g=g, is_incremental=False)
		return [x, s]

	def incremental_step(self, x, c=None, g=None, queue=None):
		return self.step(x, c=c, g=g, is_incremental=True, queue=queue)

	def step(self, x, c, g, is_incremental, queue=None):
		'''

		Args:
			x: Tensor [batch_size, channels, time_length]
			c: Tensor [batch_size, c_channels, time_length]. Local conditioning features
			g: Tensor [batch_size, g_channels, time_length], global conditioning features
			is_incremental: Boolean, whether incremental mode is on
		Returns:
			Tensor output
		'''
		residual = x
		x = tf.layers.dropout(x, rate=self.dropout, training=not is_incremental)
		if is_incremental:
			splitdim = -1
			x, queue = self.layer.incremental_step(x, queue)
		else:
			splitdim = 1
			x = self.layer(x)
			#Remove future time steps (They normally don't exist but for safety)
			x = x[:, :, :tf.shape(residual)[-1]]

		a, b = tf.split(x, num_or_size_splits=2, axis=splitdim)

		#local conditioning
		if c is not None:
			assert self.conv1x1c is not None
			c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
			ca, cb = tf.split(c, num_or_size_splits=2, axis=splitdim)
			a, b = a + ca, b + cb

		#global conditioning
		if g is not None:
			assert self.conv1x1g is not None
			g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
			ga, gb = tf.split(g, num_or_size_splits=2, axis=splitdim)
			a, b = a + ga, b + gb

		x = tf.nn.tanh(a) * tf.nn.sigmoid(b)
		#For Skip connection
		s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

		#For Residual connection
		x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

		x = (x + residual) * tf.sqrt(0.5)
		return x, s, queue

class ConvTranspose1D(tf.layers.Conv2DTranspose):
	def __init__(self, filters, kernel_size, padding, strides, **kwargs):
		#convert 1D filters to 2D
		kernel_size = (1, ) + kernel_size #(ks, ) -> (1, ks). Inputs supposed [batch_size, freq, time_steps]
		strides = (1, ) + strides #(s, ) -> (1, s)
		super(ConvTranspose1D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format='channels_first', **kwargs)

	def call(self, inputs):
		return super(ConvTranspose1D, self).call(inputs)


class ConvTranspose2D(tf.layers.Conv2DTranspose):
	def __init__(self, filters, kernel_size, padding, strides):
		freq_axis_kernel_size = kernel_size[0]
		super(ConvTranspose2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			kernel_initializer=tf.constant_initializer(1. / freq_axis_kernel_size, dtype=tf.float32),
			bias_initializer=tf.zeros_initializer(),
			data_format='channels_first')

	def call(self, inputs):
		return super(ConvTranspose2D, self).call(inputs)


def _conv1x1_forward(conv, x, is_incremental):
	"""conv1x1 step
	"""
	if is_incremental:
		return conv.incremental_step(x)
	else:
		return conv(x)

def MaskedCrossEntropyLoss(outputs, targets, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length]
	if mask is None:
		mask = sequence_mask(lengths, max_len, False)

	#One hot encode targets (outputs.shape[-1] = hparams.quantize_channels)
	targets_ = tf.one_hot(targets, depth=tf.shape(outputs)[-1])

	with tf.control_dependencies([tf.assert_equal(tf.shape(outputs), tf.shape(targets_))]):
		losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets_)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def DiscretizedMixtureLogisticLoss(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length, 1]
	if mask is None:
		mask = sequence_mask(lengths, max_len, True)

	#[batch_size, time_length, dimension]
	ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
	mask_ = mask * ones

	losses = discretized_mix_logistic_loss(
		outputs, targets, num_classes=hparams.quantize_channels,
		log_scale_min=hparams.log_scale_min, reduce=False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
		return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)

def GaussianMaximumLikelihoodEstimation(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
	if lengths is None and mask is None:
		raise RuntimeError('Please provide either lengths or mask')

	#[batch_size, time_length, 1]
	if mask is None:
		mask = sequence_mask(lengths, max_len, True)

	#[batch_size, time_length, dimension]
	ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
	mask_ = mask * ones

	losses = gaussian_maximum_likelihood_estimation_loss(
		outputs, targets, log_scale_min_gauss=hparams.log_scale_min_gauss, reduce=False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
		return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)