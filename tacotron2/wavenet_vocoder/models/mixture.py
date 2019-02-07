import numpy as np
import tensorflow as tf


def log_sum_exp(x):
	""" numerically stable log_sum_exp implementation that prevents overflow """
	axis = len(x.get_shape())-1
	m = tf.reduce_max(x, axis)
	m2 = tf.reduce_max(x, axis, keepdims=True)
	return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
	""" numerically stable log_softmax implementation that prevents overflow """
	axis = len(x.get_shape())-1
	m = tf.reduce_max(x, axis, keepdims=True)
	return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
		log_scale_min=-7.0, reduce=True):
	'''Discretized mix of logistic distributions loss.

	Note that it is assumed that input is scaled to [-1, 1]

	Args:
		y_hat: Tensor [batch_size, channels, time_length], predicted output.
		y: Tensor [batch_size, time_length, 1], Target.
	Returns:
		Tensor loss
	'''
	with tf.control_dependencies([tf.assert_equal(tf.mod(tf.shape(y_hat)[1], 3), 0), tf.assert_equal(tf.rank(y_hat), 3)]):
		nr_mix = tf.shape(y_hat)[1] // 3

	#[Batch_size, time_length, channels]
	y_hat = tf.transpose(y_hat, [0, 2, 1])

	#unpack parameters. [batch_size, time_length, num_mixtures] x 3
	logit_probs = y_hat[:, :, :nr_mix]
	means = y_hat[:, :, nr_mix:2 * nr_mix]
	log_scales = tf.maximum(y_hat[:, :, 2* nr_mix: 3 * nr_mix], log_scale_min)

	#[batch_size, time_length, 1] -> [batch_size, time_length, num_mixtures]
	y = y * tf.ones(shape=[1, 1, nr_mix], dtype=tf.float32)

	centered_y = y - means
	inv_stdv = tf.exp(-log_scales)
	plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
	cdf_plus = tf.nn.sigmoid(plus_in)
	min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
	cdf_min = tf.nn.sigmoid(min_in)

	log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
	log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)

	#probability for all other cases
	cdf_delta = cdf_plus - cdf_min

	mid_in = inv_stdv * centered_y
	#log probability in the center of the bin, to be used in extreme cases
	#(not actually used in this code)
	log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

	log_probs = tf.where(y < -0.999, log_cdf_plus,
		tf.where(y > 0.999, log_one_minus_cdf_min,
			tf.where(cdf_delta > 1e-5,
				tf.log(tf.maximum(cdf_delta, 1e-12)),
				log_pdf_mid - np.log((num_classes - 1) / 2))))
	#log_probs = log_probs + tf.nn.log_softmax(logit_probs, -1)

	log_probs = log_probs + tf.nn.log_softmax(logit_probs, axis=-1)

	if reduce:
		return -tf.reduce_sum(log_sum_exp(log_probs))
	else:
		return -tf.expand_dims(log_sum_exp(log_probs), [-1])

def sample_from_discretized_mix_logistic(y, log_scale_min=-7.):
	'''
	Args:
		y: Tensor, [batch_size, channels, time_length]
	Returns:
		Tensor: sample in range of [-1, 1]
	'''
	with tf.control_dependencies([tf.assert_equal(tf.mod(tf.shape(y)[1], 3), 0)]):
		nr_mix = tf.shape(y)[1] // 3

	#[batch_size, time_length, channels]
	y = tf.transpose(y, [0, 2, 1])
	logit_probs = y[:, :, :nr_mix]

	#sample mixture indicator from softmax
	temp = tf.random_uniform(tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5)
	temp = logit_probs - tf.log(-tf.log(temp))
	argmax = tf.argmax(temp, -1)

	#[batch_size, time_length] -> [batch_size, time_length, nr_mix]
	one_hot = tf.one_hot(argmax, depth=nr_mix, dtype=tf.float32)
	#select logistic parameters
	means = tf.reduce_sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, axis=-1)
	log_scales = tf.maximum(tf.reduce_sum(
		y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, axis=-1), log_scale_min)

	#sample from logistic & clip to interval
	#we don't actually round to the nearest 8-bit value when sampling
	u = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
	x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1 -u))

	return tf.minimum(tf.maximum(x, -1.), 1.)
