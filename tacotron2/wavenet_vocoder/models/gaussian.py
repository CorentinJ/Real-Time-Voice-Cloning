import numpy as np
import tensorflow as tf


def gaussian_maximum_likelihood_estimation_loss(y_hat, y, log_scale_min_gauss, reduce=True):
	'''compute the gaussian MLE loss'''
	with tf.control_dependencies([tf.assert_equal(tf.shape(y_hat)[1], 2), tf.assert_equal(tf.rank(y_hat), 3)]):
		#[batch_size, time_steps, channels]
		y_hat = tf.transpose(y_hat, [0, 2, 1])

	#Unpack parameters: mean and log_scale outputs
	mean = y_hat[:, :, 0]
	log_scale = tf.maximum(y_hat[:, :, 1], log_scale_min_gauss)
	scale = tf.exp(log_scale)

	#Get log probability of each sample under this distribution
	gaussian_dist = tf.contrib.distributions.Normal(loc=mean, scale=scale)
	log_prob = gaussian_dist.log_prob(value=tf.squeeze(y, [-1]), name='gaussian_log_prob')

	#Loss (Maximize log probability by minimizing negative log likelihood)
	if reduce:
		return -tf.reduce_sum(log_prob)
	else:
		return -tf.expand_dims(log_prob, [-1])

def sample_from_gaussian(y, log_scale_min_gauss):
	'''sample from learned gaussian distribution'''
	with tf.control_dependencies([tf.assert_equal(tf.shape(y)[1], 2)]):
		#[batch_size, time_length, channels]
		y = tf.transpose(y, [0, 2, 1])

	mean = y[:, :, 0]
	log_scale = tf.maximum(y[:, :, 1], log_scale_min_gauss)
	scale = tf.exp(log_scale)

	gaussian_dist = tf.contrib.distributions.Normal(loc=mean, scale=scale)
	x = gaussian_dist.sample()
	return tf.minimum(tf.maximum(x, -1.), 1.)
