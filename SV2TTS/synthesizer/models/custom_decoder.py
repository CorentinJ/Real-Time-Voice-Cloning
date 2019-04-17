from __future__ import absolute_import, division, print_function
import collections
import tensorflow as tf
from synthesizer.models.helpers import TacoTestHelper, TacoTrainingHelper
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


class CustomDecoderOutput(
		collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
	pass


class CustomDecoder(decoder.Decoder):
	"""Custom sampling decoder.

	Allows for stop token prediction at inference time
	and returns equivalent loss in training time.

	Note:
	Only use this decoder with Tacotron 2 as it only accepts tacotron custom helpers
	"""

	def __init__(self, cell, helper, initial_state, output_layer=None):
		"""Initialize CustomDecoder.
		Args:
			cell: An `RNNCell` instance.
			helper: A `Helper` instance.
			initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
				The initial state of the RNNCell.
			output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
				`tf.layers.Dense`. Optional layer to apply to the RNN output prior
				to storing the result or sampling.
		Raises:
			TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
		"""
		rnn_cell_impl.assert_like_rnncell(type(cell), cell)
		if not isinstance(helper, helper_py.Helper):
			raise TypeError("helper must be a Helper, received: %s" % type(helper))
		if (output_layer is not None
				and not isinstance(output_layer, layers_base.Layer)):
			raise TypeError(
					"output_layer must be a Layer, received: %s" % type(output_layer))
		self._cell = cell
		self._helper = helper
		self._initial_state = initial_state
		self._output_layer = output_layer

	@property
	def batch_size(self):
		return self._helper.batch_size

	def _rnn_output_size(self):
		size = self._cell.output_size
		if self._output_layer is None:
			return size
		else:
			# To use layer"s compute_output_shape, we need to convert the
			# RNNCell"s output_size entries into shapes with an unknown
			# batch size.  We then pass this through the layer"s
			# compute_output_shape and read off all but the first (batch)
			# dimensions to get the output size of the rnn with the layer
			# applied to the top.
			output_shape_with_unknown_batch = nest.map_structure(
					lambda s: tensor_shape.TensorShape([None]).concatenate(s),
					size)
			layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
					output_shape_with_unknown_batch)
			return nest.map_structure(lambda s: s[1:], layer_output_shape)

	@property
	def output_size(self):
		# Return the cell output and the id
		return CustomDecoderOutput(
				rnn_output=self._rnn_output_size(),
				token_output=self._helper.token_output_size,
				sample_id=self._helper.sample_ids_shape)

	@property
	def output_dtype(self):
		# Assume the dtype of the cell is the output_size structure
		# containing the input_state"s first component's dtype.
		# Return that structure and the sample_ids_dtype from the helper.
		dtype = nest.flatten(self._initial_state)[0].dtype
		return CustomDecoderOutput(
				nest.map_structure(lambda _: dtype, self._rnn_output_size()),
				tf.float32,
				self._helper.sample_ids_dtype)

	def initialize(self, name=None):
		"""Initialize the decoder.
		Args:
			name: Name scope for any created operations.
		Returns:
			`(finished, first_inputs, initial_state)`.
		"""
		return self._helper.initialize() + (self._initial_state,)

	def step(self, time, inputs, state, name=None):
		"""Perform a custom decoding step.
		Enables for dyanmic <stop_token> prediction
		Args:
			time: scalar `int32` tensor.
			inputs: A (structure of) input tensors.
			state: A (structure of) state tensors and TensorArrays.
			name: Name scope for any created operations.
		Returns:
			`(outputs, next_state, next_inputs, finished)`.
		"""
		with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
			#Call outputprojection wrapper cell
			(cell_outputs, stop_token), cell_state = self._cell(inputs, state)

			#apply output_layer (if existant)
			if self._output_layer is not None:
				cell_outputs = self._output_layer(cell_outputs)
			sample_ids = self._helper.sample(
					time=time, outputs=cell_outputs, state=cell_state)

			(finished, next_inputs, next_state) = self._helper.next_inputs(
					time=time,
					outputs=cell_outputs,
					state=cell_state,
					sample_ids=sample_ids,
					stop_token_prediction=stop_token)

		outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
		return (outputs, next_state, next_inputs, finished)
