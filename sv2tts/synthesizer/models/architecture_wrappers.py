"""A set of wrappers useful for tacotron 2 architecture
All notations and variable names were used in concordance with originial tensorflow implementation
"""
import collections
import tensorflow as tf
from synthesizer.models.attention import _compute_attention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest

_zero_state_tensors = rnn_cell_impl._zero_state_tensors



class TacotronEncoderCell(RNNCell):
	"""Tacotron 2 Encoder Cell
	Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
	layer to predict the hidden representation vector (or memory)
	"""

	def __init__(self, convolutional_layers, lstm_layer):
		"""Initialize encoder parameters

		Args:
			convolutional_layers: Encoder convolutional block class
			lstm_layer: encoder bidirectional lstm layer class
		"""
		super(TacotronEncoderCell, self).__init__()
		#Initialize encoder layers
		self._convolutions = convolutional_layers
		self._cell = lstm_layer

	def __call__(self, inputs, input_lengths=None):
		#Pass input sequence through a stack of convolutional layers
		conv_output = self._convolutions(inputs)

		#Extract hidden representation from encoder lstm cells
		hidden_representation = self._cell(conv_output, input_lengths)

		#For shape visualization
		self.conv_output_shape = conv_output.shape
		return hidden_representation


class TacotronDecoderCellState(
	collections.namedtuple("TacotronDecoderCellState",
	 ("cell_state", "attention", "time", "alignments",
	  "alignment_history"))):
	"""`namedtuple` storing the state of a `TacotronDecoderCell`.
	Contains:
	  - `cell_state`: The state of the wrapped `RNNCell` at the previous time
		step.
	  - `attention`: The attention emitted at the previous time step.
	  - `time`: int32 scalar containing the current time step.
	  - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
		 emitted at the previous time step for each attention mechanism.
	  - `alignment_history`: a single or tuple of `TensorArray`(s)
		 containing alignment matrices from all time steps for each attention
		 mechanism. Call `stack()` on each to convert to a `Tensor`.
	"""
	def replace(self, **kwargs):
		"""Clones the current state while overwriting components provided by kwargs.
		"""
		return super(TacotronDecoderCellState, self)._replace(**kwargs)

class TacotronDecoderCell(RNNCell):
	"""Tactron 2 Decoder Cell
	Decodes encoder output and previous mel frames into next r frames

	Decoder Step i:
		1) Prenet to compress last output information
		2) Concat compressed inputs with previous context vector (input feeding) *
		3) Decoder RNN (actual decoding) to predict current state s_{i} *
		4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
		5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
		6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)

	* : This is typically taking a vanilla LSTM, wrapping it using tensorflow"s attention wrapper,
	and wrap that with the prenet before doing an input feeding, and with the prediction layer
	that uses RNN states to project on output space. Actions marked with (*) can be replaced with
	tensorflow"s attention wrapper call if it was using cumulative alignments instead of previous alignments only.
	"""

	def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection):
		"""Initialize decoder parameters

		Args:
		    prenet: A tensorflow fully connected layer acting as the decoder pre-net
		    attention_mechanism: A _BaseAttentionMechanism instance, usefull to
			    learn encoder-decoder alignments
		    rnn_cell: Instance of RNNCell, main body of the decoder
		    frame_projection: tensorflow fully connected layer with r * num_mels output units
		    stop_projection: tensorflow fully connected layer, expected to project to a scalar
			    and through a sigmoid activation
			mask_finished: Boolean, Whether to mask decoder frames after the <stop_token>
		"""
		super(TacotronDecoderCell, self).__init__()
		#Initialize decoder layers
		self._prenet = prenet
		self._attention_mechanism = attention_mechanism
		self._cell = rnn_cell
		self._frame_projection = frame_projection
		self._stop_projection = stop_projection

		self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

	def _batch_size_checks(self, batch_size, error_message):
		return [check_ops.assert_equal(batch_size,
		  self._attention_mechanism.batch_size,
		  message=error_message)]

	@property
	def output_size(self):
		return self._frame_projection.shape

	@property
	def state_size(self):
		"""The `state_size` property of `TacotronDecoderCell`.

		Returns:
		  An `TacotronDecoderCell` tuple containing shapes used by this object.
		"""
		return TacotronDecoderCellState(
			cell_state=self._cell._cell.state_size,
			time=tensor_shape.TensorShape([]),
			attention=self._attention_layer_size,
			alignments=self._attention_mechanism.alignments_size,
			alignment_history=())

	def zero_state(self, batch_size, dtype):
		"""Return an initial (zero) state tuple for this `AttentionWrapper`.

		Args:
		  batch_size: `0D` integer tensor: the batch size.
		  dtype: The internal state data type.
		Returns:
		  An `TacotronDecoderCellState` tuple containing zeroed out tensors and,
		  possibly, empty `TensorArray` objects.
		Raises:
		  ValueError: (or, possibly at runtime, InvalidArgument), if
			`batch_size` does not match the output size of the encoder passed
			to the wrapper object at initialization time.
		"""
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			cell_state = self._cell._cell.zero_state(batch_size, dtype)
			error_message = (
				"When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the requested batch size.")
			with ops.control_dependencies(
				self._batch_size_checks(batch_size, error_message)):
				cell_state = nest.map_structure(
					lambda s: array_ops.identity(s, name="checked_cell_state"),
					cell_state)
			return TacotronDecoderCellState(
				cell_state=cell_state,
				time=array_ops.zeros([], dtype=tf.int32),
				attention=_zero_state_tensors(self._attention_layer_size, batch_size,
				  dtype),
				alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
				alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
				dynamic_size=True))

	def __call__(self, inputs, state):
		#Information bottleneck (essential for learning attention)
		prenet_output = self._prenet(inputs)

		#Concat context vector and prenet output to form LSTM cells input (input feeding)
		LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

		#Unidirectional LSTM layers
		LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)


		#Compute the attention (context) vector and alignments using
		#the new decoder cell hidden state as query vector
		#and cumulative alignments to extract location features
		#The choice of the new cell hidden state (s_{i}) of the last
		#decoder RNN Cell is based on Luong et Al. (2015):
		#https://arxiv.org/pdf/1508.04025.pdf
		previous_alignments = state.alignments
		previous_alignment_history = state.alignment_history
		context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
			LSTM_output,
			previous_alignments,
			attention_layer=None)

		#Concat LSTM outputs and context vector to form projections inputs
		projections_input = tf.concat([LSTM_output, context_vector], axis=-1)

		#Compute predicted frames and predicted <stop_token>
		cell_outputs = self._frame_projection(projections_input)
		stop_tokens = self._stop_projection(projections_input)

		#Save alignment history
		alignment_history = previous_alignment_history.write(state.time, alignments)

		#Prepare next decoder state
		next_state = TacotronDecoderCellState(
			time=state.time + 1,
			cell_state=next_cell_state,
			attention=context_vector,
			alignments=cumulated_alignments,
			alignment_history=alignment_history)

		return (cell_outputs, stop_tokens), next_state
