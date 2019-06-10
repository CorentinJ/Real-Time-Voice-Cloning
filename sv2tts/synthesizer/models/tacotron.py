import tensorflow as tf
from synthesizer.utils.symbols import symbols
from synthesizer.infolog import log
from synthesizer.models.helpers import TacoTrainingHelper, TacoTestHelper
from synthesizer.models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from synthesizer.models.architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from synthesizer.models.custom_decoder import CustomDecoder
from synthesizer.models.attention import LocationSensitiveAttention

import numpy as np


def split_func(x, split_pos):
    rst = []
    start = 0
    # x will be a numpy array with the contents of the placeholder below
    for i in range(split_pos.shape[0]):
        rst.append(x[:, start:start + split_pos[i]])
        start += split_pos[i]
    return rst


class Tacotron():
    """Tacotron-2 Feature prediction Model.
    """
    
    def __init__(self, hparams):
        self._hparams = hparams
    
    def initialize(self, inputs, input_lengths, embed_targets, mel_targets=None, 
                   stop_token_targets=None, linear_targets=None, targets_lengths=None, gta=False,
                   global_step=None, is_training=False, is_evaluating=False, split_infos=None):
        """
        Initializes the model for inference sets "mel_outputs" and "alignments" fields.
        Args:
            - inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
              steps in the input time series, and values are character IDs
            - input_lengths: int32 Tensor with shape [N] where N is batch size and values are the 
            lengths of each sequence in inputs.
            - embed_targets: float32 Tensor with shape [N, E] where E is the speaker 
            embedding size.
            - mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, 
            T_out is number of steps in the output time series, M is num_mels, and values are 
            entries in the mel spectrogram. Only needed for training.
        """
        if mel_targets is None and stop_token_targets is not None:
            raise ValueError("no multi targets were provided but token_targets were given")
        if mel_targets is not None and stop_token_targets is None and not gta:
            raise ValueError("Mel targets are provided without corresponding token_targets")
        if not gta and self._hparams.predict_linear == True and linear_targets is None and \
				is_training:
            raise ValueError(
                "Model is set to use post processing to predict linear spectrograms in training "
				"but no linear targets given!")
        if gta and linear_targets is not None:
            raise ValueError("Linear spectrogram prediction is not supported in GTA mode!")
        if is_training and self._hparams.mask_decoder and targets_lengths is None:
            raise RuntimeError(
                "Model set to mask paddings but no targets lengths provided for the mask!")
        if is_training and is_evaluating:
            raise RuntimeError(
                "Model can not be in training and evaluation modes at the same time!")
        
        split_device = "/cpu:0" if self._hparams.tacotron_num_gpus > 1 or \
								   self._hparams.split_on_cpu else "/gpu:{}".format(
            self._hparams.tacotron_gpu_start_idx)
        with tf.device(split_device):
            hp = self._hparams
            lout_int = [tf.int32] * hp.tacotron_num_gpus
            lout_float = [tf.float32] * hp.tacotron_num_gpus
            
            tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.tacotron_num_gpus,
                                           axis=0)
            tower_targets_lengths = \
                tf.split(targets_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0) if \
                    targets_lengths is not None else targets_lengths
            
            ### SV2TTS ###
            
            tower_embed_targets = tf.split(embed_targets, num_or_size_splits=hp.tacotron_num_gpus,
                                           axis=0)
            
            ##############
            
            p_inputs = tf.py_func(split_func, [inputs, split_infos[:, 0]], lout_int)
            p_mel_targets = tf.py_func(split_func, [mel_targets, split_infos[:, 1]],
                                       lout_float) if mel_targets is not None else mel_targets
            p_stop_token_targets = tf.py_func(split_func, [stop_token_targets, split_infos[:, 2]],
                                              lout_float) if stop_token_targets is not None else \
				stop_token_targets
            
            tower_inputs = []
            tower_mel_targets = []
            tower_stop_token_targets = []
            
            batch_size = tf.shape(inputs)[0]
            mel_channels = hp.num_mels
            for i in range(hp.tacotron_num_gpus):
                tower_inputs.append(tf.reshape(p_inputs[i], [batch_size, -1]))
                if p_mel_targets is not None:
                    tower_mel_targets.append(
                        tf.reshape(p_mel_targets[i], [batch_size, -1, mel_channels]))
                if p_stop_token_targets is not None:
                    tower_stop_token_targets.append(
                        tf.reshape(p_stop_token_targets[i], [batch_size, -1]))
        
        self.tower_decoder_output = []
        self.tower_alignments = []
        self.tower_stop_token_prediction = []
        self.tower_mel_outputs = []
        
        tower_embedded_inputs = []
        tower_enc_conv_output_shape = []
        tower_encoder_cond_outputs = []
        tower_residual = []
        tower_projected_residual = []
        
        # 1. Declare GPU Devices
        gpus = ["/gpu:{}".format(i) for i in
                range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx + hp.tacotron_num_gpus)]
        for i in range(hp.tacotron_num_gpus):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0",
                                                          worker_device=gpus[i])):
                with tf.variable_scope("inference") as scope:
                    assert hp.tacotron_teacher_forcing_mode in ("constant", "scheduled")
                    if hp.tacotron_teacher_forcing_mode == "scheduled" and is_training:
                        assert global_step is not None
                    
                    # GTA is only used for predicting mels to train Wavenet vocoder, so we ommit 
                    # post processing when doing GTA synthesis
                    post_condition = hp.predict_linear and not gta
                    
                    # Embeddings ==> [batch_size, sequence_length, embedding_dim]
                    self.embedding_table = tf.get_variable(
                        "inputs_embedding", [len(symbols), hp.embedding_dim], dtype=tf.float32)
                    embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, tower_inputs[i])
                    
                    # Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
                    encoder_cell = TacotronEncoderCell(
                        EncoderConvolutions(is_training, hparams=hp, scope="encoder_convolutions"),
                        EncoderRNN(is_training, size=hp.encoder_lstm_units,
                                   zoneout=hp.tacotron_zoneout_rate, scope="encoder_LSTM"))
                    
                    encoder_outputs = encoder_cell(embedded_inputs, tower_input_lengths[i])
                    
                    # For shape visualization purpose
                    enc_conv_output_shape = encoder_cell.conv_output_shape
                    
                    
                    ### SV2TT2 ###
                    
                    # Append the speaker embedding to the encoder output at each timestep
                    tileable_shape = [-1, 1, self._hparams.speaker_embedding_size]
                    tileable_embed_targets = tf.reshape(tower_embed_targets[i], tileable_shape)
                    tiled_embed_targets = tf.tile(tileable_embed_targets, 
                                                       [1, tf.shape(encoder_outputs)[1], 1])
                    encoder_cond_outputs = tf.concat((encoder_outputs, tiled_embed_targets), 2)
                    
                    ##############
                    
                    
                    # Decoder Parts
                    # Attention Decoder Prenet
                    prenet = Prenet(is_training, layers_sizes=hp.prenet_layers,
                                    drop_rate=hp.tacotron_dropout_rate, scope="decoder_prenet")
                    # Attention Mechanism
                    attention_mechanism = LocationSensitiveAttention(hp.attention_dim,
                                                                     encoder_cond_outputs, 
                                                                     hparams=hp,
                                                                     mask_encoder=hp.mask_encoder,
                                                                     memory_sequence_length=tf.reshape(
                                                                         tower_input_lengths[i],
                                                                         [-1]),
                                                                     smoothing=hp.smoothing,
                                                                     cumulate_weights=hp.cumulative_weights)
                    # Decoder LSTM Cells
                    decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
                                              size=hp.decoder_lstm_units,
                                              zoneout=hp.tacotron_zoneout_rate,
                                              scope="decoder_LSTM")
                    # Frames Projection layer
                    frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step,
                                                       scope="linear_transform_projection")
                    # <stop_token> projection layer
                    stop_projection = StopProjection(is_training or is_evaluating, shape=hp
                                                     .outputs_per_step,
                                                     scope="stop_token_projection")
                    
                    # Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
                    decoder_cell = TacotronDecoderCell(
                        prenet,
                        attention_mechanism,
                        decoder_lstm,
                        frame_projection,
                        stop_projection)
                    
                    # Define the helper for our decoder
                    if is_training or is_evaluating or gta:
                        self.helper = TacoTrainingHelper(batch_size, tower_mel_targets[i], hp, gta,
                                                         is_evaluating, global_step)
                    else:
                        self.helper = TacoTestHelper(batch_size, hp)
                    
                    # initial decoder state
                    decoder_init_state = decoder_cell.zero_state(batch_size=batch_size,
                                                                 dtype=tf.float32)
                    
                    # Only use max iterations at synthesis time
                    max_iters = hp.max_iters if not (is_training or is_evaluating) else None
                    
                    # Decode
                    (frames_prediction, stop_token_prediction,
                     _), final_decoder_state, _ = dynamic_decode(
                        CustomDecoder(decoder_cell, self.helper, decoder_init_state),
                        impute_finished=False,
                        maximum_iterations=max_iters,
                        swap_memory=hp.tacotron_swap_with_cpu)
                    
                    # Reshape outputs to be one output per entry 
                    # ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
                    decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
                    stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])
                    
                    # Postnet
                    postnet = Postnet(is_training, hparams=hp, scope="postnet_convolutions")
                    
                    # Compute residual using post-net ==> [batch_size, decoder_steps * r, 
                    # postnet_channels]
                    residual = postnet(decoder_output)
                    
                    # Project residual to same dimension as mel spectrogram 
                    # ==> [batch_size, decoder_steps * r, num_mels]
                    residual_projection = FrameProjection(hp.num_mels, scope="postnet_projection")
                    projected_residual = residual_projection(residual)
                    
                    # Compute the mel spectrogram
                    mel_outputs = decoder_output + projected_residual
                    
                    if post_condition:
                        # Add post-processing CBHG. This does a great job at extracting features 
						# from mels before projection to Linear specs.
                        post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size,
                                         [hp.cbhg_projection, hp.num_mels],
                                         hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
                                         hp.cbhg_highway_units, hp.cbhg_rnn_units, is_training,
                                         name="CBHG_postnet")
                        
                        # [batch_size, decoder_steps(mel_frames), cbhg_channels]
                        post_outputs = post_cbhg(mel_outputs, None)
                        
                        # Linear projection of extracted features to make linear spectrogram
                        linear_specs_projection = FrameProjection(hp.num_freq,
                                                                  scope="cbhg_linear_specs_projection")
                        
                        # [batch_size, decoder_steps(linear_frames), num_freq]
                        linear_outputs = linear_specs_projection(post_outputs)
                    
                    # Grab alignments from the final decoder state
                    alignments = tf.transpose(final_decoder_state.alignment_history.stack(),
                                              [1, 2, 0])
                    
                    self.tower_decoder_output.append(decoder_output)
                    self.tower_alignments.append(alignments)
                    self.tower_stop_token_prediction.append(stop_token_prediction)
                    self.tower_mel_outputs.append(mel_outputs)
                    tower_embedded_inputs.append(embedded_inputs)
                    tower_enc_conv_output_shape.append(enc_conv_output_shape)
                    tower_encoder_cond_outputs.append(encoder_cond_outputs)
                    tower_residual.append(residual)
                    tower_projected_residual.append(projected_residual)
                    
                    if post_condition:
                        self.tower_linear_outputs.append(linear_outputs)
            log("initialisation done {}".format(gpus[i]))
        
        if is_training:
            self.ratio = self.helper._ratio
        self.tower_inputs = tower_inputs
        self.tower_input_lengths = tower_input_lengths
        self.tower_mel_targets = tower_mel_targets
        # self.tower_linear_targets = tower_linear_targets
        self.tower_targets_lengths = tower_targets_lengths
        self.tower_stop_token_targets = tower_stop_token_targets
        
        self.all_vars = tf.trainable_variables()
        
        log("Initialized Tacotron model. Dimensions (? = dynamic shape): ")
        log("  Train mode:               {}".format(is_training))
        log("  Eval mode:                {}".format(is_evaluating))
        log("  GTA mode:                 {}".format(gta))
        log("  Synthesis mode:           {}".format(not (is_training or is_evaluating)))
        log("  Input:                    {}".format(inputs.shape))
        for i in range(hp.tacotron_num_gpus + hp.tacotron_gpu_start_idx):
            log("  device:                   {}".format(i))
            log("  embedding:                {}".format(tower_embedded_inputs[i].shape))
            log("  enc conv out:             {}".format(tower_enc_conv_output_shape[i]))
            log("  encoder out (cond):       {}".format(tower_encoder_cond_outputs[i].shape))
            log("  decoder out:              {}".format(self.tower_decoder_output[i].shape))
            log("  residual out:             {}".format(tower_residual[i].shape))
            log("  projected residual out:   {}".format(tower_projected_residual[i].shape))
            log("  mel out:                  {}".format(self.tower_mel_outputs[i].shape))
            if post_condition:
                log("  linear out:               {}".format(self.tower_linear_outputs[i].shape))
            log("  <stop_token> out:         {}".format(self.tower_stop_token_prediction[i].shape))
            
            # 1_000_000 is causing syntax problems for some people?! Python please :)
            log("  Tacotron Parameters       {:.3f} Million.".format(
                np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))
    
    
    def add_loss(self):
        """Adds loss to the model. Sets "loss" field. initialize must have been called."""
        hp = self._hparams
        
        self.tower_before_loss = []
        self.tower_after_loss = []
        self.tower_stop_token_loss = []
        self.tower_regularization_loss = []
        self.tower_linear_loss = []
        self.tower_loss = []
        
        total_before_loss = 0
        total_after_loss = 0
        total_stop_token_loss = 0
        total_regularization_loss = 0
        total_linear_loss = 0
        total_loss = 0

        gpus = ["/gpu:{}".format(i) for i in
                range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx + hp.tacotron_num_gpus)]
        
        for i in range(hp.tacotron_num_gpus):
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0",
                                                          worker_device=gpus[i])):
                with tf.variable_scope("loss") as scope:
                    if hp.mask_decoder:
                        # Compute loss of predictions before postnet
                        before = MaskedMSE(self.tower_mel_targets[i], self.tower_decoder_output[i],
                                           self.tower_targets_lengths[i],
                                           hparams=self._hparams)
                        # Compute loss after postnet
                        after = MaskedMSE(self.tower_mel_targets[i], self.tower_mel_outputs[i],
                                          self.tower_targets_lengths[i],
                                          hparams=self._hparams)
                        # Compute <stop_token> loss (for learning dynamic generation stop)
                        stop_token_loss = MaskedSigmoidCrossEntropy(
                            self.tower_stop_token_targets[i],
                            self.tower_stop_token_prediction[i], self.tower_targets_lengths[i],
                            hparams=self._hparams)
                        # SV2TTS extra L1 loss (disabled for now)
                        # linear_loss = MaskedLinearLoss(self.tower_mel_targets[i],
                        #                                self.tower_decoder_output[i],
                        #                                self.tower_targets_lengths[i],
                        #                                hparams=self._hparams)
                        linear_loss = 0.
                    else:
                        # Compute loss of predictions before postnet
                        before = tf.losses.mean_squared_error(self.tower_mel_targets[i],
                                                              self.tower_decoder_output[i])
                        # Compute loss after postnet
                        after = tf.losses.mean_squared_error(self.tower_mel_targets[i],
                                                             self.tower_mel_outputs[i])
                        # Compute <stop_token> loss (for learning dynamic generation stop)
                        stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.tower_stop_token_targets[i],
                            logits=self.tower_stop_token_prediction[i]))
                        
                        # SV2TTS extra L1 loss
                        l1 = tf.abs(self.tower_mel_targets[i] - self.tower_decoder_output[i])
                        linear_loss = tf.reduce_mean(l1)

                        # if hp.predict_linear:
                        #     # Compute linear loss
                        #     # From https://github.com/keithito/tacotron/blob/tacotron2-work-in
						# 	# -progress/models/tacotron.py
                        #     # Prioritize loss for frequencies under 2000 Hz.
                        #     l1 = tf.abs(self.tower_linear_targets[i] - self.tower_linear_outputs[i])
                        #     n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
                        #     linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(
                        #         l1[:, :, 0:n_priority_freq])
                        # else:
                        #     linear_loss = 0.
                    
                    # Compute the regularization weight
                    if hp.tacotron_scale_regularization:
                        reg_weight_scaler = 1. / (
                                    2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (
                            hp.max_abs_value)
                        reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
                    else:
                        reg_weight = hp.tacotron_reg_weight
                    
                    # Regularize variables
                    # Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
                    # Note that we consider attention mechanism v_a weights as a prediction projection layer and we don"t regularize it. (This gave better stability)
                    regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
                                               if not (
                                    "bias" in v.name or "Bias" in v.name or "_projection" in v.name or "inputs_embedding" in v.name
                                    or "RNN" in v.name or "LSTM" in v.name)]) * reg_weight
                    
                    # Compute final loss term
                    self.tower_before_loss.append(before)
                    self.tower_after_loss.append(after)
                    self.tower_stop_token_loss.append(stop_token_loss)
                    self.tower_regularization_loss.append(regularization)
                    self.tower_linear_loss.append(linear_loss)
                    
                    loss = before + after + stop_token_loss + regularization + linear_loss
                    self.tower_loss.append(loss)
        
        for i in range(hp.tacotron_num_gpus):
            total_before_loss += self.tower_before_loss[i]
            total_after_loss += self.tower_after_loss[i]
            total_stop_token_loss += self.tower_stop_token_loss[i]
            total_regularization_loss += self.tower_regularization_loss[i]
            total_linear_loss += self.tower_linear_loss[i]
            total_loss += self.tower_loss[i]
        
        self.before_loss = total_before_loss / hp.tacotron_num_gpus
        self.after_loss = total_after_loss / hp.tacotron_num_gpus
        self.stop_token_loss = total_stop_token_loss / hp.tacotron_num_gpus
        self.regularization_loss = total_regularization_loss / hp.tacotron_num_gpus
        self.linear_loss = total_linear_loss / hp.tacotron_num_gpus
        self.loss = total_loss / hp.tacotron_num_gpus
    
    def add_optimizer(self, global_step):
        """Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step in training
        """
        hp = self._hparams
        tower_gradients = []
        
        # 1. Declare GPU Devices
        gpus = ["/gpu:{}".format(i) for i in
                range(hp.tacotron_gpu_start_idx, hp.tacotron_gpu_start_idx + hp.tacotron_num_gpus)]
        
        grad_device = "/cpu:0" if hp.tacotron_num_gpus > 1 else gpus[0]
        
        with tf.device(grad_device):
            with tf.variable_scope("optimizer") as scope:
                if hp.tacotron_decay_learning_rate:
                    self.decay_steps = hp.tacotron_decay_steps
                    self.decay_rate = hp.tacotron_decay_rate
                    self.learning_rate = self._learning_rate_decay(
                        hp.tacotron_initial_learning_rate, global_step)
                else:
                    self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)
                
                optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
                                                   hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
        
        # 2. Compute Gradient
        for i in range(hp.tacotron_num_gpus):
            #  Device placement
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0",
                                                          worker_device=gpus[i])):
                # agg_loss += self.tower_loss[i]
                with tf.variable_scope("optimizer") as scope:
                    gradients = optimizer.compute_gradients(self.tower_loss[i])
                    tower_gradients.append(gradients)
        
        # 3. Average Gradient
        with tf.device(grad_device):
            avg_grads = []
            vars = []
            for grad_and_vars in zip(*tower_gradients):
                # grads_vars = [(grad1, var), (grad2, var), ...]
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a "tower" dimension which we will average over below.
                    grads.append(expanded_g)
                # Average over the "tower" dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                
                v = grad_and_vars[0][1]
                avg_grads.append(grad)
                vars.append(v)
            
            self.gradients = avg_grads
            # Just for causion
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.tacotron_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(avg_grads, 1.)  # __mark 0.5 refer
            else:
                clipped_gradients = avg_grads
            
            # Add dependency on UPDATE_OPS; otherwise batchnorm won"t work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, vars),
                                                          global_step=global_step)
    
    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:
        
        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps
        
        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k
        
        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self._hparams
        
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.tacotron_start_decay,
                                        # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name="lr_exponential_decay")
        
        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
