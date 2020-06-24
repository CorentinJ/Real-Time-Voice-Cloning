from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.hparams import hparams_debug_string
from synthesizer.feeder import Feeder
from synthesizer.models import create_model
from synthesizer.utils import ValueWindow, plot
from synthesizer import infolog, audio
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import traceback
import time
import os

log = infolog.log


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    # Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path
    
    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        # Initialize config
        embedding = config.embeddings.add()
        # Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta
    
    # Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)


def add_train_stats(model, hparams):
    with tf.compat.v1.variable_scope("stats") as scope:
        for i in range(hparams.tacotron_num_gpus):
            tf.compat.v1.summary.histogram("mel_outputs %d" % i, model.tower_mel_outputs[i])
            tf.compat.v1.summary.histogram("mel_targets %d" % i, model.tower_mel_targets[i])
        tf.compat.v1.summary.scalar("before_loss", model.before_loss)
        tf.compat.v1.summary.scalar("after_loss", model.after_loss)
        
        if hparams.predict_linear:
            tf.compat.v1.summary.scalar("linear_loss", model.linear_loss)
            for i in range(hparams.tacotron_num_gpus):
                tf.compat.v1.summary.histogram("mel_outputs %d" % i, model.tower_linear_outputs[i])
                tf.compat.v1.summary.histogram("mel_targets %d" % i, model.tower_linear_targets[i])
        
        tf.compat.v1.summary.scalar("regularization_loss", model.regularization_loss)
        tf.compat.v1.summary.scalar("stop_token_loss", model.stop_token_loss)
        tf.compat.v1.summary.scalar("loss", model.loss)
        tf.compat.v1.summary.scalar("learning_rate", model.learning_rate)  # Control learning rate decay speed
        if hparams.tacotron_teacher_forcing_mode == "scheduled":
            tf.compat.v1.summary.scalar("teacher_forcing_ratio", model.ratio)  # Control teacher forcing
        # ratio decay when mode = "scheduled"
        gradient_norms = [tf.norm(tensor=grad) for grad in model.gradients]
        tf.compat.v1.summary.histogram("gradient_norm", gradient_norms)
        tf.compat.v1.summary.scalar("max_gradient_norm", tf.reduce_max(input_tensor=gradient_norms))  # visualize
        # gradients (in case of explosion)
        return tf.compat.v1.summary.merge_all()


def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss,
                   loss):
    values = [
        tf.compat.v1.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_before_loss",
                                   simple_value=before_loss),
        tf.compat.v1.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_after_loss",
                                   simple_value=after_loss),
        tf.compat.v1.Summary.Value(tag="Tacotron_eval_model/eval_stats/stop_token_loss",
                                   simple_value=stop_token_loss),
        tf.compat.v1.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_loss", simple_value=loss),
    ]
    if linear_loss is not None:
        values.append(tf.compat.v1.Summary.Value(tag="Tacotron_eval_model/eval_stats/eval_linear_loss",
                                                 simple_value=linear_loss))
    test_summary = tf.compat.v1.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def model_train_mode(args, feeder, hparams, global_step):
    with tf.compat.v1.variable_scope("Tacotron_model", reuse=tf.compat.v1.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.speaker_embeddings, 
                         feeder.mel_targets, feeder.token_targets,
                         targets_lengths=feeder.targets_lengths, global_step=global_step,
                         is_training=True, split_infos=feeder.split_infos)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_train_stats(model, hparams)
        return model, stats


def model_test_mode(args, feeder, hparams, global_step):
    with tf.compat.v1.variable_scope("Tacotron_model", reuse=tf.compat.v1.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, 
                         feeder.eval_speaker_embeddings, feeder.eval_mel_targets,
                         feeder.eval_token_targets, targets_lengths=feeder.eval_targets_lengths, 
                         global_step=global_step, is_training=False, is_evaluating=True,
                         split_infos=feeder.eval_split_infos)
        model.add_loss()
        return model


def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, "taco_pretrained")
    plot_dir = os.path.join(log_dir, "plots")
    wav_dir = os.path.join(log_dir, "wavs")
    mel_dir = os.path.join(log_dir, "mel-spectrograms")
    eval_dir = os.path.join(log_dir, "eval-dir")
    eval_plot_dir = os.path.join(eval_dir, "plots")
    eval_wav_dir = os.path.join(eval_dir, "wavs")
    tensorboard_dir = os.path.join(log_dir, "tacotron_events")
    meta_folder = os.path.join(log_dir, "metas")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    
    checkpoint_fpath = os.path.join(save_dir, "tacotron_model.ckpt")
    metadat_fpath = os.path.join(args.synthesizer_root, "train.txt")
    
    log("Checkpoint path: {}".format(checkpoint_fpath))
    log("Loading training data from: {}".format(metadat_fpath))
    log("Using model: Tacotron")
    log(hparams_debug_string())
    
    # Start by setting a seed for repeatability
    tf.compat.v1.set_random_seed(hparams.tacotron_random_seed)
    
    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.compat.v1.variable_scope("datafeeder") as scope:
        feeder = Feeder(coord, metadat_fpath, hparams)
    
    # Set up model:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model, stats = model_train_mode(args, feeder, hparams, global_step)
    eval_model = model_test_mode(args, feeder, hparams, global_step)
    
    # Embeddings metadata
    char_embedding_meta = os.path.join(meta_folder, "CharacterEmbeddings.tsv")
    if not os.path.isfile(char_embedding_meta):
        with open(char_embedding_meta, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s
                
                f.write("{}\n".format(symbol))
    
    char_embedding_meta = char_embedding_meta.replace(log_dir, "..")
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    
    log("Tacotron training set to a maximum of {} steps".format(args.tacotron_train_steps))
    
    # Memory allocation on the GPU as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    # Train
    with tf.compat.v1.Session(config=config) as sess:
        try:
            summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_dir, sess.graph)
            
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        log("Loading checkpoint {}".format(checkpoint_state.model_checkpoint_path),
                            slack=True)
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    
                    else:
                        log("No model to load at {}".format(save_dir), slack=True)
                        saver.save(sess, checkpoint_fpath, global_step=global_step)
                
                except tf.errors.OutOfRangeError as e:
                    log("Cannot restore checkpoint: {}".format(e), slack=True)
            else:
                log("Starting new training!", slack=True)
                saver.save(sess, checkpoint_fpath, global_step=global_step)
            
            # initializing feeder
            feeder.start_threads(sess)
            
            # Training loop
            while not coord.should_stop() and step < args.tacotron_train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = "Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]".format(
                    step, time_window.average, loss, loss_window.average)
                log(message, end="\r", slack=(step % args.checkpoint_interval == 0))
                print(message)
                
                if loss > 100 or np.isnan(loss):
                    log("Loss exploded to {:.5f} at step {}".format(loss, step))
                    raise Exception("Loss exploded")
                
                if step % args.summary_interval == 0:
                    log("\nWriting summary at step {}".format(step))
                    summary_writer.add_summary(sess.run(stats), step)
                
                if step % args.eval_interval == 0:
                    # Run eval and save eval stats
                    log("\nRunning evaluation at step {}".format(step))
                    
                    eval_losses = []
                    before_losses = []
                    after_losses = []
                    stop_token_losses = []
                    linear_losses = []
                    linear_loss = None
                    
                    if hparams.predict_linear:
                        for i in tqdm(range(feeder.test_steps)):
                            eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, \
							mel_t, t_len, align, lin_p, lin_t = sess.run(
                                [
                                    eval_model.tower_loss[0], eval_model.tower_before_loss[0],
                                    eval_model.tower_after_loss[0],
                                    eval_model.tower_stop_token_loss[0],
                                    eval_model.tower_linear_loss[0],
                                    eval_model.tower_mel_outputs[0][0],
                                    eval_model.tower_mel_targets[0][0],
                                    eval_model.tower_targets_lengths[0][0],
                                    eval_model.tower_alignments[0][0],
                                    eval_model.tower_linear_outputs[0][0],
                                    eval_model.tower_linear_targets[0][0],
                                ])
                            eval_losses.append(eloss)
                            before_losses.append(before_loss)
                            after_losses.append(after_loss)
                            stop_token_losses.append(stop_token_loss)
                            linear_losses.append(linear_loss)
                        linear_loss = sum(linear_losses) / len(linear_losses)
                        
                        wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
                        audio.save_wav(wav, os.path.join(eval_wav_dir,
                                                         "step-{}-eval-wave-from-linear.wav".format(
                                                             step)), sr=hparams.sample_rate)
                    
                    else:
                        for i in tqdm(range(feeder.test_steps)):
                            eloss, before_loss, after_loss, stop_token_loss, mel_p, mel_t, t_len,\
							align = sess.run(
                                [
                                    eval_model.tower_loss[0], eval_model.tower_before_loss[0],
                                    eval_model.tower_after_loss[0],
                                    eval_model.tower_stop_token_loss[0],
                                    eval_model.tower_mel_outputs[0][0],
                                    eval_model.tower_mel_targets[0][0],
                                    eval_model.tower_targets_lengths[0][0],
                                    eval_model.tower_alignments[0][0]
                                ])
                            eval_losses.append(eloss)
                            before_losses.append(before_loss)
                            after_losses.append(after_loss)
                            stop_token_losses.append(stop_token_loss)
                    
                    eval_loss = sum(eval_losses) / len(eval_losses)
                    before_loss = sum(before_losses) / len(before_losses)
                    after_loss = sum(after_losses) / len(after_losses)
                    stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)
                    
                    log("Saving eval log to {}..".format(eval_dir))
                    # Save some log to monitor model improvement on same unseen sequence
                    wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
                    audio.save_wav(wav, os.path.join(eval_wav_dir,
                                                     "step-{}-eval-wave-from-mel.wav".format(step)),
                                   sr=hparams.sample_rate)
                    
                    plot.plot_alignment(align, os.path.join(eval_plot_dir,
                                                            "step-{}-eval-align.png".format(step)),
                                        title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                    time_string(),
                                                                                    step,
                                                                                    eval_loss),
                                        max_len=t_len // hparams.outputs_per_step)
                    plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir,
                                                              "step-{"
															  "}-eval-mel-spectrogram.png".format(
                                                                  step)),
                                          title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                      time_string(),
                                                                                      step,
                                                                                      eval_loss),
                                          target_spectrogram=mel_t,
                                          max_len=t_len)
                    
                    if hparams.predict_linear:
                        plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir,
                                                                  "step-{}-eval-linear-spectrogram.png".format(
                                                                      step)),
                                              title="{}, {}, step={}, loss={:.5f}".format(
                                                  "Tacotron", time_string(), step, eval_loss),
                                              target_spectrogram=lin_t,
                                              max_len=t_len, auto_aspect=True)
                    
                    log("Eval loss for global step {}: {:.3f}".format(step, eval_loss))
                    log("Writing eval summary!")
                    add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss,
                                   stop_token_loss, eval_loss)
                
                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or \
                        step == 300:
                    # Save model and current global step
                    saver.save(sess, checkpoint_fpath, global_step=global_step)
                    
                    log("\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..")
                    input_seq, mel_prediction, alignment, target, target_length = sess.run([
                        model.tower_inputs[0][0],
                        model.tower_mel_outputs[0][0],
                        model.tower_alignments[0][0],
                        model.tower_mel_targets[0][0],
                        model.tower_targets_lengths[0][0],
                    ])
                    
                    # save predicted mel spectrogram to disk (debug)
                    mel_filename = "mel-prediction-step-{}.npy".format(step)
                    np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T,
                            allow_pickle=False)
                    
                    # save griffin lim inverted wav for debug (mel -> wav)
                    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                    audio.save_wav(wav,
                                   os.path.join(wav_dir, "step-{}-wave-from-mel.wav".format(step)),
                                   sr=hparams.sample_rate)
                    
                    # save alignment plot to disk (control purposes)
                    plot.plot_alignment(alignment,
                                        os.path.join(plot_dir, "step-{}-align.png".format(step)),
                                        title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                    time_string(),
                                                                                    step, loss),
                                        max_len=target_length // hparams.outputs_per_step)
                    # save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                                                                       "step-{}-mel-spectrogram.png".format(
                                                                           step)),
                                          title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                      time_string(),
                                                                                      step, loss),
                                          target_spectrogram=target,
                                          max_len=target_length)
                    log("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
                
                if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
                    # Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    # Update Projector
                    log("\nSaving Model Character Embeddings visualization..")
                    add_embedding_stats(summary_writer, [model.embedding_table.name],
                                        [char_embedding_meta],
                                        checkpoint_state.model_checkpoint_path)
                    log("Tacotron Character embeddings have been updated on tensorboard!")
            
            log("Tacotron training complete after {} global steps!".format(
                args.tacotron_train_steps), slack=True)
            return save_dir
        
        except Exception as e:
            log("Exiting due to exception: {}".format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)
