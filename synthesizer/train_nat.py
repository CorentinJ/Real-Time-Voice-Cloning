import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import os
import time
import wandb
import librosa
from wandb import AlertLevel
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


from synthesizer import audio
from synthesizer.models.nat import NonAttentiveTacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, TextMelCollate
from synthesizer.loss import NATLoss
from synthesizer.optimizer import ScheduledOptim
from synthesizer.utils.plot import plot_spectrogram, plot_alignment
from synthesizer.utils.text import sequence_to_text
from synthesizer.utils import create_alignment

from vocoder.display import *


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string(): return datetime.now().strftime("%Y-%m-%d %H:%M")


def train(run_id: str, syn_dir: str, models_dir: str, save_every: int,
          backup_every: int,  force_restart: bool, wandbb: bool, hparams):

    if wandbb:
        wandb.init(project='RTVC', entity='garvit32',
                   name='NAT-{}'.format(run_id))

    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    alignment_output_dir = model_dir.joinpath("alignment")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    alignment_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
    metadata_fpath = syn_dir.joinpath("train.txt")

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Non Attentive Tacotron")

    # Seeding  ################################################################################

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.deterministic = hparams.cudnn_deterministic
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Train Loader  ###########################################################################

    metadata_fpath = syn_dir.joinpath("train.txt")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    duration_dir = syn_dir.joinpath("duration")

    dataset = SynthesizerDataset(
        metadata_fpath, mel_dir, embed_dir, duration_dir, hparams)
    train_loader = DataLoader(dataset,
                              collate_fn=TextMelCollate(), batch_size=hparams.batch_size**2,
                              num_workers=2,
                              shuffle=True,
                              pin_memory=False)

    model = NonAttentiveTacotron(hparams).cuda()
    if wandbb:
        wandb.watch(model)

    # Optimizer and Loss function  ############################################################

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, eps=hparams.eps)

    criterion = NATLoss()

    print("Optimizer and Loss Function Defined.")

    # logger = prepare_directories_and_logger(
    #     output_directory, log_directory, 0)

    epoch_offset = 1
    current_step = 1
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Non Attentive Tacotron from scratch\n")
        torch.save({'current_step': current_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_offset,
                    'learning_rate': learning_rate}, weights_fpath)

    else:

        checkpoint_dict = torch.load(weights_fpath, map_location='cpu')

        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        learning_rate = checkpoint_dict['learning_rate']
        current_step = checkpoint_dict['current_step']
        epoch_offset = checkpoint_dict['epoch']

        print("Loaded checkpoint '{}' from iteration {}" .format(
            weights_fpath, current_step))

    scheduled_optim = ScheduledOptim(
        optimizer, hparams.n_warm_up_step, current_step)

    model.train()

    is_overflow = False

    total_step = hparams.epochs * len(train_loader) * hparams.batch_size
    for epoch in range(epoch_offset, hparams.epochs):
        for i, batchs in enumerate(train_loader):
            for j, data_of_batch in enumerate(batchs):

                current_step = model.get_step()

                x, y, idx = model.parse_batch(data_of_batch)

                # y_pred = model.inference(x[0], x[-1], x[4])
                y_pred = model(x)

                mel_loss, mel_postnet_loss, duration_loss = criterion(
                    y_pred, y)
                total_loss = mel_loss + mel_postnet_loss + duration_loss * hparams.lambda_dur

                total_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
                if np.isnan(grad_norm.cpu()):
                    if wandbb:
                        wandb.alert(
                            title='Grad exploded',
                            text=f'Step {current_step} | Loss {total_loss} | Epoch {epoch}',
                            level=AlertLevel.WARN,
                            wait_duration=timedelta(minutes=2)
                        )
                    is_overflow = True
                    print("grad_norm was NaN!")
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                msg = "Epoch [{}/{}] | Step [{}/{}] | Total Loss: {:.4f} | Mel Loss: {:.4f} | Mel PostNet Loss: {:.4f} | Duration Loss: {:.4f}".format(
                    epoch, hparams.epochs, current_step, total_step, total_loss.item(), mel_loss.item(), mel_postnet_loss.item(), duration_loss.item())
                stream(msg)

                if wandbb:
                    wandb.log({'Epoch': epoch, 'Step': current_step,
                               'Total Loss': total_loss.item(), "Mel Loss": mel_loss.item(), 'Mel Postnet Loss': mel_postnet_loss.item(), 'Duration Loss': duration_loss.item(), 'Learning Rate': learning_rate})

                if not is_overflow and backup_every != 0 and current_step % backup_every == 0:
                    backup_fpath = Path(
                        "{}/{}_{}k.pt".format(str(weights_fpath.parent), run_id, current_step//1000))
                    torch.save({'current_step': current_step,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                'learning_rate': learning_rate}, backup_fpath)

                if not is_overflow and save_every != 0 and current_step % save_every == 0:
                    torch.save({'current_step': current_step,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                'learning_rate': learning_rate}, weights_fpath)

                step_eval = hparams.tts_eval_interval > 0 and current_step > 0 and current_step % hparams.tts_eval_interval == 0  # Every N steps
                # if True:
                if step_eval:
                    # for sample_idx in range(3):
                    for sample_idx in range(hparams.tts_eval_num_samples):
                        # At most, generate samples equal to number in the batch
                        if sample_idx + 1 <= len(x[0]):
                            # Remove padding from mels using frame length in metadata
                            mel_length = int(
                                dataset.metadata[idx[sample_idx]][5])
                            mel_prediction = np_now(
                                y_pred[0][sample_idx])[:, :mel_length]
                            target_spectrogram = np_now(
                                x[2][sample_idx]).T[:, :mel_length]
                            src_len = x[1][sample_idx]
                            duration = np_now(
                                y_pred[2][sample_idx][:src_len].long())
                            duration = create_alignment(duration)
                            eval_model(
                                mel_prediction=mel_prediction,
                                target_spectrogram=target_spectrogram,
                                input_seq=np_now(x[0][sample_idx]),
                                duration=duration,
                                step=current_step,
                                plot_dir=plot_dir,
                                dur_dir=alignment_output_dir,
                                mel_output_dir=mel_output_dir,
                                wav_dir=wav_dir,
                                sample_num=sample_idx + 1,
                                loss=total_loss.item(),
                                hparams=hparams, wandbb=wandbb)

        # Add line break after every epoch
        print("")


def eval_model(mel_prediction, target_spectrogram, input_seq, duration, step,
               plot_dir, dur_dir, mel_output_dir, wav_dir, sample_num, loss, hparams, wandbb):

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath(
        "mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction, hparams)
    wav_fpath = wav_dir.joinpath(
        "step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath(
        "step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format(
        "NAT", time_string(), step, loss)
    plot_spectrogram(mel_prediction.T, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram.T,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))

    dur_fpath = dur_dir.joinpath(
        "step-{}-alignment_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format(
        "NAT", time_string(), step, loss)
    plot_alignment(duration, str(dur_fpath), title=title_str)

    if (wandbb):
        wav, y = librosa.load(wav_fpath)
        wandb.log({"Generated wav": [wandb.Audio(wav, caption="Step {}".format(
            step), sample_rate=hparams.sample_rate)]})
        spec = plt.imread(spec_fpath)
        wandb.log({"spec": [wandb.Image(
            spec, caption="mel-spectrogram-sample-{}-{}".format(step, loss))]})
        align = plt.imread(dur_fpath)
        wandb.log({"alignment": [wandb.Image(
            align, caption="alignment-{}-{}".format(step, loss))]})
