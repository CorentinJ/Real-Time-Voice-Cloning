import os
import time
import argparse
import math
from numpy import finfo

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
from synthesizer.models.tacotron2 import Tacotron2
# from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.synthesizer_dataset import SynthesizerDataset, TextMelCollate
from synthesizer.loss import Tacotron2Loss
from synthesizer.utils.logger import Tacotron2Logger
# from hparams import create_hparams
from synthesizer.hparams import hparams
from synthesizer.utils.symbols import symbols
# torch.multiprocessing.set_start_method('spawn')


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


# def init_distributed(hparams, n_gpus, rank, group_name):
#     assert torch.cuda.is_available(), "Distributed mode requires CUDA."
#     print("Initializing Distributed")

#     # Set cuda device so everything is done on the right GPU.
#     torch.cuda.set_device(rank % torch.cuda.device_count())

#     # Initialize distributed communication
#     dist.init_process_group(
#         backend=hparams.dist_backend, init_method=hparams.dist_url,
#         world_size=n_gpus, rank=rank, group_name=group_name)

#     print("Done initializing distributed")


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    # if hparams.fp16_run:
    #     model.decoder.attention_layer.score_mask_value = finfo('float16').min

    # if hparams.distributed_run:
    #     model = apply_gradient_allreduce(model)
    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


# def validate(model, criterion, valset, iteration, batch_size, n_gpus,
#              collate_fn, logger, distributed_run, rank):
#     """Handles all the validation scoring and printing"""
#     model.eval()
#     with torch.no_grad():
#         val_sampler = DistributedSampler(valset) if distributed_run else None
#         val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
#                                 shuffle=False, batch_size=batch_size,
#                                 pin_memory=False, collate_fn=collate_fn)

#         val_loss = 0.0
#         for i, batch in enumerate(val_loader):
#             x, y = model.parse_batch(batch)
#             y_pred = model(x)
#             loss = criterion(y_pred, y)
#             # if distributed_run:
#             #     reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
#             # else:
#             reduced_val_loss = loss.item()
#             val_loss += reduced_val_loss
#         val_loss = val_loss / (i + 1)

#     model.train()
#     if rank == 0:
#         print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
#         logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(run_id: str, syn_dir: str, models_dir: str, save_every: int,
          backup_every: int, force_restart: bool, hparams):
    """Training and validation logging results to tensorboard and stdout
    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
    metadata_fpath = syn_dir.joinpath("train.txt")

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Tacotron2")

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    rank = 0

    metadata_fpath = syn_dir.joinpath("train.txt")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    duration_dir = syn_dir.joinpath("duration")

    dataset = SynthesizerDataset(
        metadata_fpath, mel_dir, embed_dir, duration_dir, hparams)
    collate_fn = TextMelCollate(hparams)
    train_loader = DataLoader(dataset,
                              collate_fn=collate_fn, batch_size=hparams.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=False)

    # if hparams.distributed_run:
    #     init_distributed(hparams, n_gpus, rank, group_name)

    # if hparams.fp16_run:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level='O2')

    # if hparams.distributed_run:
    #     model = apply_gradient_allreduce(model)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = Tacotron2Loss()

    # logger = prepare_directories_and_logger(
    #     output_directory, log_directory, 0)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        save_checkpoint(model, optimizer, learning_rate,
                        iteration, weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))

    else:
        print("\nLoading weights at %s" % weights_fpath)
        model, optimizer, _learning_rate, iteration = load_checkpoint(
            weights_fpath, model, optimizer)
        print("Tacotron2 weights loaded from step %d" % model.step)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader, 1):

            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            # print(y_pred[0].size(), y_pred[1].size(), y_pred[2].size())
            loss = criterion(y_pred, y)
            # if hparams.distributed_run:
            #     reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            # else:
            reduced_loss = loss.item()
            # print(loss)
            # if hparams.fp16_run:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()
            # print(loss)
            # if hparams.fp16_run:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         amp.master_params(optimizer), hparams.grad_clip_thresh)
            #     is_overflow = math.isnan(grad_norm)
            # else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            # if not is_overflow and rank == 0:
            #     duration = time.perf_counter() - start
            #     print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
            #         iteration, reduced_loss, grad_norm, duration))
            #     logger.log_training(
            #         reduced_loss, grad_norm, learning_rate, duration, iteration)

            # if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
            #     validate(model, criterion, valset, iteration,
            #              hparams.batch_size, n_gpus, collate_fn, logger,
            #              hparams.distributed_run, rank)
            #     if rank == 0:
            #         checkpoint_path = os.path.join(
            #             output_directory, "checkpoint_{}".format(iteration))
            #         save_checkpoint(model, optimizer, learning_rate, iteration,
            #                         checkpoint_path)

            iteration += 1


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-o', '--output_directory', type=str,
    #                     help='directory to save checkpoints')
    # parser.add_argument('-l', '--log_directory', type=str,
    #                     help='directory to save tensorboard logs')
    # parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
    #                     required=False, help='checkpoint path')
    # parser.add_argument('--warm_start', action='store_true',
    #                     help='load model weights only, ignore specified layers')
    # parser.add_argument('--n_gpus', type=int, default=1,
    #                     required=False, help='number of gpus')
    # parser.add_argument('--rank', type=int, default=0,
    #                     required=False, help='rank of current gpu')
    # parser.add_argument('--group_name', type=str, default='group_name',
    #                     required=False, help='Distributed group name')
    # parser.add_argument('--hparams', type=str,
    #                     required=False, help='comma separated name=value pairs')

    # args = parser.parse_args()
    # hparams = create_hparams(args.hparams)
    # hparams = create_hparams(args.hparams)

    # torch.backends.cudnn.enabled = hparams.cudnn_enabled
    # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    # print("FP16 Run:", hparams.fp16_run)
    # print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    # print("Distributed Run:", hparams.distributed_run)
    # print("cuDNN Enabled:", hparams.cudnn_enabled)
    # print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    # train(args.output_directory, args.log_directory, args.checkpoint_path,
    #       args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
