from pathlib import Path

from utils.argutils import print_args
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("syn_dir", type=Path, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help= \
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("-s", "--save_every", type=int, default=100, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-p", "--print_every", type=int, default=10, help= \
        "Print every N")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    parser.add_argument("--use_amp", action="store_true", help=\
        "Use Pytorch amp.")
    parser.add_argument("--use_tweaked", action="store_true", help=\
        "Use Tweaked")
    parser.add_argument("--multi_gpu", action="store_true", help=\
        "Use Multigpu")
    parser.add_argument("--hparams", default="", help= \
        "Hyperparameter overrides as a comma-separated list of name=value pairs")
    parser.add_argument("--log_file", type=str, default="log.txt", help= \
        "Log filename")
    parser.add_argument('--lr', default=0.1, type=float,
                        help='base learning rate (default=0.1)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default=1e-4)')
    parser.add_argument('--gradinit_lr', default=1e-3, type=float,
                        help='The learning rate of GradInit.')
    parser.add_argument('--gradinit_iters', default=390, type=int,
                        help='Total number of iterations for GradInit.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--gradinit-alg', default='sgd', type=str,
                        help='The target optimization algorithm, deciding the direction of the first gradient step.')
    parser.add_argument('--gradinit-eta', default=0.1, type=float,
                        help='The eta in GradInit.')
    parser.add_argument('--gradinit-min-scale', default=0.01, type=float,
                        help='The lower bound of the scaling factors.')
    parser.add_argument('--gradinit-grad-clip', default=1, type=float,
                        help='Gradient clipping (per dimension) for GradInit.')
    parser.add_argument('--gradinit-gamma', default=float('inf'), type=float,
                        help='The gradient norm constraint.')
    parser.add_argument('--gradinit-normalize-grad', default=False, action='store_true',
                        help='Whether to normalize the gradient for the algorithm A.')
    parser.add_argument('--gradinit-resume', default='', type=str,
                        help='Path to the gradinit or metainit initializations.')
    parser.add_argument('--gradinit-bsize', default=16, type=int,
                        help='Batch size for GradInit.')
    parser.add_argument('--batch-no-overlap', default=False, action='store_true',
                        help=r'Whether to make \tilde{S} and S different.')
    parser.add_argument('--n_epoch', default=200, type=int,
                        help='total number of epochs')

    args = parser.parse_args()
    print_args(args, parser)

    if args.use_tweaked:
        from synthesizer.models.tacotron_tweaked.hparams import hparams
        from synthesizer.models.tacotron_tweaked.train import train
    else:
        from synthesizer.models.tacotron.hparams import hparams
        from synthesizer.models.tacotron.train import train

    args.hparams = hparams.parse(args.hparams)

    # Run the training
    train(**vars(args))
