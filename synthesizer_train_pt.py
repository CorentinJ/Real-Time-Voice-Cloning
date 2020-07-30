from synthesizer.train import train
from utils.argutils import print_args
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the run and of the logging directory.")
    parser.add_argument("synthesizer_root", type=str, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("--mode", default="synthesis",
                        help="mode for synthesis of tacotron after training")
    parser.add_argument("--GTA", default="True",
                        help="Ground truth aligned synthesis, defaults to True, only considered "
							 "in Tacotron synthesis mode")
    parser.add_argument("--restore", type=bool, default=True,
                        help="Set this to False to do a fresh training")
    parser.add_argument("--summary_interval", type=int, default=2500,
                        help="Steps between running summary ops")
    parser.add_argument("--embedding_interval", type=int, default=10000,
                        help="Steps between updating embeddings projection visualization")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, # Was 5000
                        help="Steps between writing checkpoints")
    parser.add_argument("--eval_interval", type=int, default=100000, # Was 10000
                        help="Steps between eval on test data")
    parser.add_argument("--tacotron_train_steps", type=int, default=2000000, # Was 100000
                        help="total number of tacotron training steps")
    args = parser.parse_args()
    print_args(args, parser)
    
    tacotron_train(args)
