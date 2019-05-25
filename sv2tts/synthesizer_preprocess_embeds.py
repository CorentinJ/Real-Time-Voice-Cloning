from synthesizer.preprocess import create_embeddings
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Creates embedding from the utterances for the synthesizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=str, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=str, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Process the arguments
    args.synthesizer_root = Path(args.synthesizer_root)
    args.encoder_model_fpath = Path(args.encoder_model_fpath)

    # Preprocess the dataset
    create_embeddings(**vars(args))    


if __name__ == "__main__":
    main()
