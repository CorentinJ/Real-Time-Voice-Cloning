import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import sys

# Usage: 
# For embeddings only: 
# `python advanced_cli.py --o data/generated/myvoice01.npy --i data/input/myvoice01.wav --embeddings_only --text_to_speak_location data/text_to_speak.txt`
# For cloned voice: 
# `python advanced_cli.py --o data/generated/cloned01.wav --i data/input/myvoice01.wav --text_to_speak_location data/text_to_speak.txt`
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--o", type=str, help="Location of the output file with extension. (e.g. .npy for embeddings, .wav for cloned audio voice)")
    parser.add_argument("--i", type=str, help="Location of input file.")
    parser.add_argument("--embeddings_only", action="store_true", help="If provided, generate the embeddings only")
    parser.add_argument("--text_to_speak_location", default="text_to_speak.txt", help="The text that will be spoken in the voice provided as input provided inside a text file at this location with line breaks for pause.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    print("Completed loading the models \n")

    in_fpath = Path(args.i)
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    # important: there is preprocessing that must be applied.

    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    print("Pre processed the input WAV file")

    # - If the wav is already loaded:
    #original_wav, sampling_rate = librosa.load(str(in_fpath))
    #preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    #print("Loaded file succesfully")

    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")
    embeddings_file_location = args.o.replace('.wav', '.npy')
    np.save(embeddings_file_location, embed)
    print(f"Generated the embeddings at {embeddings_file_location}")
    if args.embeddings_only:
        print("Generating only the embeddings and exiting")
        sys.exit(0)

    ## Generating the spectrogram
    with open(args.text_to_speak_location, 'r') as fr:
        output_text = fr.read()
    texts = output_text.split("\n")
    embeds = [embed] * len(texts)
    
    # If seed is specified, reset torch seed and force synthesizer reload
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    # texts = [text]
    # embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)
    print("Created the mel spectrogram")

    ## Generating the waveform
    print("Synthesizing the waveform:")

    # If seed is specified, reset torch seed and reload vocoder
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)


    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [generated_wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Save it on the disk
    filename = args.o
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % filename)
            

        
