import torch
from synthesizer import hparams
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.text import text_to_sequence
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os


def run_synthesis(in_dir, out_dir, model_dir):
    # This generates ground truth-aligned mels for vocoder training
    synth_dir = os.path.join(out_dir, "mels_gta")
    os.makedirs(synth_dir, exist_ok=True)
    metadata_filename = os.path.join(in_dir, "train.txt")

    # Load the model in memory
    weights_dir = Path(model_dir)
    model_fpath = weights_dir.joinpath(weights_dir.stem).with_suffix(".pt"))

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tacotron_num_threads = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        tacotron_num_threads = os.cpu_count()
    print('Synthesizer using device:', device)

    # Instantiate Tacotron model
    synth = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=0, # Use zero dropout for gta mels
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    synth.load(model_fpath)
    synth.eval()

    # Load the metadata
    with open(metadata_filename, encoding="utf-8") as f:
        metadata = [line.strip().split("|") for line in f]
        frame_shift_ms = hparams.hop_length / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / 3600
        print("Loaded metadata for {} examples ({:.2f} hours)".format(len(metadata), hours))

    #Set inputs batch wise
    metadata = [metadata[i: i + hparams.tacotron_synthesis_batch_size] for i in
                range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]
    # TODO: come on big boy, fix this
    # Quick and dirty fix to make sure that all batches have the same size 
    metadata = metadata[:-1]

    print("Starting Synthesis")
    mel_dir = os.path.join(in_dir, "mels")
    embed_dir = os.path.join(in_dir, "embeds")
    meta_out_fpath = os.path.join(out_dir, "synthesized.txt")
    with open(meta_out_fpath, "w") as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
            embed_filenames = [os.path.join(embed_dir, m[2]) for m in meta]
            basenames = [os.path.basename(m).replace(".npy", "").replace("mel-", "") 
                         for m in mel_filenames]

            synthesize(synth, texts, basenames, synth_dir, mel_filenames, embed_filenames)

            for elems in meta:
                file.write("|".join([str(x) for x in elems]) + "\n")

    print("Synthesized mel spectrograms at {}".format(synth_dir))
    return meta_out_fpath

def synthesize(synth, texts, basenames, out_dir, mel_filenames, embed_filenames):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(",")]

    assert 0 == len(texts) % tacotron_num_threads
    seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
    input_lengths = [len(seq) for seq in seqs]

    size_per_device = len(seqs) // tacotron_num_threads
    outputs_per_step = 2

    #Pad inputs according to each GPU max length
    input_seqs = None
    split_infos = []
    for i in range(tacotron_num_threads):
        device_input = seqs[size_per_device*i: size_per_device*(i+1)]
        device_input, max_seq_len = _prepare_inputs(device_input)
        input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
        split_infos.append([max_seq_len, 0, 0, 0])

    feed_dict = {
        "inputs": input_seqs,
        "input_lengths": np.asarray(input_lengths, dtype=np.int32),
    }

    np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
    target_lengths = [len(np_target) for np_target in np_targets]
    #pad targets according to each GPU max length
    target_seqs = None
    for i in range(tacotron_num_threads):
        device_target = np_targets[size_per_device*i: size_per_device*(i+1)]
        device_target, max_target_len = _prepare_targets(device_target, outputs_per_step)
        target_seqs = np.concatenate((target_seqs, device_target), axis=1) if target_seqs is not None else device_target
        split_infos[i][1] = max_target_len #Not really used but setting it in case for future development maybe?

    feed_dict["mel_targets"] = target_seqs
    assert len(np_targets) == len(texts)

    feed_dict["split_infos"] = np.asarray(split_infos, dtype=np.int32)
    feed_dict["speaker_embeddings"] = [np.load(f) for f in embed_filenames]

    #Replace this line with the pytorch equivalent
    mels, alignments, stop_tokens = self.session.run(
        [self.mel_outputs, self.alignments, self.stop_token_prediction],
        feed_dict=feed_dict)

    #Linearize outputs (1D arrays)
    mels = [mel for gpu_mels in mels for mel in gpu_mels]
    alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
    stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

    #Take off the batch wise padding
    mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
    assert len(mels) == len(texts)

    if basenames is None:
        raise NotImplemented()

    saved_mels_paths = []
    for i, mel in enumerate(mels):
        # Write the spectrogram to disk
        # Note: outputs mel-spectrogram files and target ones have same names, just different folders
        mel_filename = os.path.join(out_dir, "mel-{}.npy".format(basenames[i]))
        np.save(mel_filename, mel, allow_pickle=False)
        saved_mels_paths.append(mel_filename)

    return saved_mels_paths

def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder

def _prepare_inputs(inputs):
    max_len = max([len(x) for x in inputs])
    return np.stack([_pad_input(x, max_len) for x in inputs]), max_len

def _pad_input(x, length):
    #pad input sequences with the <pad_token> 0 ( _ )
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=0)

def _prepare_targets(targets, alignment):
    max_len = max([len(t) for t in targets])
    data_len = _round_up(max_len, alignment)
    return np.stack([_pad_target(t, data_len) for t in targets]), data_len

def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode="constant", constant_values=-hparams.max_abs_value)
