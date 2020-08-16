import torch
from torch.utils.data import DataLoader
from synthesizer import hparams
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.text import text_to_sequence
from synthesizer.utils.symbols import symbols
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def run_synthesis(in_dir, out_dir, model_dir):
    # This generates ground truth-aligned mels for vocoder training
    synth_dir = os.path.join(out_dir, "mels_gta")
    os.makedirs(synth_dir, exist_ok=True)

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if synthesis_batch_size % torch.cuda.device_count() != 0:
            raise ValueError("`hparams.synthesis_batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

    # Instantiate Tacotron model
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
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
                     dropout=0., # Use zero dropout for gta mels
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Load the weights
    model_dir = Path(model_dir)
    model_fpath = model_dir.joinpath(model_dir.stem).with_suffix(".pt")
    print("\nLoading weights at %s" % model_fpath)
    model.load(model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    # Set model to eval mode (disable gradient and zoneout)
    model.eval()

    # Initialize the dataset
    in_dir = Path(in_dir)
    metadata_fpath = in_dir.joinpath("train.txt")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir)
    data_loader = DataLoader(dataset,
                             collate_fn=lambda batch: collate_synthesizer(batch, model.r),
                             batch_size=hparams.synthesis_batch_size,
                             num_workers=2,
                             shuffle=False,
                             pin_memory=True)

    # Generate GTA mels
    meta_out_fpath = os.path.join(out_dir, "synthesized.txt")

    with open(meta_out_fpath, "w") as file:
        for i, (x, m, e, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            #x = text, m = mel, e = embed, idx = index (used later)
            x, m, e = x.to(device), m.to(device), e.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                _, mel, _ = data_parallel_workaround(model, x, m, e)
            else:
                _, mel, _ = model(x, m, e)

            for j, k in enumerate(idx):
                # Write the spectrogram to disk
                # Note: outputs mel-spectrogram files and target ones have same names, just different folders
                mel_filename = os.path.join(synth_dir, dataset.metadata[k][1])
                np.save(mel_filename, mel[j].detach().cpu().numpy().T, allow_pickle=False)

                # Write metadata into the synthesized file
                file.write("|".join(dataset.metadata[k]))
