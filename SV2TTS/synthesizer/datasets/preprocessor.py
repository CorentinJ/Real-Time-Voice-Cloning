from synthesizer.datasets import audio
from encoder import inference as speaker_encoder
from pathlib import Path
import numpy as np


def preprocess_librispeech(datasets_root: Path, out_dir: Path, wav_out_dir: Path,
                           encoder_model_fpath: Path, skip_existing: bool, hparams):
    # Gather the input directories
    dataset_root = datasets_root.joinpath("LibriTTS")
    input_dirs = [dataset_root.joinpath("train-clean-100"), 
                  dataset_root.joinpath("train-clean-360")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    
    # Create the output directories for each output file type
    mel_out_dir = out_dir.joinpath("mels")
    embed_out_dir = out_dir.joinpath("embed")
    wav_out_dir = wav_out_dir.joinpath("audio")
    mel_out_dir.mkdir(exist_ok=True)
    embed_out_dir.mkdir(exist_ok=True)
    wav_out_dir.mkdir(exist_ok=True)
    
    # Load the speaker encoder
    print("Using the speaker encoder model weights located at:\n\t%s" % encoder_model_fpath)
    speaker_encoder.load_model(encoder_model_fpath)
    print("Speaker encoder loaded. Make sure to keep a backup of that model for inference.\n")
    
    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")
    
    # Preprocess the dataset
    print("Preprocessing utterances:")
    total_sample_count = 0
    remaining_sample_count = 0
    for input_dir in input_dirs:
        print("[%s]" % input_dir.stem)
        for speaker_dir in list(input_dir.glob("*"))[:10]:  # TODO
            print("    Speaker %s" % speaker_dir.stem)
            for book_dir in speaker_dir.glob("*"):
                # Gather the utterance audios and texts
                text_fpaths = list(book_dir.glob("*.normalized.txt"))
                wav_fpaths = list(book_dir.glob("*.wav"))
                assert len(text_fpaths) == len(wav_fpaths)
                
                # Preprocess each utterance individually
                for text_fpath, wav_fpath in zip(text_fpaths, wav_fpaths):
                    basename = wav_fpath.stem
                    with text_fpath.open("r") as text_file:
                        text = next(text_file).rstrip().lower()
                    result = _process_utterance(mel_out_dir, embed_out_dir, wav_out_dir, basename,
                                                wav_fpath, text, skip_existing, hparams)
                    
                    # Write the resulting sample metadata to the metadata file
                    if result is not None:
                        metadata_file.write("|".join(str(x) for x in result) + "\n")
                        remaining_sample_count += 1
                    total_sample_count += 1
    metadata_file.close()
    perc_samples_remaining = (remaining_sample_count / total_sample_count) * 100
    print("Found %d utterances. %d were processed (%.1f%%)." %
          (total_sample_count, remaining_sample_count, perc_samples_remaining))
    
    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))
    
    
def _process_utterance(mel_out_dir: Path, embed_out_dir: Path, wav_out_dir: Path, basename: str, 
                       wav_path: Path, text: str, skip_existing: bool, hparams):
    # Skip existing utterances if needed
    mel_fpath = mel_out_dir.joinpath("mel-%s.npy" % basename)
    embed_fpath = embed_out_dir.joinpath("embed-%s.npy" % basename)
    wav_fpath = wav_out_dir.joinpath("audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and embed_fpath.exists() and wav_fpath.exists():
        return None
    
    # Load the audio waveform from the disk
    wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    if hparams.rescale:
        wav = (wav / np.abs(wav).max()) * hparams.rescaling_max

    # Compute its mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    # Compute the speaker embedding of the utterance
    embed = speaker_encoder.embed_utterance(wav)
    
    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(embed_fpath, embed, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    
    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, embed_fpath.name, len(wav), mel_frames, text
 