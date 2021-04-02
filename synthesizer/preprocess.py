from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
from synthesizer.utils.alignment import get_alignment


def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int,
                       skip_existing: bool, hparams,
                       datasets_name: str, subfolders: str, start: int, end: int):
    # Gather the input directories
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(
        subfolder.strip()) for subfolder in subfolders.split(",")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)
    out_dir.joinpath("duration").mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open(
        "a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*")
                                            for input_dir in input_dirs))[start:end]
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing,
                   hparams=hparams)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[5]) for m in metadata])
    timesteps = sum([int(m[4]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" %
          max(len(m[6]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[5]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[4]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams):
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        # Gather the utterance audios and texts
        # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
        extensions = ["*.wav", "*.flac", "*.mp3"]
        for extension in extensions:
            wav_fpaths = book_dir.glob(extension)
            for i, wav_fpath in enumerate(wav_fpaths):
                phoneme_fpaths = wav_fpath.with_suffix(".lab")
                if not phoneme_fpaths.exists():
                    continue
                else:
                    with phoneme_fpaths.open("r") as phoneme_file:
                        phone, duration, start, end = get_alignment(
                            phoneme_file)
                        # '{A}{B}{$}{C}', $ represents silent phones
                        phoneme = '{' + '}{'.join(phone) + '}'
                        phoneme = phoneme.replace(
                            '{$}', ' ')    # '{A}{B} {C}'
                        phoneme = phoneme.replace(
                            '}{', ' ')     # '{A B} {C}'
                        # Load the audio waveform
                        if start >= end:
                            phoneme = ''
                        wav, _ = librosa.load(
                            str(wav_fpath), hparams.sample_rate)
                        wav = wav[int(
                            hparams.sample_rate*start):int(hparams.sample_rate*end)].astype(np.float32)
                        # Get the corresponding text
                        # Check for .txt (for compatibility with other datasets)
                        text_fpath = wav_fpath.with_suffix(
                            ".normalized.txt")
                        if not text_fpath.exists():
                            text_fpath = wav_fpath.with_suffix(".txt")
                            assert text_fpath.exists()
                        with text_fpath.open("r") as text_file:
                            text = "".join([line for line in text_file])
                            text = text.replace("\"", "")
                            text = text.strip()
                        # Process the utterance
                        metadata.append(process_utterance(wav, text, phoneme, duration, out_dir, str(
                            wav_fpath.with_suffix("").name), skip_existing, hparams))

    return [m for m in metadata if m is not None]


def process_utterance(wav: np.ndarray, text: str, phoneme: str, duration: list, out_dir: Path, basename: str,
                      skip_existing: bool, hparams):
    # FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    duration_fpath = out_dir.joinpath("duration", "duration-%s.npy" % basename)

    if skip_existing and mel_fpath.exists() and wav_fpath.exists() and duration_fpath.exists():
        return None
    if len(phoneme.split(' ')) != len(duration):
        return None

    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)[
        :, :sum(duration)]
    mel_frames = mel_spectrogram.shape[1]

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    np.save(duration_fpath, duration, allow_pickle=False)
    # # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, duration_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text, phoneme


def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int, start: int, end: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    # print(metadata_fpath)
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file][start:end]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[3]))
                  for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
