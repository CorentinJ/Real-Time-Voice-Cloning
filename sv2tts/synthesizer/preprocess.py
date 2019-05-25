from synthesizer import audio
from multiprocessing.pool import Pool 
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa


def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing: bool, hparams):
    # Gather the input directories
    dataset_root = datasets_root.joinpath("LibriSpeech")
    input_dirs = [dataset_root.joinpath("train-clean-100"), 
                  dataset_root.joinpath("train-clean-360")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    
    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)
    
    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, 
                   hparams=hparams)
    job = Pool().imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, "LibriSpeech", len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

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


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams):
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        # Gather the utterance audios and texts
        try:
            alignments_fpath = next(book_dir.glob("*.alignment.txt"))
            with alignments_fpath.open("r") as alignments_file:
                alignments = [line.rstrip().split(" ") for line in alignments_file]
        except StopIteration:
            # A few alignment files will be missing
            continue
        
        # Iterate over each entry in the alignments file
        for wav_fname, words, end_times in alignments:
            wav_fpath = book_dir.joinpath(wav_fname + ".flac")
            assert wav_fpath.exists()
            words = words.replace("\"", "").split(",")
            end_times = list(map(float, end_times.replace("\"", "").split(",")))
            
            # Process each sub-utterance
            wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
            for i, (wav, text) in enumerate(zip(wavs, texts)):
                sub_basename = "%s_%02d" % (wav_fname, i)
                metadata.append(process_utterance(wav, text, out_dir, sub_basename, 
                                                  skip_existing, hparams))
                
    return [m for m in metadata if m is not None]


def split_on_silences(audio_fpath, words, end_times, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(audio_fpath, hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == '' and words[-1] == ''
    
    # Break the sentence on pauses that are too long
    mask = (words == '') & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]
    segment_times = [[end_times[s], start_times[e]] for s, e in zip(breaks[:-1], breaks[1:])]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [' '.join(words[s + 1:e]).replace("  ", " ") for s, e in zip(breaks[:-1], breaks[1:])]
    
    ## DEBUG: play the audio segments
    # import sounddevice as sd
    # print("From %s" % audio_fpath)
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")
    
    return wavs, texts
    
    
def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, 
                      skip_existing: bool, hparams):
    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    
    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text
 
 
def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)
    
 
def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)
    
    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
        
    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
    