import os
from vlibs import fileio
import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize
import sys
sys.path.append('../encoder')
encoder_model_fpath = '../encoder/saved_models/all.pt'
from encoder import inference


def build_from_path(hparams, input_dirs, mel_dir, embed_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- embed_dir: output directory of the utterance embeddings
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	data = []
	index = 1
	inference.load_model(encoder_model_fpath, 'cuda')
	for input_dir in input_dirs:
		for speaker_dir in fileio.listdir(input_dir, full_path=True):
			for utterance_dir in fileio.listdir(speaker_dir, full_path=True):
				meta_file = fileio.get_files(utterance_dir, '.trans.txt')[0]
				with open(meta_file, encoding='utf-8') as f:
					for line in f:
						line = line.strip()
						split = line.find(' ')
						basename = line[:split]
						wav_path = fileio.join(utterance_dir, basename + '.flac')
						text = line[split + 1:]
						data.append(_process_utterance(mel_dir, embed_dir, wav_dir, basename, 
													   wav_path, text, hparams))
						if index % 100 == 0:
							print('Processed %d utterances' % index)
						index += 1
						
	n_all_samples = len(data)
	data = [d for d in data if d is not None]
	n_remaining_samples = len(data)
	print("Processed %d samples, pruned %d (remaining: %d)" % 
		  (n_all_samples, n_all_samples - n_remaining_samples, n_remaining_samples))
	return data



def _process_utterance2(mel_dir, embed_dir, wav_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- embed_dir: the directory to write the embedding into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text)
	"""
	wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	
	# rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max
	
	# M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)
	
	# Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		# [0, quantize_channels)
		out = mulaw_quantize(wav, hparams.quantize_channels)
		
		# Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start: end]
		out = out[start: end]
		
		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16
	
	elif is_mulaw(hparams.input_type):
		# [-1, 1]
		out = mulaw(wav, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32
	
	else:
		# [-1, 1]
		out = wav
		constant_values = 0.
		out_dtype = np.float32
	
	### SV2TTS ###
	# Compute the embedding of the utterance
	embed = inference.embed_utterance(wav)
	
	##############
	
	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]
	
	if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		return None
	
	if hparams.use_lws:
		# Ensure time resolution adjustement between audio and mel-spectrogram
		fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
		l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))
		
		# Zero pad audio signal
		out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	else:
		# Ensure time resolution adjustement between audio and mel-spectrogram
		pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))
		
		# Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
		out = np.pad(out, pad, mode='reflect')
	
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)
	
	# time resolution adjustement
	# ensure length of raw audio is multiple of hop size so that we can use
	# transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)
	
	# Write the spectrogram, embed and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	embed_filename = 'embed-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(embed_dir, embed_filename), embed, allow_pickle=False)
	
	# Return a tuple describing this training example
	return (audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text)


### TEMP
def _process_utterance(mel_dir, embed_dir, wav_dir, index, wav_path, text, hparams):
	mel_filename = 'mel-{}.npy'.format(index)
	try:
		mel_spectrogram = np.load(os.path.join(mel_dir, mel_filename))
	except:
		return _process_utterance2(mel_dir, embed_dir, wav_dir, index, wav_path, text, hparams)
	mel_frames = mel_spectrogram.shape[0]
	
	audio_filename = 'audio-{}.npy'.format(index)
	out = np.load(os.path.join(wav_dir, audio_filename))
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)
	
	embed_filename = 'embed-{}.npy'.format(index)
	
	return (audio_filename, mel_filename, embed_filename, time_steps, mel_frames, text)
