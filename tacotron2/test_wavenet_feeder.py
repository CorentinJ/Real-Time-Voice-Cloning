import numpy as np 
import os
import argparse 
from hparams import hparams
from datasets import audio
from tqdm import tqdm



def _limit_time(hparams):
	'''Limit time resolution to save GPU memory.
	'''
	if hparams.max_time_sec is not None:
		return int(hparams.max_time_sec * hparams.sample_rate)
	elif hparams.max_time_steps is not None:
		return hparams.max_time_steps
	else:
		return None


def get_groups(args, hparams, meta, local_condition):
	if hparams.train_with_GTA:
		mel_file = meta[2]
	else:
		mel_file = meta[1]
	audio_file = meta[0]

	input_data = np.load(os.path.join(args.base_dir, audio_file))

	if local_condition:
		local_condition_features = np.load(os.path.join(args.base_dir, mel_file))
	else:
		local_condition_features = None

	return (input_data, local_condition_features, None, len(input_data))

def _adjust_time_resolution(hparams, batch, local_condition, max_time_steps):
		'''Adjust time resolution between audio and local condition
		'''
		if local_condition:
			new_batch = []
			for b in batch:
				x, c, g, l = b
				_assert_ready_for_upsample(hparams, x, c)
				if max_time_steps is not None:
					max_steps = _ensure_divisible(max_time_steps, audio.get_hop_size(hparams), True)
					if len(x) > max_time_steps:
						max_time_frames = max_steps // audio.get_hop_size(hparams)
						start = np.random.randint(0, len(c) - max_time_frames)
						time_start = start * audio.get_hop_size(hparams)
						x = x[time_start: time_start + max_time_frames * audio.get_hop_size(hparams)]
						c = c[start: start + max_time_frames, :]
						_assert_ready_for_upsample(hparams, x, c)

				new_batch.append((x, c, g, l))
			return new_batch
		else:
			new_batch = []
			for b in batch:
				x, c, g, l = b
				x = audio.trim_silence(x, hparams)
				if max_time_steps is not None and len(x) > max_time_steps:
					start = np.random.randint(0, len(c) - max_time_steps)
					x = x[start: start + max_time_steps]
				new_batch.append((x, c, g, l))
			return new_batch

def _assert_ready_for_upsample(hparams, x, c):
	assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size(hparams)

def check_time_alignment(hparams, batch, local_condition):
	#No need to check beyond this step when preparing data
	#Limit time steps to save GPU Memory usage
	max_time_steps = _limit_time(hparams)
	#Adjust time resolution for upsampling
	batch = _adjust_time_resolution(hparams, batch, local_condition, max_time_steps)

def _ensure_divisible(length, divisible_by=256, lower=True):
	if length % divisible_by == 0:
		return length
	if lower:
		return length - length % divisible_by
	else:
		return length + (divisible_by - length % divisible_by)

def run(args, hparams):
	with open(args.metadata, 'r') as file:
		metadata = [line.strip().split('|') for line in file]

	local_condition = hparams.cin_channels > 0

	examples = [get_groups(args, hparams, meta, local_condition) for meta in metadata]
	batches = [examples[i: i+hparams.wavenet_batch_size] for i in range(0, len(examples), hparams.wavenet_batch_size)]

	for batch in tqdm(batches):
		check_time_alignment(hparams, batch, local_condition)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--metadata', default='tacotron_output/gta/map.txt')
	args = parser.parse_args()

	modified_hparams = hparams.parse(args.hparams)
	run(args, modified_hparams)


if __name__ == '__main__':
	main()