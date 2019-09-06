import io
import logging
import typing

import librosa
import numpy

import encoder.audio
from encoder.model import SpeakerEncoder
from encoder.params_data import *
from matplotlib import cm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


class Model(object):
    # shared by all instances of the class
    models = {}  # type: typing.Dict[str, SpeakerEncoder]
    devices = {}  # type: typing.Dict[str, torch.device]

    def __init__(self):
        pass

    def load(self, weights_fpath: Path, device=None, model_name='default'):
        """
        Loads the model in memory. If this function is not explicitely called, it will be run on the
        first call to embed_frames() with the default weights file.

        :param weights_fpath: the path to saved model weights.
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The
        model will be loaded and will run on this device. Outputs will however always be on the cpu.
        If None, will default to your GPU if it"s available, otherwise your CPU.
        :param model_name: an identifier to uniquely identify a model in order to avoid loading
        the same model more than once
        """
        if model_name in self.models:
            return

        # TODO: I think the slow loading of the encoder might have something to do with the device it
        #   was saved on. Worth investigating.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        model = SpeakerEncoder(device, torch.device("cpu"))
        checkpoint = torch.load(weights_fpath, map_location=None if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logging.info('Loaded encoder {} trained to step {}'.format(weights_fpath.name, checkpoint["step"]))

        self.models[model_name] = model
        self.devices[model_name] = device

    def is_loaded(self, model_name='default'):
        return model_name in self.models and self.models[model_name] is not None

    def embed_frames_batch(self, frames_batch, model_name='default'):
        """
        Computes embeddings for a batch of mel spectrogram.

        :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
        (batch_size, n_frames, n_channels)
        :param model_name: the unique identifier of a model to use
        :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise Exception("Model was not loaded. Call load_model() before inference.")

        model = self.models[model_name]  # type: SpeakerEncoder
        device = self.devices[model_name]  # type: torch.device
        frames = torch.from_numpy(frames_batch).to(device)
        embed = model.forward(frames).detach().cpu().numpy()
        return embed

    @staticmethod
    def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                               min_pad_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(
            self,
            wav_file: typing.Union[str, Path, io.BytesIO, np.ndarray],
            using_partials=True,
            return_partials=False,
            source_sr=None,
            model_name='default',
            **kwargs
    ) -> numpy.ndarray:
        """
        Computes an embedding for a single utterance.

        # TODO: handle multiple wavs to benefit from batching on GPU
        :param wav_file: a filepath to an audio file (many extensions are supported, not just .wav).
        :param using_partials: if True, then the utterance is split in partial utterances of
        <partial_utterance_n_frames> frames and the utterance embedding is computed from their
        normalized average. If False, the utterance is instead computed from feeding the entire
        spectogram to the network.
        :param return_partials: if True, the partial embeddings will also be returned along with the
        wav slices that correspond to the partial embeddings.
        :param model_name: the unique identifier of a model
        :param kwargs: additional arguments to compute_partial_splits()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned. If <using_partials> is simultaneously set to False, both these values will be None
        instead.
        """
        # Load the wav from disk if needed
        if isinstance(wav_file, str) or isinstance(wav_file, Path) or isinstance(wav_file, io.BytesIO):
            wav, source_sr = librosa.load(wav_file, sr=None)
        elif isinstance(wav_file, np.ndarray):
            wav = wav_file
        else:
            raise ValueError('unsupported format of wav_file')

        # preprocess utterance waveform to a numpy array of float32
        wav = encoder.audio.preprocess_wav(wav, source_sr)

        # Process the entire utterance if not using partials
        if not using_partials:
            frames = encoder.audio.wav_to_mel_spectrogram(wav)
            embed = self.embed_frames_batch(frames[None, ...])[0]
            if return_partials:
                return embed, None, None
            return embed

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        frames = encoder.audio.wav_to_mel_spectrogram(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        partial_embeds = self.embed_frames_batch(frames_batch)

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed

    def embed_speaker(
            self,
            wavs: typing.Iterable[typing.Union[str, Path, io.BytesIO]],
            model_name='default',
            **kwargs
    ) -> np.ndarray:
        embeds = []
        for wav in wavs:
            embed = self.embed_utterance(wav, model_name=model_name)
            embed /= np.linalg.norm(embed)
            embeds.append(embed)

        return np.mean(embeds, axis=0)

    @staticmethod
    def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
        if ax is None:
            ax = plt.gca()

        if shape is None:
            height = int(np.sqrt(len(embed)))
            shape = (height, -1)
        embed = embed.reshape(shape)

        cmap = cm.get_cmap()
        mappable = ax.imshow(embed, cmap=cmap)
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.mappable.set_clim(*color_range)

        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title)
