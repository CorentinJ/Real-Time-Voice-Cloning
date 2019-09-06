import argparse
from pathlib import Path

import typing
import uuid
import logging

import numpy as np
import scipy.spatial.distance
import soundfile

from encoder.inference import Model as EncoderModel
from synthesizer.inference import Synthesizer
from vocoder.inference import Model as VocoderModel


_WAV_FODLER = Path('/Users/dalei/Downloads/VCTK-Corpus/wav48')
_TXT_FODLER = Path('/Users/dalei/Downloads/VCTK-Corpus/txt')


def run(args: argparse.Namespace):
    # Load encoder model
    encoder = EncoderModel()
    encoder.load(Path('encoder/saved_models/pretrained.pt'))
    # Synthesize the spectrogram
    synthesizer = Synthesizer(Path('synthesizer/saved_models/logs-pretrained/taco_pretrained'))
    # Load vocoder
    vocoder = VocoderModel()
    vocoder.load_from(Path('vocoder/saved_models/pretrained/pretrained.pt'), verbose=False)

    # [p304, p305, ...]
    speaker_dirs = [f.parts[-1] for f in _WAV_FODLER.glob("*") if f.is_dir()]
    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory")

    # 'p304' -> [001.wav, 002.wav, ...]
    speaker_utterances = dict()  # type: typing.Dict[str, typing.List[str]]
    for d in speaker_dirs:
        speaker_utterances[d] = [w.parts[-1] for w in _WAV_FODLER.joinpath(d).glob('*.wav')]

    speaker_embeddings = dict()  # type: typing.Dict[str, np.ndarray]
    for d in speaker_utterances:
        utterances = speaker_utterances[d]
        enrollments = utterances[:3]
        logging.error(f'speaker: {d}, enrollments: {enrollments}')
        speaker_embeddings[d] = encoder.embed_speaker([_WAV_FODLER.joinpath(d, u) for u in enrollments])

    # Same speaker attack
    for d in speaker_utterances:
        utterances = speaker_utterances[d]
        # Repeat 5 times
        for utterance in np.random.choice(utterances, size=5, replace=False):  # type: str
            # generated audio
            txt = _TXT_FODLER.joinpath(d, utterance).with_suffix('.txt')
            text = txt.read_text()

            # original audio
            utterance_embedding = encoder.embed_utterance(_WAV_FODLER.joinpath(d, utterance), source_sr=Synthesizer.sample_rate)
            cosine_similarity = 1.0 - scipy.spatial.distance.cosine(speaker_embeddings[d], utterance_embedding)
            logging.error(f'ori: speaker: {d}, utterance: {utterance}, text: {text}, sim: {cosine_similarity}')

            specs = synthesizer.synthesize_spectrograms([text], [speaker_embeddings[d]])
            spec = np.concatenate(specs, axis=1)
            wav = vocoder.infer_waveform(spec)

            utterance_embedding = encoder.embed_utterance(wav, source_sr=Synthesizer.sample_rate)
            cosine_similarity = 1.0 - scipy.spatial.distance.cosine(speaker_embeddings[d], utterance_embedding)
            logging.error(f'gen: speaker: {d}, utterance: {utterance}, text: {text}, sim: {cosine_similarity}')

            # Save wav
            filename = f'/tmp/gen-{d}-{utterance}'
            soundfile.write(filename, wav, Synthesizer.sample_rate, 'PCM_16')
            logging.error(f"Saved audio to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='info',
        type=str,
        help='log level, debug, info, warning, error',
    )
    args, _ = parser.parse_known_args()

    run(args)

