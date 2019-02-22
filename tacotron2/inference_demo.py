from datasets.audio import inv_mel_spectrogram
from tacotron import synthesizer
from hparams import hparams
from vlibs import fileio
import sounddevice as sd
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_speaker_embed(speaker_id):
    embed_root = r"E:\Datasets\Synthesizer\embed"
    embeds = [np.load(f) for f in fileio.get_files(embed_root, "embed-%d-" % speaker_id)]
    speaker_embed = np.mean(embeds, axis=0)
    speaker_embed /= np.linalg.norm(speaker_embed, 2)
    return speaker_embed[None, ...]

if __name__ == '__main__':
    checkpoint_dir = os.path.join('logs-conditioned', 'taco_pretrained')
    checkpoint_fpath = tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path

    synth = synthesizer.Synthesizer()
    synth.load(checkpoint_fpath, hparams)
    
    while True:
        speaker_id = int(input("Speaker ID: "))
        speaker_embed = get_speaker_embed(speaker_id)
        text = input("Text: ")
        mel = synth.my_synthesize(speaker_embed, text)
        wav = inv_mel_spectrogram(mel.T, hparams)
        sd.play(wav, 16000)
        sd.wait()
