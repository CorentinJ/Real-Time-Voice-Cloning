from datasets.audio import inv_mel_spectrogram
from tacotron import synthesizer
from hparams import hparams
from vlibs import fileio
import sounddevice as sd
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../wave-rnn')
from vocoder import inference as vocoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

use_griffin_lim = False
if not use_griffin_lim:
    vocoder.load_model('../wave-rnn/checkpoints/mu_law.pt')
    
all_embeds_fpaths = fileio.get_files(r"E:\Datasets\Synthesizer\embed", "embed")

def get_speaker_embed(speaker_id):
    embed_root = r"E:\Datasets\Synthesizer\embed"
    embeds = [np.load(f) for f in fileio.get_files(embed_root, "embed-%d-" % speaker_id)]
    speaker_embed = np.mean(embeds, axis=0)
    speaker_embed /= np.linalg.norm(speaker_embed, 2)
    return speaker_embed[None, ...]

def get_random_embed():
    fpath = np.random.choice(all_embeds_fpaths)
    return np.load(fpath)[None, ...], fpath

if __name__ == '__main__':
    checkpoint_dir = os.path.join('logs-two_asr', 'taco_pretrained')
    checkpoint_fpath = tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path

    synth = synthesizer.Synthesizer()
    synth.load(checkpoint_fpath, hparams)
    from datasets.audio import save_wav

    while True:
        # Retrieve the embedding
        # speaker_id = int(input("Speaker ID: "))
        # speaker_embed = get_speaker_embed(speaker_id)
        speaker_embed, embed_fpath = get_random_embed()
        print(embed_fpath)
        a = embed_fpath[embed_fpath.find('embed-')+6:]
        speaker_id = int(a[:a.find('-')])
        print(speaker_id)

        # Synthesize the text with the embedding
        text = input("Text: ")
        mel = synth.my_synthesize(speaker_embed, text)
        
        wav = inv_mel_spectrogram(mel.T, hparams)
        wav = np.concatenate((wav, [0] * hparams.sample_rate))
        print("Griffin-lim:")
        sd.play(wav, 16000)
        wav1 = wav
        
        
        
        wav = vocoder.infer_waveform(mel.T)
        wav = np.concatenate((wav, [0] * hparams.sample_rate))
        sd.wait()
        print("\nWave-RNN:")
        sd.play(wav, 16000)
        sd.wait()

        save_wav(wav1, "%s_%s.wav" % (speaker_id, 'griffin'), 16000)
        save_wav(wav, "%s_%s.wav" % (speaker_id, 'wavernn'), 16000)





        # # Synthesize the text with the embedding
        # speaker_embed = get_speaker_embed(speaker_id)
        # 
        # mel = synth.my_synthesize(speaker_embed, text)
        # 
        # wav = inv_mel_spectrogram(mel.T, hparams)
        # wav = np.concatenate((wav, [0] * hparams.sample_rate))
        # print("Griffin-lim:")
        # sd.play(wav, 16000)
        # 
        # wav = vocoder.infer_waveform(mel.T)
        # wav = np.concatenate((wav, [0] * hparams.sample_rate))
        # sd.wait()
        # print("\nWave-RNN:")
        # sd.play(wav, 16000)
        # sd.wait()


        # # Infer the waveform of the synthsized spectrogram
        # if use_griffin_lim:
        #     wav = inv_mel_spectrogram(mel.T, hparams)
        # else:
        #     wav = vocoder.infer_waveform(mel.T)
        #     print('')
        #     
        # # Pad the end of the waveform
        # wav = np.concatenate((wav, [0] * hparams.sample_rate))
        # 
        # # Play the audio
        # sd.play(wav, 16000)
        # sd.wait()
