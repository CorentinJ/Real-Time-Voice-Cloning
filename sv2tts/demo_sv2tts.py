from synthesizer.audio import inv_mel_spectrogram
from synthesizer.hparams import hparams
from synthesizer import synthesizer
import sounddevice as sd
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from vocoder import inference as vocoder
from encoder import inference as encoder

encoder.load_model("SV2TTS/encoder/saved_models/all.pt")
vocoder.load_model("../wave-rnn/checkpoints/mu_law.pt")
    

def get_random_embed():
    root = r"C:\Datasets\LibriSpeech\test-clean"
    speakers = fileio.listdir(root)
    speaker_id = np.random.choice(speakers)
    speaker_root = fileio.join(root, speaker_id)
    wav_fpaths = fileio.get_files(speaker_root, "\.flac", recursive=True)
    wav_fpath = np.random.choice(wav_fpaths)
    print(wav_fpath)
    wav = encoder.load_and_preprocess_wave(wav_fpath)
    print("Source audio (5 secs max):")
    sd.play(wav[:16000 * 5], 16000)
    sd.wait()
    embed = encoder.embed_utterance(wav)[None, ...]
    return embed, speaker_id, wav

if __name__ == "__main__":
    checkpoint_dir = os.path.join("logs-two_asr", "taco_pretrained")
    checkpoint_fpath = tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path

    synth = synthesizer.Synthesizer()
    synth.load(checkpoint_fpath, hparams)
    from datasets.audio import save_wav
    
    user = True
    i = 0
    
    while True:
        # Retrieve the embedding
        if user:
            from encoder.audio import rec_wave, preprocess_wave
            from time import sleep
            print("Watch out, recording in 2 seconds...")
            sleep(2)
            print("Recording 5 seconds!")
            wav_source = preprocess_wave(rec_wave(5))
            print("Done!", end=" ")
            sleep(1)
            print("Here is your audio:")
            sd.play(wav_source, 16000)
            sd.wait()
            speaker_id = "user_%02d" % i
            i += 1
            speaker_embed = encoder.embed_utterance(wav_source)[None, ...]
        else:
            speaker_embed, speaker_id, wav_source = get_random_embed()
            print(speaker_id)

        # Synthesize the text with the embedding
        text = input("Text: ")
        mel = synth.my_synthesize(speaker_embed, text)
        
        wav_griffin = inv_mel_spectrogram(mel.T, hparams)
        wav_griffin = np.concatenate((wav_griffin, [0] * hparams.sample_rate))
        print("Griffin-lim:")
        sd.play(wav_griffin, 16000)
        
        wav_wavernn = vocoder.infer_waveform(mel.T)
        wav_wavernn = np.concatenate((wav_wavernn, [0] * hparams.sample_rate))
        sd.wait()
        print("\nWave-RNN:")
        sd.play(wav_wavernn, 16000)
        sd.wait()

        save_wav(wav_source, "../%s_%s.wav" % (speaker_id, "source"), 16000)
        save_wav(wav_griffin, "../%s_%s.wav" % (speaker_id, "griffin"), 16000)
        save_wav(wav_wavernn, "../%s_%s.wav" % (speaker_id, "wavernn"), 16000)


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
        #     print("")
        #     
        # # Pad the end of the waveform
        # wav = np.concatenate((wav, [0] * hparams.sample_rate))
        # 
        # # Play the audio
        # sd.play(wav, 16000)
        # sd.wait()
