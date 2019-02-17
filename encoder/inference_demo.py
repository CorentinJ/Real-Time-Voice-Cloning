from vlibs import fileio
from time import perf_counter
from encoder import inference
from config import *
from encoder.params_data import sampling_rate

if __name__ == '__main__':
    fpath = r"E:\Datasets\LibriSpeech\train-other-500\149\125760\149-125760-0003.flac"
    wave = inference.load_and_preprocess_wave(fpath)
    
    model_fpath = fileio.join(model_dir, "all.pt") 
    start = perf_counter()
    inference.load_model(model_fpath, device)
    print("Loaded model in %.2fs" % (perf_counter() - start))
    
    duration = len(wave) / sampling_rate
    start = perf_counter()
    embed = inference.embed_utterance(wave)
    print("Processed %.2fs long utterance in %.2fs" % (duration, perf_counter() - start))
    
    start = perf_counter()
    embed = inference.embed_utterance(wave)
    print("Processed %.2fs long utterance in %.2fs" % (duration, perf_counter() - start))
    
    print(embed)
