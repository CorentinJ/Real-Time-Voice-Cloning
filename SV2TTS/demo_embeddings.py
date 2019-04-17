from time import perf_counter
from encoder import inference
from encoder.params_data import sampling_rate
from pathlib import Path
import numpy as np
import torch

if __name__ == "__main__":
    fpath = r"E:\Datasets\LibriSpeech\train-other-500\149\125760\149-125760-0003.flac"
    wav = inference.load_preprocess_waveform(fpath)

    models_dir = Path("encoder/saved_models")
    model_fpath = models_dir.joinpath("all.pt")
    torch.cuda.synchronize()
    
    start = perf_counter()
    inference.load_model(model_fpath)
    print("Loaded model in %.2fs" % (perf_counter() - start))
    torch.cuda.synchronize()
    
    duration = len(wav) / sampling_rate
    start = perf_counter()
    embed = inference.embed_utterance(wav)
    torch.cuda.synchronize()
    print("Processed %.2fs long utterance in %.2fs" % (duration, perf_counter() - start))
    
    start = perf_counter()
    embed = inference.embed_utterance(wav)
    torch.cuda.synchronize()
    print("Processed %.2fs long utterance in %.2fs" % (duration, perf_counter() - start))
    
    np.set_printoptions(precision=2, suppress=True)
    print(embed)
