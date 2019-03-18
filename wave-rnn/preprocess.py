# from multiprocessing import Pool, cpu_count
import pickle
from utils.display import *
from utils.dsp import *
from vlibs import fileio

bits = 9
source_dir = r'E:\Datasets\Synthesizer\audio'

out_dir = r'E:\Datasets\Vocoder'
quant_path = fileio.join(out_dir, 'quant')
mel_path = fileio.join(out_dir, 'mel')

fileio.ensure_dir(out_dir)
fileio.ensure_dir(quant_path)
fileio.ensure_dir(mel_path)

def load_sample(wav_path, mel_path):
    # Load the wav and quantize it
    wav = np.load(wav_path)
    quant = (wav + 1.) * (2 ** 9 - 1) / 2
    
    # Load the mel spectrogram and adjust its range to [0, 1] 
    mel = np.load(mel_path)
    assert np.abs(mel) <= mel_max_abs_value
    mel = mel / (mel_max_abs_value * 2) + 0.5
    
    return mel.astype(np.float32), quant.astype(np.int)
    
def main():
    wav_files = fileio.get_files(source_dir, "\.npy", recursive=True)

    # This will take a while depending on size of dataset
    dataset_ids = []
    process_wav(wav_files[0])
    pool = Pool(processes=cpu_count())
    for i, fname in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        dataset_ids += [fname]
        stream('Processing: %i/%i', (i, len(wav_files)))
    
    with open(fileio.join(out_dir, 'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)

if __name__ == '__main__':
    main()