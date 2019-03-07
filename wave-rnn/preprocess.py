from multiprocessing import Pool, cpu_count
import pickle, glob
from utils.display import *
from utils.dsp import *
from vlibs import fileio

source_dir = r'E:\Datasets\LibriSpeech\train-clean*'
extension = 'flac'

out_dir = r'E:\Datasets\Vocoder'
quant_path = fileio.join(out_dir, 'quant')
mel_path = fileio.join(out_dir, 'mel')

fileio.ensure_dir(out_dir)
fileio.resetdir(quant_path)
fileio.resetdir(mel_path)

def convert_file(path):
    wav = load_wav(path, encode=False)
    mel = melspectrogram(wav)
    quant = (wav + 1.) * (2 ** 9 - 1) / 2
    return mel.astype(np.float32), quant.astype(np.int)

def process_wav(path):
    fname = fileio.leaf(path)
    fname = fname[:fname.rfind('.')]
    m, x = convert_file(path)
    np.save(fileio.join(mel_path, fname + ".npy"), m)
    np.save(fileio.join(quant_path, fname + ".npy"), x)
    return fname

def main():
    def get_files(path):
        filenames = []
        for filename in glob.iglob(f'{path}/**/*.{extension}', recursive=True):
            filenames += [filename]
        return filenames
    
    wav_files = get_files(source_dir)

    # This will take a while depending on size of dataset
    pool = Pool(processes=cpu_count())
    dataset_ids = []
    for i, fname in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        dataset_ids += [fname]
        stream('Processing: %i/%i', (i, len(wav_files)))
    
    with open(fileio.join(out_dir, 'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)

if __name__ == '__main__':
    main()