from vlibs import fileio, console
from config import *
import audio
import numpy as np
from datetime import datetime

class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
    def __init__(self, root, name):
        self.text_file = open(fileio.join(root, "log_%s.txt" % name), 'w')
        self.sample_data = dict()
        
        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self.log_params()
        
    def log_params(self):
        import params
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params) if not p.startswith('__')):
            value = getattr(params, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")
    
    def write_line(self, line):
        self.text_file.write("%s\n" % line)
        
    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)
            
    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()
        

def preprocess_librispeech(n_speakers=None, n_utterances=None):
    fileio.ensure_dir(clean_data_root)
    
    for dataset_name in librispeech_datasets:
        dataset_root = fileio.join(librispeech_root, dataset_name)
        out_dir = fileio.ensure_dir(fileio.join(clean_data_root, dataset_name))
        logger = DatasetLog(clean_data_root, dataset_name)
        
        # Get the speaker directories
        speaker_ids = fileio.listdir(dataset_root, numerical_sorting=True)[:n_speakers]
        print("Librispeech: Preprocessing data for %d speakers." % len(speaker_ids))

        # Process the utterances for each speaker
        for speaker_id in speaker_ids:
            speaker_name = "LibriSpeech_%s_%s" % (dataset_name, speaker_id)
            speaker_in_dir = fileio.join(dataset_root, speaker_id)
            speaker_out_dir = fileio.ensure_dir(fileio.join(out_dir, speaker_name))
            sources_file = open(fileio.join(speaker_out_dir, "sources.txt"), 'w')
            
            fpaths = fileio.get_files(speaker_in_dir, r"\.flac", recursive=True)[:n_utterances]
            message = "\tProcessing %3d utterances from speaker %s" % (len(fpaths), speaker_id)
            for i, fpath in enumerate(fpaths):
                wave, sampling_rate = audio.load(fpath, 16000)
                logger.add_sample(duration=len(wave)/sampling_rate)
                frames = audio.wave_to_mel_filterbank(wave, sampling_rate)
                fname = fileio.leaf(fpath).replace(".flac", ".npy")
                out_fpath = fileio.join(speaker_out_dir, fname)
                np.save(out_fpath, frames)
                sources_file.write("%s %s\n" % (fname, out_fpath))
                console.progress_bar(message, i + 1, len(fpaths))

            sources_file.close()
                
        logger.finalize()
            
if __name__ == '__main__':
    preprocess_librispeech(10, 50)