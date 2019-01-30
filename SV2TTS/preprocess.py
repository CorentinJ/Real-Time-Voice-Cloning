from vlibs.ui import console
from vlibs import fileio
from config import *
import audio
import numpy as np
from datetime import datetime
from params_data import *
from pathos.multiprocessing import ThreadPool

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
        self._log_params()
        
    def _log_params(self):
        import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith('__')):
            value = getattr(params_data, param_name)
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
        
def preprocess_wave(wave):
    """ 
    This is the standard routine that should be used on every audio file before being used in 
    this project.
    """
    wave = audio.normalize_volume(wave, audio_norm_target_dBFS, increase_only=True)
    wave = audio.trim_long_silences(wave)
    return wave

def preprocess_librispeech(n_speakers=None, n_utterances=None):
    fileio.ensure_dir(clean_data_root)
    
    for dataset_name in librispeech_datasets:
        dataset_root = fileio.join(librispeech_root, dataset_name)
        out_dir = fileio.ensure_dir(fileio.join(clean_data_root, dataset_name))
        logger = DatasetLog(clean_data_root, dataset_name)
        
        # Get the speaker directories
        speaker_ids = fileio.listdir(dataset_root, numerical_sorting=True)[:n_speakers]
        print("Librispeech: Preprocessing data for %d speakers." % len(speaker_ids))
        
        # Function to preprocess utterances for one speaker
        def preprocess_speaker(speaker_id):
            print("Starting speaker %s" % speaker_id)
            speaker_name = "LibriSpeech_%s_%s" % (dataset_name, speaker_id)
            speaker_in_dir = fileio.join(dataset_root, speaker_id)
            speaker_out_dir = fileio.ensure_dir(fileio.join(out_dir, speaker_name))
            fileio.resetdir(speaker_out_dir)
            sources_file = open(fileio.join(speaker_out_dir, "sources.txt"), 'w')
            
            fpaths = fileio.get_files(speaker_in_dir, r"\.flac", recursive=True)[:n_utterances]
            for i, in_fpath in enumerate(fpaths):
                # Load and preprocess the waveform
                wave = audio.load(in_fpath)
                wave = preprocess_wave(wave)
                
                # Create and save the mel spectrogram
                frames = audio.wave_to_mel_filterbank(wave)
                if len(frames) < partial_utterance_length:  
                    continue
                fname = fileio.leaf(in_fpath).replace(".flac", ".npy")
                out_fpath = fileio.join(speaker_out_dir, fname)
                np.save(out_fpath, frames)
                
                logger.add_sample(duration=len(wave) / sampling_rate)
                sources_file.write("%s %s\n" % (fname, in_fpath))

            sources_file.close()
            print("Speaker %s done!" % speaker_id)

        # Process the utterances for each speaker
        with ThreadPool(8) as pool:
            list(pool.imap(preprocess_speaker, speaker_ids))
        logger.finalize()


def preprocess_voxceleb1(n_speakers=None, n_utterances=None):
    fileio.ensure_dir(clean_data_root)

    dataset_name = "voxceleb1"
    out_dir = fileio.ensure_dir(fileio.join(clean_data_root, dataset_name))
    logger = DatasetLog(clean_data_root, dataset_name)
    
    # Get the contents of the meta file
    metadata = fileio.read_all_lines(fileio.join(voxceleb1_root, "vox1_meta.csv"))[1:]
    metadata_fields = [line.split('\t') for line in metadata]
    
    # Select the ID and the nationality, filter out non-anglophone speakers
    nationalities = {line[0]: line[3] for line in metadata_fields}
    speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                   nationality.lower() in anglophone_nationalites]
    speaker_ids = speaker_ids[:n_speakers]
    print("VoxCeleb1: using samples from %d (assumed anglophone) speakers out of %d." % 
          (len(speaker_ids), len(nationalities)))
    
    # Get the speaker directories
    speakers_root = fileio.join(voxceleb1_root, "wav")
    disk_speaker_ids = fileio.listdir(speakers_root)
    speaker_ids_len = len(speaker_ids)
    speaker_ids = list(filter(lambda s_id: s_id in disk_speaker_ids, speaker_ids))
    print("Found %d speakers on the disk, %d missing (this is normal)." % 
          (len(speaker_ids), speaker_ids_len - len(speaker_ids)))
    print("Preprocessing data for %d speakers." % len(speaker_ids))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_id):
        print("Starting speaker %s" % speaker_id)
        speaker_name = "VoxCeleb1_%s" % speaker_id
        speaker_in_dir = fileio.join(speakers_root, speaker_id)
        speaker_out_dir = fileio.ensure_dir(fileio.join(out_dir, speaker_name))
        fileio.resetdir(speaker_out_dir)
        sources_file = open(fileio.join(speaker_out_dir, "sources.txt"), 'w')
        
        fpaths = fileio.get_files(speaker_in_dir, r"\.wav", recursive=True)[:n_utterances]
        for i, in_fpath in enumerate(fpaths):
            # Load and preprocess the waveform
            wave = audio.load(in_fpath)
            wave = preprocess_wave(wave)
            
            # Create and save the mel spectrogram
            frames = audio.wave_to_mel_filterbank(wave)
            if len(frames) < partial_utterance_length:
                continue
            video_id = fileio.leaf(fileio.leafdir(in_fpath))
            fname = video_id + '_' + fileio.leaf(in_fpath).replace(".wav", ".npy")
            out_fpath = fileio.join(speaker_out_dir, fname)
            np.save(out_fpath, frames)
            
            logger.add_sample(duration=len(wave) / sampling_rate)
            sources_file.write("%s %s\n" % (fname, in_fpath))
        
        sources_file.close()
        print("Speaker %s done!" % speaker_id)
        
    # Process the utterances for each speaker
    with ThreadPool(8) as pool:
        list(pool.imap(preprocess_speaker, sorted(speaker_ids)))
    logger.finalize()


if __name__ == '__main__':
    # preprocess_librispeech()
    preprocess_voxceleb1()
