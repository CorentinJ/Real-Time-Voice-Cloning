from datasets.speaker_batch import SpeakerBatch
from datasets.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from params import partial_utterance_length
from vlibs import fileio
from config import *
import numpy as np
import random

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets, speakers_per_batch, utterances_per_speaker):
        self.datasets = datasets
        self.utterances_per_speaker = utterances_per_speaker
        self.speakers_per_batch = speakers_per_batch
        
        self.speakers = []
        for dataset in datasets:
            dataset_root = fileio.join(clean_data_root, dataset) 
            speaker_dirs = fileio.join(dataset_root, fileio.listdir(dataset_root))[:10]
            self.speakers.extend(Speaker(speaker_dir) for speaker_dir in speaker_dirs)

        self.mean_n_utterances = np.mean([len(s.utterances) for s in self.speakers])
        print('Dataset of %d speakers with %d utterances per speaker on average' % 
              (len(self.speakers), self.mean_n_utterances))

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        speakers = random.sample(self.speakers, self.speakers_per_batch)
        batch = SpeakerBatch(speakers, self.utterances_per_speaker, partial_utterance_length)
        return batch
    
    def collate(batches):
        return batches[0]
    
    def get_logs(self):
        log_string = ""
        for dataset in self.datasets:
            log_fpath = fileio.join(clean_data_root, clean_data_root, "log_%s.txt" % dataset)
            log_string += "\n".join(fileio.read_all_lines(log_fpath))
        return log_string
    
    def get_params(self):
        params = OrderedDict([
            ("Total speakers", len(self.speakers)),
            ("Average utterances per speaker", self.mean_n_utterances),
            ("Speakers per batch", self.speakers_per_batch),
            ("Utterances per speaker", self.utterances_per_speaker),
            ("Datasets", ','.join(self.datasets)),
        ])
        return params
    