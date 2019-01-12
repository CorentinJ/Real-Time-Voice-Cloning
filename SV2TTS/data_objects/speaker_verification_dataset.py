from data_objects.speaker_batch import SpeakerBatch
from data_objects.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from params import partial_utterance_length
from vlibs.structs.random_cycler import RandomCycler
from vlibs import fileio
from config import *
import numpy as np
import random

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.speakers = []
        for dataset in datasets:
            dataset_root = fileio.join(clean_data_root, dataset) 
            speaker_dirs = fileio.join(dataset_root, fileio.listdir(dataset_root))[:10]
            self.speakers.extend(Speaker(speaker_dir) for speaker_dir in speaker_dirs)
        self.speaker_cycler = RandomCycler(self.speakers)
        self.mean_n_utterances = np.mean([len(s.utterances) for s in self.speakers])

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
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
            ("Datasets", ','.join(self.datasets)),
        ])
        return params
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partial_utterance_length) 