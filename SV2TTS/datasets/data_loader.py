from torch.utils.data import Dataset, DataLoader
from datasets.speaker import Speaker
from datasets.speaker_batch import SpeakerBatch
from vlibs import fileio
from config import *
from params import partial_utterance_length
import random

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets, speakers_per_batch, utterances_per_speaker):
        self.utterances_per_speaker = utterances_per_speaker
        self.speakers_per_batch = speakers_per_batch
        
        self.speakers = []
        for dataset in datasets:
            dataset_root = fileio.join(clean_data_root, dataset) 
            speaker_dirs = fileio.join(dataset_root, fileio.listdir(dataset_root))
            self.speakers.extend(Speaker(speaker_dir) for speaker_dir in speaker_dirs)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        speakers = random.sample(self.speakers, self.speakers_per_batch)
        batch = SpeakerBatch(speakers, self.utterances_per_speaker, partial_utterance_length)
        return batch
    
    def collate(batches):
        return batches[0]
    