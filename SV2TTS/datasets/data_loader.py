from torch.utils.data import Dataset, DataLoader
from datasets.speaker import Speaker
from datasets.speaker_batch import SpeakerBatch
from vlibs import fileio
from config import *
from params import partial_utterance_length
import random

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets, speakers_per_batch, utterances_per_speaker, test_split=0.8):
        self.utterances_per_speaker = utterances_per_speaker
        self.speakers_per_batch = speakers_per_batch
        
        self.speakers = []
        for dataset in datasets:
            dataset_root = fileio.join(clean_data_root, dataset) 
            speaker_dirs = fileio.join(dataset_root, fileio.listdir(dataset_root))
            self.speakers.extend(Speaker(speaker_dir, test_split) for speaker_dir in speaker_dirs)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        speakers = random.sample(self.speakers, self.speakers_per_batch)
        batch = SpeakerBatch(speakers, self.utterances_per_speaker, partial_utterance_length)
        return batch
    
    def collate(batches):
        # We don't want SpeakerBatches to be aggregated in a single numpy array because they do 
        # not have the same shape. By override the collate function with identity, they are 
        # aggregated in a list instead.
        return batches
    
    def test_data(self):
        return {s.name: s.test_partial_utterances(partial_utterance_length) for s in self.speakers}
