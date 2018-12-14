from torch.utils.data import Dataset, DataLoader
from datasets.speaker import Speaker
from datasets.speaker_batch import SpeakerBatch
from vlibs import fileio
from config import *
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
        batch = SpeakerBatch(speakers, self.utterances_per_speaker, 160)
        return batch.data

    # def collate(batches):
    #     # We don't want SpeakerBatches to be aggregated in a single numpy array because they do 
    #     # not have the same shape. By override the collate function with indentity, they are 
    #     # aggregated in a list instead.
    #     return batches

if __name__ == '__main__':
    from audio import plot_mel_filterbank
    
    dataset = SpeakerVerificationDataset(['train-other-500'], 3, 4)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)
    
    for batches in loader:
        speaker_batch = batches[0]
        plot_mel_filterbank(speaker_batch[0].numpy(), 16000)
    