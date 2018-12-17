from torch.utils.data import DataLoader
from datasets.data_loader import SpeakerVerificationDataset
from ui.speaker_matrix_ui import SpeakerMatrixUI


if __name__ == '__main__':
    dataset = SpeakerVerificationDataset(['train-other-500'], 3, 4)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, 
                        collate_fn=SpeakerVerificationDataset.collate)
    for batch in loader:
        batch = batch[0]
        SpeakerMatrixUI(batch.speakers, batch.partial_utterances)
        