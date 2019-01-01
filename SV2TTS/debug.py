import audio
import numpy as np
from vlibs import fileio
from ui.speaker_matrix_ui import SpeakerMatrixUI
from datasets.data_loader import SpeakerVerificationDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # fpath = r"E:\Datasets\LibriSpeech\train-other-500\25\123319\25-123319-0006.flac"
    # wave = audio.load(fpath)
    # import matplotlib.pyplot as plt
    # plt.subplot(211)
    # plt.plot(wave)
    # 
    # wave2 = audio.trim_long_silences(wave)
    # plt.subplot(212)
    # plt.plot(wave2)
    # audio.play_wave(wave2)
    # plt.show()
    # 
    # 
    # quit()
    dataset = SpeakerVerificationDataset(['train-other-500'], 3, 4)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, 
                        collate_fn=SpeakerVerificationDataset.collate)
    for batch in loader:
        SpeakerMatrixUI(batch.speakers, batch.partial_utterances)
    