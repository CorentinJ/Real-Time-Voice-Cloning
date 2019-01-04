from data_objects.speaker import Speaker
from typing import List
import numpy as np
from params import mel_n_channels

class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.partial_utterances = {s: s.random_partial_utterances(utterances_per_speaker, n_frames) 
                                   for s in speakers}
        
        # Array of shape (n_speakers * n_utterances) * n_frames * mel_n, e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.zeros((len(speakers) * utterances_per_speaker, n_frames, mel_n_channels),
                             dtype=np.float32)
        i = 0
        for speaker in speakers:
            for _, frames, _ in self.partial_utterances[speaker]:
                self.data[i] = frames
                i += 1
        