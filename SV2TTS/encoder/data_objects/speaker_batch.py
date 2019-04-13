import numpy as np
from typing import List
from encoder.data_objects.speaker import Speaker

class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])
