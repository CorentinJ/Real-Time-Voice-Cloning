from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
import sounddevice as sd
import numpy as np
import umap
import sys
import os
from encoder.params_data import sampling_rate
from encoder import audio, inference
from toolbox.ui import UI


recognized_datasets = [
    "Librispeech/dev-clean",
    "Librispeech/dev-other",
    "Librispeech/test-clean",
    "Librispeech/test-other",
    "Librispeech/train-other-500",
    "Librispeech/train-clean-100",
    "Librispeech/train-clean-360",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VCTK-Corpus/wav48",
]


class Toolbox:
    def __init__(self, datasets_root, encoder_fpath):
        self.datasets_root = datasets_root
        self.current_utterance = None
        
        
        # Initialize the events and the interface
        self.ui = UI()
        self.setup_events()
        self.ui.start()
        
    def setup_events(self):
        pass
    
    def load_current_utterance(self, embed_it: bool):
        pass
        
    
if __name__ == '__main__':
    Toolbox(r"E:\Datasets", "encoder/saved_models/pretrained.pt")

        
        