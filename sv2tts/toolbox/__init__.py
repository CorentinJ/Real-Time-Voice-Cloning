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
from pathlib import Path

recognized_datasets = [
    "Librispeech/dev-clean",
    "Librispeech/dev-other",
    "Librispeech/test-clean",
    "Librispeech/test-other",
    "Librispeech/train-clean-100",
    "Librispeech/train-clean-360",
    "Librispeech/train-other-500",
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
        self.init()
        self.ui.start()
        
    def setup_events(self):
        self.ui.browser_load_button.clicked.connect(self.load_from_browser)

    def load_from_browser(self):
        fpath = Path(self.datasets_root,
                     self.ui.current_dataset_name,
                     self.ui.current_speaker_name,
                     self.ui.current_utterance_name)
        wav = inference.load_preprocess_waveform(fpath)
        name = str(fpath.relative_to(self.datasets_root))
        self.load_utterance(name,
                            wav,
                            self.ui.current_speaker_name)
        self.ui.log("Loaded %s" % name)
        
    def load_utterance(self, name, wav, speaker_name):
        pass
    
    def init(self):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, False)
    
    def load_current_utterance(self, embed_it: bool):
        pass
        
    
if __name__ == '__main__':
    datasets_root = Path(r"C:\Datasets")
    Toolbox(datasets_root, "encoder/saved_models/pretrained.pt")

        
        