from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PyQt4.QtGui import *
from PyQt4 import QtGui
import numpy as np
import librosa
import audio
import sys
from params_data import sampling_rate, mel_window_step
from preprocess import preprocess_wave
import umap
from collections import OrderedDict

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255 



class UMapDemoUI(QtGui.QDialog):
    def __init__(self, loader, get_embeds):
        self.loader = loader
        self.get_embeds = get_embeds
        self.embeds = OrderedDict()
        self.ax = None
        
        # Create the ui
        app = QtGui.QApplication(sys.argv)
        super().__init__(None)
        
        # Display it and stay in mainloop until the window is closed
        self.setup_ui()
        self.show()
        app.exec_()
    
    def setup_ui(self):
        root_layout = QtGui.QHBoxLayout()
        
        # UMAP plot
        canvas = FigureCanvas(Figure(figsize=(10, 20)))
        self.ax = canvas.figure.subplots()

        # # Load the partial utterance's frames and waveform
        # utterance, frames, frames_range = partial_utterance
        # wave_fpath = utterance.wave_fpath
        # wave = audio.load(wave_fpath)
        # wave = preprocess_wave(wave)
        # 
        # wave_range = np.array(frames_range) * sampling_rate * (mel_window_step / 1000)
        # wave = wave[int(wave_range[0]):int(wave_range[1])]

        menu_layout = QtGui.QVBoxLayout()
        add_speaker_button = QtGui.QPushButton('Add a random speaker')
        def add_speaker_button_action():
            self.add_speaker()
            self.draw_umap()
        add_speaker_button.clicked.connect(add_speaker_button_action)
        menu_layout.addWidget(add_speaker_button)
        
        record_one_button = QtGui.QPushButton('Record one')
        def record_one_button_action():
            waves = [audio.rec_wave(2) for _ in range(5)]
            # waves = map(preprocess_wave, waves)
            waves = list(map(audio.wave_to_mel_filterbank, waves))
            data = np.array(waves)
            embeds = self.get_embeds(None, data=data)
            self.embeds['user'] = embeds
            self.draw_umap()
        record_one_button.clicked.connect(record_one_button_action)
        menu_layout.addWidget(record_one_button)
        
        clear_button = QtGui.QPushButton('Clear')
        def clear_button_action():
            self.embeds.clear()
            self.draw_umap()
        clear_button.clicked.connect(clear_button_action)
        menu_layout.addWidget(clear_button)

        # label = QLabel(speaker.name)
        root_layout.addWidget(canvas)
        root_layout.addLayout(menu_layout)
        self.setLayout(root_layout)
        
        for i in range(3):
            self.add_speaker()
        self.draw_umap()

    def add_speaker(self):
        speaker_batch = next(self.loader)
        embeds = self.get_embeds(speaker_batch)
        name = speaker_batch.speakers[0].name
        self.embeds[name] = embeds

    def draw_umap(self):
        self.ax.clear()
        
        n_speakers = len(self.embeds)
        if n_speakers == 0:
            self.ax.figure.canvas.draw()
            return
        all_embeds = np.concatenate(tuple(self.embeds.values()))
        
        # utterances_per_speaker = len(next(self.embeds.values().__iter__()))
        # ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        # colors = [colormap[i] for i in ground_truth]
        # labels = []
        # for speaker_name in self.embeds.keys():
        #     labels.extend([speaker_name] * utterances_per_speaker)

        reducer = umap.UMAP()
        transformer = reducer.fit(all_embeds)
        for i, (speaker_name, embeds) in enumerate(self.embeds.items()):
            projected = transformer.transform(embeds)
            self.ax.scatter(projected[:, 0], projected[:, 1], c=[colormap[i]], label=speaker_name)
        self.ax.set_aspect('equal', 'datalim')
        self.ax.set_title('UMAP projection')
        self.ax.legend()
    
        # figure.tight_layout()
        self.ax.figure.canvas.draw()
