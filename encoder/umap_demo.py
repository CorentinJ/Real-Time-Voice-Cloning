from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
from PyQt4 import QtGui
from vlibs import fileio
import sounddevice as sd
import numpy as np
import umap
import sys
import os
from encoder.params_data import sampling_rate
from encoder.config import demo_datasets_root
from encoder import audio, inference

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
recognized_datasets = [
    'Librispeech/train-other-500',
    'Librispeech/train-clean-100',
    'Librispeech/train-clean-360',
    'LJSpeech-1.1',
    'VoxCeleb1/wav',
    'VoxCeleb2/dev/aac',
    'VCTK-Corpus/wav48',
]
# recognized_datasets = [fileio.join(d) for d in recognized_datasets]

class UMapDemoUI(QtGui.QDialog):
    def __init__(self):
        self.embeds = []
        self.utterance = None
        self.is_record = None
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
        
        ## UMAP plot (left)
        canvas = FigureCanvas(Figure(figsize=(10, 20)))
        self.ax = canvas.figure.subplots()
        self.draw_umap()
        
        ## Menu layout (right)
        menu_layout = QtGui.QVBoxLayout()
        
        ## Browser (top right)
        browser_grid = QtGui.QGridLayout()
        self.dataset_box = QtGui.QComboBox()
        self.speaker_box = QtGui.QComboBox()
        self.utterance_box = QtGui.QComboBox()
        random_dataset_button = QtGui.QPushButton('Random')
        random_speaker_button = QtGui.QPushButton('Random')
        random_utterance_button = QtGui.QPushButton('Random')
        play_button = QtGui.QPushButton('Play')
        stop_button = QtGui.QPushButton('Stop')
        self.dataset_box.currentIndexChanged.connect(lambda: self.select_random(1))
        self.speaker_box.currentIndexChanged.connect(lambda: self.select_random(2))
        self.dataset_box.currentIndexChanged.connect(lambda: self.load_utterance())
        random_dataset_button.clicked.connect(lambda: self.select_random(0))
        random_speaker_button.clicked.connect(lambda: self.select_random(1))
        random_utterance_button.clicked.connect(lambda: self.select_random(2))
        play_button.clicked.connect(lambda: sd.play(self.utterance, sampling_rate))
        stop_button.clicked.connect(lambda: sd.stop())
        browser_grid.addWidget(QtGui.QLabel('Dataset', ), 0, 0)
        browser_grid.addWidget(QtGui.QLabel('Speaker'), 0, 1)
        browser_grid.addWidget(QtGui.QLabel('Utterance'), 0, 2)
        browser_grid.addWidget(self.dataset_box, 1, 0)
        browser_grid.addWidget(self.speaker_box, 1, 1)
        browser_grid.addWidget(self.utterance_box, 1, 2)
        browser_grid.addWidget(random_dataset_button, 2, 0)
        browser_grid.addWidget(random_speaker_button, 2, 1)
        browser_grid.addWidget(random_utterance_button, 2, 2)
        media_buttons_layout = QHBoxLayout()
        media_buttons_layout.addWidget(play_button)
        media_buttons_layout.addWidget(stop_button)
        browser_grid.addLayout(media_buttons_layout, 3, 0, 1, -1)
        menu_layout.addLayout(browser_grid)
        menu_layout.addStretch()
        self.select_random(0)
        
        ## Audio visualization (middle right)
        # TODO if I feel like it
        
        ## Embeds (bottom right)
        embeds_grid = QtGui.QGridLayout()
        embed_button = QtGui.QPushButton('Embed utterance (direct)')
        embed_demo_button = QtGui.QPushButton('Embed utterance (demo)')
        embed_speaker_button = QtGui.QPushButton('Embed all from speaker')
        self.partials_buttons = QtGui.QRadioButton('Use partials')
        self.partials_buttons.toggle()
        embed_button.clicked.connect(lambda: self.embed_utterance(False))
        embed_demo_button.clicked.connect(lambda: self.embed_utterance(True))
        embeds_grid.addWidget(self.partials_buttons, 0, 0)
        embeds_grid.addWidget(embed_button, 1, 0)
        embeds_grid.addWidget(embed_demo_button, 1, 1)
        embeds_grid.addWidget(embed_speaker_button, 2, 0, 1, -1)
        # TODO add overlap and n_frames
        menu_layout.addLayout(embeds_grid)
        menu_layout.addStretch()
        
        
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
        
        max_size = QDesktopWidget().availableGeometry(self).size()
        self.resize(max_size * 0.8)
        
    def select_random(self, level):
        def repopulate_box(box, items):
            box.blockSignals(True)
            box.clear()
            box.addItems(items)
            box.setCurrentIndex(np.random.randint(len(items)))
            box.blockSignals(False)
            
        # Select a random dataset
        if level <= 0:
            available_datasets = [d for d in recognized_datasets if 
                                  fileio.exists(fileio.join(demo_datasets_root, d))]
            repopulate_box(self.dataset_box, available_datasets)
            
        # Select a random speaker
        if level <= 1:
            speakers_root = fileio.join(demo_datasets_root, self.dataset_box.currentText())
            available_speakers = [d for d in fileio.listdir(speakers_root) if 
                                  os.path.isdir(fileio.join(speakers_root, d))]
            repopulate_box(self.speaker_box, available_speakers)
            
        # Select a random utterance
        if level <= 2:
            utterances_root = fileio.join(demo_datasets_root, 
                                          self.dataset_box.currentText(),
                                          self.speaker_box.currentText())
            available_utterances = fileio.get_files(utterances_root, 
                                                    r'(.mp3|.flac|.wav|.m4a)',
                                                    recursive=True,
                                                    full_path=False) 
            repopulate_box(self.utterance_box, available_utterances)
            
        # Reload the new utterance
        self.load_utterance()

    def load_utterance(self, fpath=None):
        if fpath is None:
            fpath = fileio.join(demo_datasets_root,
                                self.dataset_box.currentText(),
                                self.speaker_box.currentText(),
                                self.utterance_box.currentText())
        self.utterance = inference.load_and_preprocess_wave(fpath)
        self.is_record = False

    def embed_utterance(self, demo):
        speaker = 'user' if self.is_record else self.speaker_box.currentText()
        use_partials = self.partials_buttons.isChecked()
        
        embeds = inference.embed_utterance(self.utterance, use_partials)
        if not use_partials:
            self.embeds.append((embeds, 'o', speaker))
        else:
            for embed in embeds:
                self.embeds.append((embed, '.', speaker))
                
        self.draw_umap()

    def draw_umap(self):
        self.ax.clear()
        
        speakers = np.unique([e[2] for e in self.embeds])
        speakers_dict = {s: i for i, s in enumerate(speakers)}
        if len(self.embeds) <= 5:
            self.ax.figure.canvas.draw()
            return
        all_embeds = np.array([e[0] for e in self.embeds])
        
        # utterances_per_speaker = len(next(self.embeds.values().__iter__()))
        # ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        # colors = [colormap[i] for i in ground_truth]
        # labels = []
        # for speaker_name in self.embeds.keys():
        #     labels.extend([speaker_name] * utterances_per_speaker)

        # TODO: save transformer, plot the partials, consistent dict order, go next toggle
        
        reducer = umap.UMAP()
        transformer = reducer.fit(all_embeds)
        for embed, mark, speaker in self.embeds:
            color = [colormap[speakers_dict[speaker]]]
            projected = transformer.transform([embed])
            self.ax.scatter(projected[0, 0], projected[0, 1], c=color, marker=mark, label=speaker)
        self.ax.set_aspect('equal', 'datalim')
        self.ax.set_title('UMAP projection')
        self.ax.legend()
    
        # figure.tight_layout()
        self.ax.figure.canvas.draw()


if __name__ == '__main__':
    UMapDemoUI()
