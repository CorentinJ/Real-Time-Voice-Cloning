from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
from vlibs import fileio
import sounddevice as sd
import numpy as np
import umap
import sys
import os
from encoder.params_data import sampling_rate
from encoder import audio, inference
from config import demo_datasets_root

colormap = np.array([
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
    [76, 255, 0],
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


class UMapDemoUI(QDialog):
    def __init__(self):
        self.embeds = []
        self.utterance = None
        self.is_record = None
        self.ax = None
        
        # Load and prime the models
        print("Loading the model in memory, please wait...")
        inference.load_model('saved_models/all.pt', 'cuda')
        
        # Create the ui
        self.setWindowTitle("Voice embedding visualizer")
        app = QApplication(sys.argv)
        super().__init__(None)
        
        # Display it and stay in mainloop until the window is closed
        self.setup_ui()
        self.show()
        app.exec_()
        
    def setup_ui(self):
        root_layout = QHBoxLayout()
        
        ## UMAP plot (left)
        canvas = FigureCanvas(Figure(figsize=(10, 20)))
        self.ax = canvas.figure.subplots()
        self.draw_umap()
        
        ## Menu layout (right)
        menu_layout = QVBoxLayout()
        
        ## Browser (top right)
        browser_grid = QGridLayout()
        self.dataset_box = QComboBox()
        self.speaker_box = QComboBox()
        self.utterance_box = QComboBox()
        random_dataset_button = QPushButton('Random')
        random_speaker_button = QPushButton('Random')
        random_utterance_button = QPushButton('Random')
        play_button = QPushButton('Play')
        stop_button = QPushButton('Stop')
        self.dataset_box.currentIndexChanged.connect(lambda: self.select_random(1))
        self.speaker_box.currentIndexChanged.connect(lambda: self.select_random(2))
        self.utterance_box.currentIndexChanged.connect(lambda: self.load_utterance())
        random_dataset_button.clicked.connect(lambda: self.select_random(0))
        random_speaker_button.clicked.connect(lambda: self.select_random(1))
        random_utterance_button.clicked.connect(lambda: self.select_random(2))
        play_button.clicked.connect(lambda: sd.play(self.utterance, sampling_rate))
        stop_button.clicked.connect(lambda: sd.stop())
        browser_grid.addWidget(QLabel('Dataset', ), 0, 0)
        browser_grid.addWidget(QLabel('Speaker'), 0, 1)
        browser_grid.addWidget(QLabel('Utterance'), 0, 2)
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
        embeds_grid = QGridLayout()
        embed_button = QPushButton('Embed utterance (direct)')
        embed_demo_button = QPushButton('Embed utterance (demo)')
        record_one_button = QPushButton('Record one')
        self.use_partials_button = QCheckBox('Use partials')
        self.show_partials_button = QCheckBox('Show partials')
        self.go_next_button = QCheckBox('Auto pick next')
        self.user_id_box = QSpinBox()
        self.user_id_box.setRange(0, 9)
        self.use_partials_button.setChecked(True)
        self.show_partials_button.setChecked(True)
        self.go_next_button.setChecked(True)
        embed_button.clicked.connect(lambda: self.embed_utterance(False))
        embed_demo_button.clicked.connect(lambda: self.embed_utterance(True))
        record_one_button.clicked.connect(self.record_one)
        embeds_grid.addWidget(self.use_partials_button, 0, 0)
        embeds_grid.addWidget(self.show_partials_button, 1, 0)
        embeds_grid.addWidget(self.go_next_button, 2, 0)
        embeds_grid.addWidget(embed_button, 3, 0)
        embeds_grid.addWidget(embed_demo_button, 3, 1)
        embeds_grid.addWidget(record_one_button, 4, 0)
        embeds_grid.addWidget(self.user_id_box, 4, 1)
        # TODO add overlap and n_frames
        menu_layout.addLayout(embeds_grid)
        menu_layout.addStretch()
        
        clear_button = QPushButton('Clear')
        def clear_button_action():
            self.embeds.clear()
            self.draw_umap()
        clear_button.clicked.connect(clear_button_action)
        menu_layout.addWidget(clear_button)

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

    def select_next(self):
        def select_next_box(box):
            index = (box.currentIndex() + 1) % len(box)
            box.setCurrentIndex(index)
            return index == 0
        
        if select_next_box(self.utterance_box):
            if select_next_box(self.speaker_box):
                select_next_box(self.dataset_box)
        self.load_utterance()

    def load_utterance(self, fpath=None):
        if fpath is None:
            fpath = fileio.join(demo_datasets_root,
                                self.dataset_box.currentText(),
                                self.speaker_box.currentText(),
                                self.utterance_box.currentText())
        self.utterance = inference.load_and_preprocess_wave(fpath)
        self.is_record = False

    def embed_utterance(self, demo, speaker_name=None, go_next=None):
        if speaker_name is None:
            speaker_name = '%s/%s' % (self.dataset_box.currentText(), 
                                      self.speaker_box.currentText())
        use_partials = self.use_partials_button.isChecked()
        go_next = go_next if go_next is not None else self.go_next_button.isChecked() 
        
        # Compute the embedding(s)
        embed, partial_embeds, wave_splits = inference.embed_utterance(
            self.utterance, use_partials, return_partials=True)
        if use_partials:
            self.embeds.append((embed, 'o', speaker_name))
            for partial_embed in partial_embeds:
                self.embeds.append((partial_embed, '.', speaker_name))
        else:
            self.embeds.append((embed, 's', speaker_name))
            
        self.draw_umap()
        
        if go_next:
            self.select_next()

    def record_one(self):
        self.utterance = audio.preprocess_wave(audio.rec_wave(4))
        speaker_name = 'user_' + self.user_id_box.text() 
        self.embed_utterance(False, speaker_name, False)
    
    def draw_umap(self):
        self.ax.clear()
        if len(self.embeds) < 5:
            self.ax.figure.canvas.draw()
            return
        
        # Hide or show partials
        show_partials = self.show_partials_button.isChecked()
        if not show_partials:
            embeds = [e for e in self.embeds if e[1] != '.']
        else:
            embeds = self.embeds

        # Compute the projections
        speaker_names, indices = np.unique([e[2] for e in embeds], return_index=True)
        speaker_names = speaker_names[np.argsort(indices)]
        speaker_dict = {s: i for i, s in enumerate(speaker_names)}
        embed_data = np.array([e[0] for e in embeds])
        reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embed_data)))), metric='cosine')
        projections = reducer.fit_transform(embed_data)

        # TODO: record, lines
        
        legend_done = set()
        for projection, (embed, mark, speaker_name) in zip(projections, embeds):
            color = [colormap[speaker_dict[speaker_name]]]
            legend = None
            if not speaker_name in legend_done:
                legend = speaker_name
                legend_done.add(speaker_name)
            self.ax.scatter(projection[0], projection[1], c=color, marker=mark, label=legend)
        self.ax.set_aspect('equal', 'datalim')
        self.ax.set_title('UMAP projection')
        self.ax.legend()

        # figure.tight_layout()
        self.ax.figure.canvas.draw()


if __name__ == '__main__':
    UMapDemoUI()
