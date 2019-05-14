from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
from encoder import audio as encoder_audio
from encoder import inference as encoder
from pathlib import Path
import sounddevice as sd
import numpy as np
import umap
import sys


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

def clean_figure(figure, axs, square=True):
    figure.patch.set_facecolor("#F0F0F0")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        if square:
            ax.set_aspect("equal", "datalim")
    

class UI(QDialog):
    def __init__(self):
        ## Initialize the application
        # self.setWindowTitle("SV2TTS toolbox")
        app = QApplication(sys.argv)
        super().__init__(None)
        
        
        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)
        
        # Projections
        proj_layout = QVBoxLayout()
        root_layout.addLayout(proj_layout, 0, 0, 2, 1)
        
        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 1)
        
        # Visualizations
        vis_layout = QHBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 1, 2)
        
        # Generation
        gen_layout = QGridLayout()
        root_layout.addLayout(gen_layout, 0, 2)


        ## Projections
        umap_canvas = FigureCanvas(Figure(figsize=(10, 20)))
        proj_layout.addWidget(umap_canvas)
        self.umap_ax = umap_canvas.figure.subplots()
        clean_figure(umap_canvas.figure, [self.umap_ax])
        # self.draw_umap()


        ## Browser
        # Dataset & speaker selection
        self.dataset_box = QComboBox()
        browser_layout.addWidget(self.dataset_box, 0, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(self.speaker_box, 0, 1)
        
        # Random dataset & speaker buttons
        self.random_dataset_button = QPushButton("Random dataset")
        browser_layout.addWidget(self.random_dataset_button, 1, 0)
        self.random_speaker_button = QPushButton("Random speaker")
        browser_layout.addWidget(self.random_speaker_button, 1, 1)
        
        # Random & next utterance buttons
        self.random_utterance_button = QPushButton("Random utterance")
        browser_layout.addWidget(self.random_utterance_button, 2, 0)
        self.next_utterance_button = QPushButton("Next utterance")
        browser_layout.addWidget(self.next_utterance_button, 2, 1)
        
        # History selection & take generated button
        self.history_box = QComboBox()
        browser_layout.addWidget(self.history_box, 3, 0)
        self.take_generated_button = QPushButton("Take generated")
        browser_layout.addWidget(self.take_generated_button, 3, 1)
        
        # Utterance box
        self.utterance_box = QComboBox()
        browser_layout.addWidget(self.utterance_box, 4, 0, 1, 2)

        # Random & next utterance buttons
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, 5, 0)
        self.record_button = QPushButton("Next utterance")
        browser_layout.addWidget(self.record_button, 5, 1)
        
        # self.dataset_box.currentIndexChanged.connect(lambda: self.select_random(1))
        # self.speaker_box.currentIndexChanged.connect(lambda: self.select_random(2))
        # self.utterance_box.currentIndexChanged.connect(lambda: self.load_utterance())
        # random_dataset_button.clicked.connect(lambda: self.select_random(0))
        # random_speaker_button.clicked.connect(lambda: self.select_random(1))
        # random_utterance_button.clicked.connect(lambda: self.select_random(2))
        # play_button.clicked.connect(lambda: sd.play(self.utterance, sampling_rate))
        # stop_button.clicked.connect(lambda: sd.stop())
        # self.select_random(0)


        ## Embed & spectrograms
        embed_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        vis_layout.addWidget(embed_canvas)
        self.embed_fig = embed_canvas.figure
        self.embed_axs = self.embed_fig.subplots(2, 1)
        clean_figure(self.embed_fig, self.embed_axs)
        
        spec_canvas = FigureCanvas(Figure(figsize=(5, 10)))
        vis_layout.addWidget(spec_canvas)
        self.spec_fig = spec_canvas.figure
        self.spec_axs = self.spec_fig.subplots(2, 1)
        clean_figure(self.spec_fig, self.spec_axs, False)


        # ## Embeds (bottom right)
        # embeds_grid = QGridLayout()
        # embed_button = QPushButton("Embed utterance (direct)")
        # embed_demo_button = QPushButton("Embed utterance (demo)")
        # self.record_one_button = QPushButton("Record one")
        # self.use_partials_button = QCheckBox("Use partials")
        # self.show_partials_button = QCheckBox("Show partials")
        # self.go_next_button = QCheckBox("Auto pick next")
        # self.user_id_box = QSpinBox()
        # self.user_id_box.setRange(0, 9)
        # self.use_partials_button.setChecked(True)
        # self.show_partials_button.setChecked(True)
        # self.go_next_button.setChecked(True)
        # embed_button.clicked.connect(lambda: self.embed_utterance(False))
        # embed_demo_button.clicked.connect(lambda: self.embed_utterance(True))
        # self.record_one_button.clicked.connect(self.record_one)
        # embeds_grid.addWidget(self.use_partials_button, 0, 0)
        # embeds_grid.addWidget(self.show_partials_button, 1, 0)
        # embeds_grid.addWidget(self.go_next_button, 2, 0)
        # embeds_grid.addWidget(embed_button, 3, 0)
        # embeds_grid.addWidget(embed_demo_button, 3, 1)
        # embeds_grid.addWidget(self.record_one_button, 4, 0)
        # embeds_grid.addWidget(self.user_id_box, 4, 1)
        # # TODO add overlap and n_frames
        # menu_layout.addLayout(embeds_grid)
        # menu_layout.addStretch()
        # 
        # clear_button = QPushButton("Clear")
        # 
        # def clear_button_action():
        #     self.embeds.clear()
        #     self.draw_umap()
        # 
        # clear_button.clicked.connect(clear_button_action)
        # menu_layout.addWidget(clear_button)
        # 
        # root_layout.addLayout(menu_layout)
        

        ## Run the application
        max_size = QDesktopWidget().availableGeometry(self).size()
        self.resize(max_size * 0.8)
        self.show()
        app.exec_()
        
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
                                                    r"(.mp3|.flac|.wav|.m4a)",
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
        self.utterance = inference.load_preprocess_waveform(fpath)
        self.is_record = False

    def embed_utterance(self, demo, speaker_name=None, go_next=None):
        if speaker_name is None:
            speaker_name = "%s/%s" % (self.dataset_box.currentText(), 
                                      self.speaker_box.currentText())
        use_partials = self.use_partials_button.isChecked()
        go_next = go_next if go_next is not None else self.go_next_button.isChecked() 
        
        # Compute the embedding(s)
        embed, partial_embeds, wave_splits = inference.embed_utterance(
            self.utterance, use_partials, return_partials=True)
        if use_partials:
            self.embeds.append((embed, "o", speaker_name))
            for partial_embed in partial_embeds:
                self.embeds.append((partial_embed, ".", speaker_name))
        else:
            self.embeds.append((embed, "s", speaker_name))
            
        # Draw the embed and the UMAP projection
        self.draw_embed()
        self.draw_umap()
        
        if go_next:
            self.select_next()

    def record_one(self):
        self.record_one_button.setText("Recording...")
        self.record_one_button.setDisabled(True)
        self.utterance = audio.preprocess_wave(audio.rec_wave(4))
        self.record_one_button.setText("Done!")
        speaker_name = "user_" + self.user_id_box.text() 
        self.embed_utterance(False, speaker_name, False)
        self.record_one_button.setText("Record one")
        self.record_one_button.setDisabled(False)
    
    def draw_embed(self):
        if len(self.embeds) != 0:
            embed, _, speaker_name = self.embeds[-1]
            if len(self.embed_ax.images) > 0:
                self.embed_ax.images[0].colorbar.remove()
            self.embed_ax.clear()
            inference.plot_embedding_as_heatmap(embed, self.embed_ax, 
                                                "Last embedding for %s" % speaker_name)
        self.embed_ax.figure.canvas.draw()
    
    def draw_umap(self):
        self.umap_ax.clear()
        if len(self.embeds) < 5:
            self.umap_ax.figure.canvas.draw()
            return

        # Compute the projections
        speaker_names, indices = np.unique([e[2] for e in self.embeds], return_index=True)
        speaker_names = speaker_names[np.argsort(indices)]
        speaker_dict = {s: i for i, s in enumerate(speaker_names)}
        embed_data = np.array([e[0] for e in self.embeds])
        reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embed_data)))), metric="cosine")
        projections = reducer.fit_transform(embed_data)

        # Hide or show partials
        show_partials = self.show_partials_button.isChecked()
        if not show_partials:
            projections = [p for e, p in zip(self.embeds, projections) if e[1] != "."]
            embeds = [e for e in self.embeds if e[1] != "."]
        else:
            embeds = self.embeds

        # TODO: lines
        
        legend_done = set()
        for projection, (embed, mark, speaker_name) in zip(projections, embeds):
            color = [colormap[speaker_dict[speaker_name]]]
            legend = None
            if not speaker_name in legend_done:
                legend = speaker_name
                legend_done.add(speaker_name)
            self.umap_ax.scatter(projection[0], projection[1], c=color, marker=mark, label=legend)
        self.umap_ax.set_title("UMAP projection")
        self.umap_ax.legend()

        # figure.tight_layout()
        self.umap_ax.figure.canvas.draw()


if __name__ == "__main__":
    UI()
