from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtCore import QSize, Qt
from PyQt4.QtGui import *
from itertools import chain
from encoder import audio as encoder_audio
from encoder import inference as encoder
from pathlib import Path
from typing import List
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

   
class UI(QDialog):
    min_umap_points = 4
    max_log_lines = 5
    
    def draw_embed_heatmap(self, embed, speaker_name, which):
        ax = self.embed_axs[0 if which == "current" else 1]
        
        # Clear the plot
        if len(ax.images) > 0:
            ax.images[0].colorbar.remove()
        ax.clear()
        
        # Draw the embed
        if embed is not None:
            encoder.plot_embedding_as_heatmap(embed, ax, speaker_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "datalim")
        ax.figure.canvas.draw()
            
    def draw_spec(self, spec, utterance_name, which):
        ax = self.spec_axs[0 if which == "current" else 1]
        
        # Clear the plot
        ax.clear()
        
        # Draw the spectrogram
        if spec is not None:
            # TODO
            pass
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.canvas.draw()

    def draw_umap(self, embeds: dict):
        self.umap_ax.clear()

        if len(embeds) > len(colormap):
            print("Warning: maximum number of speakers/colors reached", file=sys.stderr)
        
        # Gather the embeddings to be plotted
        to_plot = []        # List of tuples (color, mark)
        embed_data = []     # List of embeds
        for speaker_dict, color in zip(embeds.values(), colormap):
            to_plot.extend([(color, "o")] * len(speaker_dict))
            embed_data.extend(embed for embed, _, _ in speaker_dict.values())
        # # Hide or show partials
        # show_partials = self.show_partials_button.isChecked()
        # if not show_partials:
        #     projections = [p for e, p in zip(embeds, projections) if e[1] != "."]
        #     embeds = [e for e in embeds if e[1] != "."]
        # else:
        #     embeds = embeds
        # 
        # TODO: lines

        # Display a message if there aren't enough points
        if len(embed_data) < self.min_umap_points:
            self.umap_ax.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                              (self.min_umap_points - len(embed_data)), 
                              horizontalalignment='center', fontsize=15)
            self.umap_ax.set_title("")
            
        # Compute the projections
        else:
            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embed_data)))), metric="cosine")
            projections = reducer.fit_transform(embed_data)
    
            for projection, (color, mark) in zip(projections, to_plot):
                self.umap_ax.scatter(projection[0], projection[1], c=[color], marker=mark)
            self.umap_ax.set_title("UMAP projections")
        
        # Draw the plot
        self.umap_ax.set_xticks([])
        self.umap_ax.set_yticks([])
        self.umap_ax.figure.canvas.draw()
        
    @property        
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()
    
    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()
    
    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join 
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.blockSignals(False)
    
    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        # Select a random dataset
        if level <= 0:
            datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
            datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
            if len(datasets) == 0:
                raise Exception("Could not find any of the datasets %s under %s" %
                                (recognized_datasets, datasets_root))
            self.repopulate_box(self.dataset_box, datasets, random)
    
        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)
    
        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name, 
                self.current_speaker_name
            )
            utterances = []
            for extension in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)

    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())
    
    @property
    def current_synthesizer_model_dir(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())
    
    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    def populate_models(self, encoder_models_dir: Path, synthesizer_models_dir: Path, 
                        vocoder_models_dir: Path):
        # Encoder
        encoder_fpaths = list(encoder_models_dir.glob("*.pt"))
        self.repopulate_box(self.encoder_box, [(f.stem, f) for f in encoder_fpaths])
        
        # Synthesizer
        synthesizer_model_dirs = list(synthesizer_models_dir.glob("*"))
        synthesizer_items = [(f.name.replace("logs-", ""), f) for f in synthesizer_model_dirs]
        self.repopulate_box(self.synthesizer_box, synthesizer_items)

        # Vocoder
        vocoder_fpaths = list(vocoder_models_dir.glob("**/*.pt"))
        vocoder_items = [(f.stem, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)
        
    def log(self, line):
        self.logs.append(line)
        if len(self.logs) > self.max_log_lines:
            del self.logs[0]
        log_text = '\n'.join(self.logs)
        self.log_window.setText(log_text)
        self.app.processEvents()

    def set_loading(self, value, maximum):
        self.loading_bar.setValue(value)
        self.loading_bar.setMaximum(maximum)

    def reset_interface(self):
        self.draw_embed_heatmap(None, "", "current")
        self.draw_embed_heatmap(None, "", "generated")
        self.draw_spec(None, "", "current")
        self.draw_spec(None, "", "generated")
        self.draw_umap({})

    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("SV2TTS toolbox")
        
        
        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)
        
        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 1)
        
        # Visualizations
        vis_layout = QHBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 2, 3)
        
        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2)


        ## Projections
        # Legend
        self.legend_layout = QVBoxLayout()
        root_layout.addLayout(self.legend_layout, 0, 0)
        
        # UMap
        umap_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.umap_ax = umap_canvas.figure.subplots()
        umap_canvas.figure.patch.set_facecolor("#F0F0F0")
        root_layout.addWidget(umap_canvas, 1, 0)


        ## Browser
        i = 0
        
        # Dataset, speaker and utterance selection
        self.dataset_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Dataset</b>"), i, 0)
        browser_layout.addWidget(self.dataset_box, i + 1, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Speaker</b>"), i, 1)
        browser_layout.addWidget(self.speaker_box, i + 1, 1)
        self.utterance_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Utterance</b>"), i, 2)
        browser_layout.addWidget(self.utterance_box, i + 1, 2)
        self.browser_load_button = QPushButton("Load")
        browser_layout.addWidget(self.browser_load_button, i + 1, 3)
        i += 2
        
        # Random buttons
        self.random_dataset_button = QPushButton("Random")
        browser_layout.addWidget(self.random_dataset_button, i, 0)
        self.random_speaker_button = QPushButton("Random")
        browser_layout.addWidget(self.random_speaker_button, i, 1)
        self.random_utterance_button = QPushButton("Random")
        browser_layout.addWidget(self.random_utterance_button, i, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        browser_layout.addWidget(self.auto_next_checkbox, i, 3)
        i += 1
        
        # Utterance box
        browser_layout.addWidget(QLabel("<b>Recent utterances</b>"), i, 0)
        self.current_load_button = QPushButton("Reload")
        browser_layout.addWidget(self.current_load_button, i + 1, 3)
        i += 1
        
        # Random & next utterance buttons
        self.current_utterance_button = QComboBox()
        browser_layout.addWidget(self.current_utterance_button, i, 0, 1, 3)
        i += 1
        
        # Random & next utterance buttons
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, i, 0)
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, i, 1)
        self.take_generated_button = QPushButton("Take generated")
        browser_layout.addWidget(self.take_generated_button, i, 2)
        i += 2

        # Model selection
        self.encoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)
        i += 2


        ## Embed & spectrograms
        embed_canvas = FigureCanvas(Figure(figsize=(2, 2)))
        vis_layout.addWidget(embed_canvas)
        self.embed_axs = embed_canvas.figure.subplots(2, 1)
        # embed_canvas.figure.patch.set_facecolor("#F0F0F0")
        vis_layout.addStretch()
        
        spec_canvas = FigureCanvas(Figure(figsize=(2, 2)))
        vis_layout.addWidget(spec_canvas)
        self.spec_axs = spec_canvas.figure.subplots(2, 1)
        # spec_canvas.figure.patch.set_facecolor("#F0F0F0")


        ## Generation
        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)
        
        self.generate_button = QPushButton("Generate")
        gen_layout.addWidget(self.generate_button)
        
        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()
        
        

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
        
        
        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.8
        self.resize(max_size)
        
        ## Finalize the display
        self.reset_interface()
        self.show()

    def start(self):
        self.app.exec_()

    def record_one(self):
        self.record_one_button.setText("Recording...")
        self.record_one_button.setDisabled(True)
        self.utterance = audio.preprocess_wave(audio.rec_wave(4))
        self.record_one_button.setText("Done!")
        speaker_name = "user_" + self.user_id_box.text() 
        self.embed_utterance(False, speaker_name, False)
        self.record_one_button.setText("Record one")
        self.record_one_button.setDisabled(False)
    

if __name__ == "__main__":
    UI()
