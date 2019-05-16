from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
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
    min_umap_points = 5
    
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

    def draw_umap(self, embeds):
        self.umap_ax.clear()
        
        # Display a message if there aren't enough points
        if len(embeds) <= self.min_umap_points:
            self.umap_ax.text(.5, .5, "Add %d more points to\ngenerate the projections" % 
                              (self.min_umap_points - len(embeds)), horizontalalignment='center',
                              fontsize=15)
            self.umap_ax.set_xticks([])
            self.umap_ax.set_yticks([])
            self.umap_ax.figure.canvas.draw()
            return
    
        # Compute the projections
        speaker_names, indices = np.unique([e[2] for e in embeds], return_index=True)
        speaker_names = speaker_names[np.argsort(indices)]
        speaker_dict = {s: i for i, s in enumerate(speaker_names)}
        embed_data = np.array([e[0] for e in embeds])
        reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embed_data)))), metric="cosine")
        projections = reducer.fit_transform(embed_data)
    
        # Hide or show partials
        show_partials = self.show_partials_button.isChecked()
        if not show_partials:
            projections = [p for e, p in zip(embeds, projections) if e[1] != "."]
            embeds = [e for e in embeds if e[1] != "."]
        else:
            embeds = embeds
    
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
        
    @property        
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()
    
    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()
    
    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        def repopulate_box(box, items):
            box.blockSignals(True)
            box.clear()
            box.addItems(map(str, items))
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
            box.blockSignals(False)
    
        # Select a random dataset
        if level <= 0:
            datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
            datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
            repopulate_box(self.dataset_box, datasets)
    
        # Select a random speaker
        if level <= 1:
            speakers_root = Path(datasets_root, self.current_dataset_name())
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            repopulate_box(self.speaker_box, speaker_names)
    
        # Select a random utterance
        if level <= 2:
            utterances_root = Path(datasets_root, 
                                   self.current_dataset_name, 
                                   self.current_speaker_name)
            utterances = Path(utterances_root).glob("*/**")
            utterances = [fpath for fpath in utterances if fpath.suffix in "[.mp3|.flac|.wav|.m4a]"]
            repopulate_box(self.utterance_box, utterances)

    def browser_select_next(self):
        def select_next_box(box):
            index = (box.currentIndex() + 1) % len(box)
            box.setCurrentIndex(index)
            return index == 0
    
        if select_next_box(self.utterance_box):
            if select_next_box(self.speaker_box):
                select_next_box(self.dataset_box)

    def reset_interface(self):
        self.draw_embed_heatmap(None, "", "current")
        self.draw_embed_heatmap(None, "", "generated")
        self.draw_spec(None, "", "current")
        self.draw_spec(None, "", "generated")
        self.draw_umap([])

    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("SV2TTS toolbox")
        
        
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
        umap_canvas = FigureCanvas(Figure(figsize=(10, 10)))
        proj_layout.addWidget(umap_canvas)
        self.umap_ax = umap_canvas.figure.subplots()
        umap_canvas.figure.patch.set_facecolor("#F0F0F0")

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
        self.embed_axs = embed_canvas.figure.subplots(2, 1)
        embed_canvas.figure.patch.set_facecolor("#F0F0F0")
        
        spec_canvas = FigureCanvas(Figure(figsize=(5, 10)))
        vis_layout.addWidget(spec_canvas)
        self.spec_axs = spec_canvas.figure.subplots(2, 1)
        spec_canvas.figure.patch.set_facecolor("#F0F0F0")


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
        
        
        ## Finalize the display
        max_size = QDesktopWidget().availableGeometry(self).size()
        self.resize(max_size * 0.8)
        self.reset_interface()
        self.show()

    def start(self):
        self.app.exec_()

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
            self.browser_select_next()

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
