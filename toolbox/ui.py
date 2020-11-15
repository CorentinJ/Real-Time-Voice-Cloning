import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *
from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance
from pathlib import Path
from typing import List, Set
import sounddevice as sd
import soundfile as sf
import numpy as np
from toolbox.browser import Browser
from toolbox.textgen import TextGeneration
from toolbox.viz import Visualize
from toolbox.projection import Projection

# from sklearn.manifold import TSNE         # You can try with TSNE if you like, I prefer UMAP
from time import sleep
import umap
import sys
from warnings import filterwarnings, warn

filterwarnings("ignore")


colormap = (
    np.array(
        [
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
        ],
        dtype=np.float,
    )
    / 255
)


class UI(QFrame):
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20

    def __init__(self, toolbox):

        self.toolbox = toolbox

        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("SV2TTS toolbox")

        ## Main layouts
        root_layout = QGridLayout()
        self.setLayout(root_layout)

        self.browser = Browser()
        root_layout.addWidget(self.browser, 0, 0)

        self.text = TextGeneration()
        root_layout.addWidget(self.text, 1, 0)

        self.viz = Visualize()
        root_layout.addWidget(self.viz, 0, 1)

        self.projection = Projection()
        root_layout.addWidget(self.projection, 1, 1)

        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.5
        self.resize(max_size)

        ## Finalize the display
        self.reset_interface()
        self.show()

    def setup_events(self):
        # Audio
        self.ui.setup_audio_devices(Synthesizer.sample_rate)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def reset_interface(self):
        self.viz.draw_embed(None, None, "current")
        self.viz.draw_embed(None, None, "generated")
        # self.viz.draw_spec(None, "current")
        # self.viz.draw_spec(None, "generated")
        self.projection.draw_umap_projections(set())
        # self.set_loading(0)
        self.browser.play_button.setDisabled(True)
        self.text.generate_button.setDisabled(True)
        self.text.synthesize_button.setDisabled(True)
        self.text.vocode_button.setDisabled(True)
        self.browser.replay_wav_button.setDisabled(True)
        self.browser.export_wav_button.setDisabled(True)
        # [self.log("") for _ in range(self.max_log_lines)]

    def reset_ui(
        self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir, seed
    ):
        self.browser.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(
            encoder_models_dir, synthesizer_models_dir, vocoder_models_dir
        )
        self.ui.populate_gen_options(seed, self.trim_silences)

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = "\n".join(self.logs)

        self.log_window.setText(log_text)
        self.app.processEvents()

    def start(self):
        self.app.exec_()
