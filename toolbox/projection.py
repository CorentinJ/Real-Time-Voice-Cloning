from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import List, Set
from toolbox.utterance import Utterance
import numpy as np


class Projection(QFrame):
    min_umap_points = 4

    def __init__(self):
        super().__init__()

        # Projections
        self.projections_layout = QVBoxLayout()
        self.setLayout(self.projections_layout)

        ## Projections
        # UMap
        fig, self.umap_ax = plt.subplots(figsize=(3, 3), facecolor="#F0F0F0")
        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.projections_layout.addWidget(FigureCanvas(fig))
        self.umap_hot = False
        self.clear_button = QPushButton("Clear")
        self.projections_layout.addWidget(self.clear_button)

    def draw_umap_projections(self, utterances: Set[Utterance]):
        self.umap_ax.clear()

        speakers = np.unique([u.speaker_name for u in utterances])
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        embeds = [u.embed for u in utterances]

        # Display a message if there aren't enough points
        if len(utterances) < self.min_umap_points:
            self.umap_ax.text(
                0.5,
                0.5,
                "Add %d more points to\ngenerate the projections"
                % (self.min_umap_points - len(utterances)),
                horizontalalignment="center",
                fontsize=15,
            )
            self.umap_ax.set_title("")

        # Compute the projections
        else:
            if not self.umap_hot:
                self.log(
                    "Drawing UMAP projections for the first time, this will take a few seconds."
                )
                self.umap_hot = True

            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            # reducer = TSNE()
            projections = reducer.fit_transform(embeds)

            speakers_done = set()
            for projection, utterance in zip(projections, utterances):
                color = colors[utterance.speaker_name]
                mark = "x" if "_gen_" in utterance.name else "o"
                label = (
                    None
                    if utterance.speaker_name in speakers_done
                    else utterance.speaker_name
                )
                speakers_done.add(utterance.speaker_name)
                self.umap_ax.scatter(
                    projection[0], projection[1], c=[color], marker=mark, label=label
                )
            # self.umap_ax.set_title("UMAP projections")
            self.umap_ax.legend(prop={"size": 10})

        # Draw the plot
        self.umap_ax.set_aspect("equal", "datalim")
        self.umap_ax.set_xticks([])
        self.umap_ax.set_yticks([])
        self.umap_ax.figure.canvas.draw()
