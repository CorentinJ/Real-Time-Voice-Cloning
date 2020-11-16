from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Visualize(QFrame):
    def __init__(self):
        super().__init__()
        # Visualizations
        vis_layout = QVBoxLayout()
        self.setLayout(vis_layout)
        ## Embed & spectrograms
        vis_layout.addStretch()

        gridspec_kw = {"width_ratios": [1, 4]}
        fig, self.current_ax = plt.subplots(
            1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", gridspec_kw=gridspec_kw
        )
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        fig, self.gen_ax = plt.subplots(
            1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", gridspec_kw=gridspec_kw
        )
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        for ax in self.current_ax.tolist() + self.gen_ax.tolist():
            ax.set_facecolor("#F0F0F0")
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)

    def draw_embed(self, embed, name, which):
        embed_ax, _ = self.current_ax if which == "current" else self.gen_ax
        embed_ax.figure.suptitle("" if embed is None else name)

        ## Embedding
        # Clear the plot
        if len(embed_ax.images) > 0:
            embed_ax.images[0].colorbar.remove()
        embed_ax.clear()

        # Draw the embed
        if embed is not None:
            plot_embedding_as_heatmap(embed, embed_ax)
            embed_ax.set_title("embedding")
        embed_ax.set_aspect("equal", "datalim")
        embed_ax.set_xticks([])
        embed_ax.set_yticks([])
        embed_ax.figure.canvas.draw()

    def draw_spec(self, spec, which):
        _, spec_ax = self.current_ax if which == "current" else self.gen_ax

        ## Spectrogram
        # Draw the spectrogram
        spec_ax.clear()
        if spec is not None:
            im = spec_ax.imshow(spec, aspect="auto", interpolation="none")
            # spec_ax.figure.colorbar(mappable=im, shrink=0.65, orientation="horizontal",
            # spec_ax=spec_ax)
            spec_ax.set_title("mel spectrogram")

        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.figure.canvas.draw()
        if which != "current":
            self.vocode_button.setDisabled(spec is None)
