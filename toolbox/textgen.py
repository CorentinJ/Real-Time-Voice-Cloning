from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *

# from pathlib import Path
# from typing import List, Set
# from toolbox.utterance import Utterance


class TextGeneration(QFrame):
    default_text = (
        "Welcome to the toolbox! To begin, load an utterance from your datasets or record one "
        "yourself.\nOnce its embedding has been created, you can synthesize any text written here.\n"
        "With the current synthesizer model, punctuation and special characters will be ignored.\n"
        "The synthesizer expects to generate "
        "outputs that are somewhere between 5 and 12 seconds.\nTo mark breaks, write a new line. "
        "Each line will be treated separately.\nThen, they are joined together to make the final "
        "spectrogram. Use the vocoder to generate audio.\nThe vocoder generates almost in constant "
        "time, so it will be more time efficient for longer inputs like this one.\nOn the left you "
        "have the embedding projections. Load or record more utterances to see them.\nIf you have "
        "at least 2 or 3 utterances from a same speaker, a cluster should form.\nSynthesized "
        "utterances are of the same color as the speaker whose voice was used, but they're "
        "represented with a cross."
    )

    def __init__(self):
        super().__init__()

        gen_layout = QVBoxLayout()
        self.setLayout(gen_layout)
        ## Generation
        self.text_prompt = QPlainTextEdit(self.default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)

        self.generate_button = QPushButton("Synthesize and vocode")
        gen_layout.addWidget(self.generate_button)

        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only")
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)

        layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed:")
        self.random_seed_checkbox.setToolTip(
            "When checked, makes the synthesizer and vocoder deterministic."
        )
        layout_seed.addWidget(self.random_seed_checkbox, 0, 0)
        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(80)
        layout_seed.addWidget(self.seed_textbox, 0, 1)
        self.trim_silences_checkbox = QCheckBox("Enhance vocoder output")
        self.trim_silences_checkbox.setToolTip(
            "When checked, trims excess silence in vocoder output."
            " This feature requires `webrtcvad` to be installed."
        )
        layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        gen_layout.addLayout(layout_seed)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)

        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()

    def setup_events(self):
        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

    def update_seed_textbox(self):
        if self.random_seed_checkbox.isChecked():
            self.seed_textbox.setEnabled(True)
        else:
            self.seed_textbox.setEnabled(False)

    def populate_gen_options(self, seed, trim_silences):
        if seed is not None:
            self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        if not trim_silences:
            self.trim_silences_checkbox.setChecked(False)
            self.trim_silences_checkbox.setDisabled(True)

    def set_loading(self, value, maximum=1):
        self.loading_bar.setValue(value * 100)
        self.loading_bar.setMaximum(maximum * 100)
        self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()
