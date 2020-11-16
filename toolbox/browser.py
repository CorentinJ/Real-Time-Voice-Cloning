from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import *
from pathlib import Path
from typing import List, Set
from toolbox.utterance import Utterance


class Browser(QFrame):
    def __init__(self):
        super().__init__()

        self.init_ui()
        # self.setup_events()

    def init_ui(self):
        # Browser
        browser_layout = QGridLayout()
        self.setLayout(browser_layout)

        # Dataset, speaker and utterance selection
        self.dataset_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Dataset</b>"), 0, 0)
        browser_layout.addWidget(self.dataset_box, 0 + 1, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Speaker</b>"), 0, 1)
        browser_layout.addWidget(self.speaker_box, 0 + 1, 1)
        self.utterance_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Utterance</b>"), 0, 2)
        browser_layout.addWidget(self.utterance_box, 0 + 1, 2)
        self.browser_load_button = QPushButton("Load")
        browser_layout.addWidget(self.browser_load_button, 0 + 1, 3)

        # Random buttons
        self.random_dataset_button = QPushButton("Random")
        browser_layout.addWidget(self.random_dataset_button, 2, 0)
        self.random_speaker_button = QPushButton("Random")
        browser_layout.addWidget(self.random_speaker_button, 2, 1)
        self.random_utterance_button = QPushButton("Random")
        browser_layout.addWidget(self.random_utterance_button, 2, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        browser_layout.addWidget(self.auto_next_checkbox, 2, 3)

        # Horizontal line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Raised)
        browser_layout.addWidget(self.separator, 3, 0, 1, -1)

        # Utterance box
        browser_layout.addWidget(QLabel("<b>Use embedding from:</b>"), 4, 0)
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, 4, 1, 1, 3)

        # Random & next utterance buttons
        self.browser_browse_button = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button, 5, 0)
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, 5, 1)
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, 5, 2)
        self.stop_button = QPushButton("Stop")
        browser_layout.addWidget(self.stop_button, 5, 3)
        i = 6

        # Horizontal line
        sepa = QFrame()
        sepa.setFrameShape(QFrame.HLine)
        sepa.setFrameShadow(QFrame.Raised)
        browser_layout.addWidget(sepa, i + 1, 0, 1, -1)
        i += 2

        # Model and audio output selection
        self.encoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)

        self.audio_out_devices_cb = QComboBox()
        browser_layout.addWidget(QLabel("<b>Audio Output</b>"), i, 3)
        browser_layout.addWidget(self.audio_out_devices_cb, i + 1, 3)
        i += 2

        # Horizontal line
        sepa = QFrame()
        sepa.setFrameShape(QFrame.HLine)
        sepa.setFrameShadow(QFrame.Raised)
        browser_layout.addWidget(sepa, i + 1, 0, 1, -1)
        i += 2

        # Replay & Save Audio
        browser_layout.addWidget(QLabel("<b>Toolbox Output:</b>"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        self.waves_cb.setToolTip(
            "Select one of the last generated waves in this section for replaying or exporting"
        )
        browser_layout.addWidget(self.waves_cb, i, 1)
        self.replay_wav_button = QPushButton("Replay")
        self.replay_wav_button.setToolTip("Replay last generated vocoder")
        browser_layout.addWidget(self.replay_wav_button, i, 2)
        self.export_wav_button = QPushButton("Export")
        self.export_wav_button.setToolTip(
            "Save last generated vocoder audio in filesystem as a wav file"
        )
        browser_layout.addWidget(self.export_wav_button, i, 3)

    def populate_browser(
        self, datasets_root: Path, recognized_datasets: List, level: int, random=True
    ):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [
                    d.relative_to(datasets_root) for d in datasets if d.exists()
                ]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Warning: you d" + (
                    "id not pass a root directory for datasets as argument"
                    if datasets_root is None
                    else "o not have any of the recognized datasets"
                    " in %s" % datasets_root
                )
                self.log(msg)
                msg += (
                    ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You "
                    "can still use the toolbox by recording samples yourself."
                    % ("\n\t".join(recognized_datasets))
                )
                print(msg, file=sys.stderr)

                self.random_utterance_button.setDisabled(True)
                self.random_speaker_button.setDisabled(True)
                self.random_dataset_button.setDisabled(True)
                self.utterance_box.setDisabled(True)
                self.speaker_box.setDisabled(True)
                self.dataset_box.setDisabled(True)
                self.browser_load_button.setDisabled(True)
                self.auto_next_checkbox.setDisabled(True)
                return
            self.repopulate_box(self.dataset_box, datasets, random)

        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)

        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name, self.current_speaker_name
            )
            utterances = []
            for extension in ["mp3", "flac", "wav", "m4a"]:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)

    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    def draw_utterance(self, utterance: Utterance, which):
        self.draw_spec(utterance.spec, which)
        self.draw_embed(utterance.embed, utterance.name, which)

    def register_utterance(self, utterance: Utterance):
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)

        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        self.play_button.setDisabled(False)
        self.generate_button.setDisabled(False)
        self.synthesize_button.setDisabled(False)

    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)",
        )
        return Path(fpath[0]) if fpath[0] != "" else ""

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
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)

    def populate_models(
        self,
        encoder_models_dir: Path,
        synthesizer_models_dir: Path,
        vocoder_models_dir: Path,
    ):
        # Encoder
        encoder_fpaths = list(encoder_models_dir.glob("*.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % encoder_models_dir)
        self.repopulate_box(self.encoder_box, [(f.stem, f) for f in encoder_fpaths])

        # Synthesizer
        synthesizer_model_dirs = list(synthesizer_models_dir.glob("*"))
        synthesizer_items = [
            (f.name.replace("logs-", ""), f) for f in synthesizer_model_dirs
        ]
        if len(synthesizer_model_dirs) == 0:
            raise Exception(
                "No synthesizer models found in %s. For the synthesizer, the expected "
                "structure is <syn_models_dir>/logs-<model_name>/taco_pretrained/"
                "checkpoint" % synthesizer_models_dir
            )
        self.repopulate_box(self.synthesizer_box, synthesizer_items)

        # Vocoder
        vocoder_fpaths = list(vocoder_models_dir.glob("**/*.pt"))
        vocoder_items = [(f.stem, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(
                self.datasets_root,
                self.ui.current_dataset_name,
                self.ui.current_speaker_name,
                self.ui.current_utterance_name,
            )
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = (
                self.ui.current_dataset_name + "_" + self.ui.current_speaker_name
            )

            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        if fpath.suffix.lower() == ".mp3" and self.no_mp3_support:
            self.ui.log(
                "Error: No mp3 file argument was passed but an mp3 file was used"
            )
            return

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.browser_load_button.clicked.connect(lambda: self.load_from_browser())

        random_func = lambda level: lambda: self.ui.populate_browser(
            self.datasets_root, recognized_datasets, level
        )
        self.browser.random_dataset_button.clicked.connect(random_func(0))
        self.random_speaker_button.clicked.connect(random_func(1))
        self.random_utterance_button.clicked.connect(random_func(2))
        self.dataset_box.currentIndexChanged.connect(random_func(1))
        self.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection
        self.encoder_box.currentIndexChanged.connect(self.init_encoder)

        def func():
            self.synthesizer = None

        self.synthesizer_box.currentIndexChanged.connect(func)
        self.vocoder_box.currentIndexChanged.connect(self.init_vocoder)

        # Utterance selection
        self.browser_browse_button.clicked.connect(
            lambda: self.load_from_browser(self.ui.browse_file())
        )
        self.ui.utterance_history.currentIndexChanged.connect(
            lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        )

        # Utterance selection
        self.play_button.clicked.connect(
            lambda: self.ui.play(
                self.ui.selected_utterance.wav, Synthesizer.sample_rate
            )
        )
        self.stop_button.clicked.connect(self.ui.stop)
        self.record_button.clicked.connect(self.record)

        # Wav playback & save
        self.replay_wav_button.clicked.connect(lambda: self.replay_last_wav())
        self.export_wav_button.clicked.connect(lambda: self.export_current_wave())
        self.waves_cb.currentIndexChanged.connect(self.set_current_wav)

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath

        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return

        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    @property
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()

    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()

    @property
    def selected_utterance(self):
        return self.utterance_history.itemData(self.utterance_history.currentIndex())
