from toolbox.ui import UI
from encoder import inference as encoder
from pathlib import Path
from time import perf_counter as timer


recognized_datasets = [
    "Librispeech/dev-clean",
    "Librispeech/dev-other",
    "Librispeech/test-clean",
    "Librispeech/test-other",
    "Librispeech/train-clean-100",
    "Librispeech/train-clean-360",
    "Librispeech/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VCTK-Corpus/wav48",
]


class Toolbox:
    def __init__(self, datasets_root, encoder_fpath):
        self.datasets_root = datasets_root
        self.embeds = dict()
        self.encoder_fpath = encoder_fpath
        
        # Initialize the events and the interface
        self.ui = UI()
        self.setup_events()
        self.init()
        self.ui.start()
        
    def setup_events(self):
        self.ui.browser_load_button.clicked.connect(self.load_from_browser)

    def load_from_browser(self):
        fpath = Path(self.datasets_root,
                     self.ui.current_dataset_name,
                     self.ui.current_speaker_name,
                     self.ui.current_utterance_name)
        utterance_name = str(fpath.relative_to(self.datasets_root))
        speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
        
        # Select the next utterance
        if self.ui.auto_next_checkbox.isChecked():
            self.ui.browser_select_next()
        
        # Get the wav from the disk
        wav = encoder.load_preprocess_waveform(fpath)
        self.ui.log("Loaded %s" % utterance_name)
        self.embed_utterance(utterance_name, wav, speaker_name)
            
    def embed_utterance(self, utterance_name, wav, speaker_name):
        if not encoder.is_loaded():
            self.init_encoder()
        
        # Compute the embeddings
        embed, partial_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)
        
        # Add the embedding to the speaker
        if not speaker_name in self.embeds:
            self.embeds[speaker_name] = dict()
        self.embeds[speaker_name][utterance_name] = (embed, partial_embeds, wav_splits)
        
        # Draw the embed and the UMAP projection
        self.draw_embed()
        self.draw_umap()
    

    def init(self):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, False)
        
    def init_encoder(self):
        self.ui.log("Loading the encoder for the first time...")
        start = timer()
        encoder.load_model(self.encoder_fpath)
        self.ui.log("Loaded the encoder in %dms" % int(1000 * (timer() - start)))
        