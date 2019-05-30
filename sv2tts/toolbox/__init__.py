from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.synthesizer import Synthesizer
from synthesizer.hparams import hparams as synthesizer_hparams
from pathlib import Path
from time import perf_counter as timer
# from 
from tensorflow.train import get_checkpoint_state


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
    def __init__(self, datasets_root, encoder_models_dir, synthesizer_models_dir, 
                 vocoder_models_dir):
        self.datasets_root = datasets_root
        self.embeds = dict()
        self.synthesizer = None
        
        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        self.setup_events()
        self.ui.start()
        
    def setup_events(self):
        ## All the nasty UI events code
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(self.load_from_browser)
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))
        
        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        self.ui.synthesizer_box.currentIndexChanged.connect(self.init_synthesizer)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        
        # Generation
        self.ui.generate_button.clicked.connect(self.generate)

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
            # TODO: ordereddict
            self.embeds[speaker_name] = dict()
        self.embeds[speaker_name][utterance_name] = (embed, partial_embeds, wav_splits)
        
        # Draw the embed and the UMAP projection
        self.ui.draw_umap(self.embeds)
        
    def generate(self):
        # TODO
        self.init_synthesizer()

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, False)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        
    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath
        
        self.ui.log("Loading the encoder %s" % model_fpath)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Loaded the encoder in %dms." % int(1000 * (timer() - start)))
        
    def init_synthesizer(self):
        model_dir = self.ui.current_synthesizer_model_dir
        checkpoints_dir = model_dir.joinpath("taco_pretrained")
        checkpoint_fpath = get_checkpoint_state(checkpoints_dir).model_checkpoint_path

        display_path = Path(checkpoint_fpath).relative_to(model_dir.parent)
        self.ui.log("Loading the synthesizer %s" % display_path)
        start = timer()
        synth = Synthesizer()
        synth.load(checkpoint_fpath, synthesizer_hparams)
        self.ui.log("Loaded the synthesizer in %dms." % int(1000 * (timer() - start)))
    
    def init_vocoder(self):
        pass
