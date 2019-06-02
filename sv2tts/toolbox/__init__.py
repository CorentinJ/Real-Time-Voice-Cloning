from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer import inference as synthesizer
from vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np


recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
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
        self.utterances = set()
        
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
        
        # Utterance selection
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        
        # Generation
        self.ui.generate_button.clicked.connect(self.generate)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, False)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        
    def load_from_browser(self):
        fpath = Path(self.datasets_root,
                     self.ui.current_dataset_name,
                     self.ui.current_speaker_name,
                     self.ui.current_utterance_name)
        name = str(fpath.relative_to(self.datasets_root))
        speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
        
        # Select the next utterance
        if self.ui.auto_next_checkbox.isChecked():
            self.ui.browser_select_next()
        
        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = synthesizer.load_preprocess_wav(fpath)
        duration = len(wav) / synthesizer.sample_rate
        self.ui.log("Loaded %s" % name)

        # Compute the mel spectrogram
        spec = synthesizer.make_spectrogram(wav)
        
        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.load_preprocess_wav(fpath)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)
        
        # Plot it
        self.ui.draw_umap(self.utterances)
        self.ui.draw_utterance(utterance, "current")
        
    def generate(self):
        # Synthesize the spectrogram
        if not synthesizer.is_loaded():
            self.init_synthesizer()
        self.ui.log("Generating the mel spectrogram...")
        text = self.ui.text_prompt.toPlainText()
        embed = self.ui.selected_utterance.embed
        spec = synthesizer.synthesize_spectrogram(text, embed)
        
        # Synthesize the waveform
        if not vocoder.is_loaded():
            self.init_vocoder()
        self.ui.log("Generating the waveform...")
        if self.ui.current_vocoder_fpath is not None:
            wav = vocoder.infer_waveform(spec)
        else:
            wav = synthesizer.griffin_lim(spec)
        wav = wav / np.abs(wav).max() * 0.97
        
        # Play it
        self.ui.log("Playing the generated waveform")
        self.ui.play(wav, synthesizer.sample_rate)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.load_preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        speaker_name = "User"
        name = speaker_name + "_001"
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds)
        self.utterances.add(utterance)
        
        # Plot it
        self.ui.draw_umap(self.utterances)
        self.ui.draw_utterance(utterance, "generated")
        
    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath
        
        self.ui.log("Loading the encoder %s" % model_fpath)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Loaded the encoder in %dms." % int(1000 * (timer() - start)))
        
    def init_synthesizer(self):
        model_dir = self.ui.current_synthesizer_model_dir
        checkpoints_dir = model_dir.joinpath("taco_pretrained")

        # display_path = Path(checkpoint_fpath).relative_to(model_dir.parent)
        self.ui.log("Loading the synthesizer %s" % checkpoints_dir)
        start = timer()
        synthesizer.load_model(checkpoints_dir)
        self.ui.log("Loaded the synthesizer in %dms." % int(1000 * (timer() - start)))
    
    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return 
    
        self.ui.log("Loading the vocoder %s" % model_fpath)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Loaded the vocoder in %dms." % int(1000 * (timer() - start)))
