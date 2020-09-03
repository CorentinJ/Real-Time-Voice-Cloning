from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import torch
import librosa
from audioread.exceptions import NoBackendError

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

#Maximum of generated wavs to keep on memory
MAX_WAVES = 15

class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, low_mem, seed, no_mp3_support):
        if not no_mp3_support:
            try:
                librosa.load("samples/6829_00000.mp3")
            except NoBackendError:
                print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
                  "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
                exit(-1)
        self.no_mp3_support = no_mp3_support
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.low_mem = low_mem
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
        
        self.synthesizer = None # type: Synthesizer
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(enc_models_dir, syn_models_dir, voc_models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)
        
    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
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
        def func(): 
            self.synthesizer = None
        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        
        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        #Audio
        self.ui.setup_audio_devices(Synthesizer.sample_rate)

        #Wav playback & save
        func = lambda: self.replay_last_wav()
        self.ui.replay_wav_button.clicked.connect(func)
        func = lambda: self.export_current_wave()
        self.ui.export_wav_button.clicked.connect(func)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        self.ui.save_audio_file(self.current_wav, Synthesizer.sample_rate)

    def replay_last_wav(self):
        self.ui.play(self.current_wav, Synthesizer.sample_rate)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir, seed):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        self.ui.populate_gen_options(seed, self.trim_silences)
        
    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
            
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return 
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        if fpath.suffix.lower() == ".mp3" and self.no_mp3_support:
                self.ui.log("Error: No mp3 file argument was passed but an mp3 file was used")
                return

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)
        
    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return 
        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)
        
    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)
        
    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)
        
    def synthesize(self):
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)
        
        # Synthesize the spectrogram
        if self.synthesizer is None:
            model_dir = self.ui.current_synthesizer_model_dir
            checkpoints_dir = model_dir.joinpath("taco_pretrained")
            self.synthesizer = Synthesizer(checkpoints_dir, low_mem=self.low_mem)
        if not self.synthesizer.is_loaded():
            self.ui.log("Loading the synthesizer %s" % self.synthesizer.checkpoint_fpath)

        # Update the synthesizer random seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = self.synthesizer.set_seed(int(self.ui.seed_textbox.text()))
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = self.synthesizer.set_seed(None)
        
        texts = self.ui.text_prompt.toPlainText().split("\n")
        embed = self.ui.selected_utterance.embed
        embeds = np.stack([embed] * len(texts))
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = self.synthesizer.set_seed(int(self.ui.seed_textbox.text()))
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)
        if self.ui.current_vocoder_fpath is not None:
            self.ui.log("")
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")
        
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if self.ui.trim_silences_checkbox.isChecked():
            wav = encoder.preprocess_wav(wav)

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        self.ui.play(wav, Synthesizer.sample_rate)

        # Name it (history displayed in combobox)
        # TODO better naming for the combobox items?
        wav_name = str(self.waves_count + 1)

        #Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVES:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Update current wav
        self.set_current_wav(0)
        
        #Enable replay and save buttons:
        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)
        
        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)
        
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

    def update_seed_textbox(self):
       self.ui.update_seed_textbox() 
