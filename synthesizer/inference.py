from multiprocess.context import SpawnContext
from synthesizer.hparams import hparams
from multiprocess.pool import Pool  # You're free to use either one
#from multiprocessing import Pool   # 
from synthesizer import audio
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams
    
    @staticmethod
    def _get_checkpoint_file(checkpoints_dir: Path):
        # Reimplementation of tf.train.get_checkpoint_state but without tensorflow
        index_fpath = checkpoints_dir.joinpath("checkpoint")
        if not index_fpath.exists():
            raise Exception("Could not find any synthesizer weights under %s" % checkpoints_dir)
        with index_fpath.open("r") as index_file:
            for line in index_file:
                if line.startswith("model_checkpoint_path"):
                    break
        checkpoint_fname = line[line.find(" ") + 1:].replace("\"", "").rstrip()
        checkpoint_fpath = checkpoints_dir.joinpath(checkpoint_fname)
        model_name = checkpoints_dir.parent.name.replace("logs-", "")
        step = int(checkpoint_fname[checkpoint_fname.rfind('-') + 1:])
        return checkpoint_fpath, model_name, step
    
    def __init__(self, checkpoints_dir: Path, verbose=True, low_mem=False):
        """
        Creates a synthesizer ready for inference. The actual model isn't loaded in memory until
        needed or until load() is called.
        
        :param checkpoints_dir: path to the directory containing the checkpoint file as well as the
        weight files (.data, .index and .meta files)
        :param verbose: if False, only tensorflow's output will be printed TODO: suppress them too
        :param low_mem: if True, the model will be loaded in a separate process and its resources 
        will be released after each usage. Adds a large overhead, only recommended if your GPU 
        memory is low (<= 2gb)
        """
        self.verbose = verbose
        self._low_mem = low_mem
        
        # Prepare the model
        self._model = None
        self.checkpoint_fpath, model_name, step = self._get_checkpoint_file(checkpoints_dir)
        if verbose:
            print("Found synthesizer \"%s\" trained to step %d" % (model_name, step))
     
    def is_loaded(self):
        """
        Whether the model is loaded in GPU memory.
        """
        return self._model is not None
    
    def load(self):
        """
        Effectively loads the model to GPU memory given the weights file that was passed in the
        constructor.
        """
        if self._low_mem:
            raise Exception("Cannot load the synthesizer permanently in low mem mode")
        
        from synthesizer.tacotron2 import Tacotron2
        self._model = Tacotron2(str(self.checkpoint_fpath), hparams)
            
    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        if not self._low_mem:
            # Usual inference mode: load the model on the first request and keep it loaded.
            if not self.is_loaded():
                self.load()
            specs, alignments = self._model.my_synthesize(embeddings, texts)
        else:
            # Low memory inference mode: load the model upon every request. The model has to be 
            # loaded in a separate process to be able to release GPU memory (a simple workaround 
            # to tensorflow's intricacies)
            specs, alignments = Pool(1, context=SpawnContext()).starmap(
                Synthesizer._one_shot_synthesize_spectrograms,
                [(self.checkpoint_fpath, embeddings, texts)]
            )[0]
    
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def _one_shot_synthesize_spectrograms(checkpoint_fpath, embeddings, texts):
        from synthesizer.tacotron2 import Tacotron2
        import numba.cuda
    
        # Load the model and forward the inputs
        model = Tacotron2(str(checkpoint_fpath), hparams)
        specs, alignments = model.my_synthesize(embeddings, texts)
        
        # Detach the outputs (not doing so will cause the process to hang)
        specs, alignments = [spec.copy() for spec in specs], alignments.copy()
        
        # Close cuda for this process
        model.session.close()
        numba.cuda.select_device(0)
        numba.cuda.close()
        
        return specs, alignments

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer. 
        """
        wav = librosa.load(fpath, hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that 
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav
        
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram
    
    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)
    