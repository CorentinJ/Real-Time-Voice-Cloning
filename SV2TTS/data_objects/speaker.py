from vlibs import fileio
from vlibs.structs.random_cycler import RandomCycler
from data_objects.utterance import Utterance

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root):
        self.root = root
        self.name = fileio.leaf(root)
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        sources = fileio.read_all_lines(fileio.join(self.root, 'sources.txt'))
        sources = list(map(lambda l: l.split(' '), sources))
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(fileio.join(self.root, f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial_utterances(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()
        
        utterances = self.utterance_cycler.sample(count)
        return [(u,) + u.random_partial_utterance(n_frames) for u in utterances]
