from vlibs import fileio, core
from datasets.utterance import Utterance


# Set of utterances for a single speaker
class Speaker:
    def __init__(self, root):
        self.name = fileio.leaf(root)
        sources = fileio.read_all_lines(fileio.join(root, 'sources.txt'))
        sources = list(map(lambda l: l.split(' '), sources))
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(fileio.join(root, f), w) for f, w in sources.items()]
        self.next_utterances = []
               
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
        if count > len(self.utterances):
            raise Exception('Not enough utterances available')
        
        # Sample the utterances
        utterances = []
        while count > 0:
            n = min(count, len(self.next_utterances))
            utterances.extend(self.next_utterances[:n])
            self.next_utterances = self.next_utterances[n:]
            if len(self.next_utterances) == 0:
                new_utterances = [u for u in self.utterances if not u in utterances[-n:]]
                self.next_utterances = core.shuffle(new_utterances)
            count -= n
        
        return [(u,) + u.random_partial_utterance(n_frames) for u in utterances]
