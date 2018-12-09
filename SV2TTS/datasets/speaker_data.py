from vlibs import fileio, core
import numpy as np

# Set of utterances for a single speaker
class SpeakerData:
    def __init__(self, root, name):
        self.name = name
        self.utterances = fileio.get_files(root, r"\.npy")
        self.next_utterances = []
        
    def random_utterances(self, count):
        """
        Samples a batch of <count> unique utterances from the disk in a way that they all come up at 
        least once every two cycles and in a random order every time.
        
        :param count: The number of utterances to sample from the set of utterances from that 
        speaker. Utterances are guaranteed not to be repeated if <count> is not larger than the 
        number of utterances available.
        :return: A list of utterances loaded in memory. 
        """
        
        # Sample the utterances
        fpaths = []
        while count > 0:
            n = min(count, len(self.next_utterances))
            fpaths.extend(self.next_utterances[:n])
            self.next_utterances = self.next_utterances[n:]
            if len(self.next_utterances) == 0:
                new_utterances = [u for u in self.utterances if not u in fpaths[-n:]]
                self.next_utterances = core.shuffle(new_utterances)
            count -= n
        
        # Load them
        return list(map(np.load, fpaths))