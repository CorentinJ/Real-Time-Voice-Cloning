from vlibs import fileio
import numpy as np
from hparams import hparams

root = '/home/cjemine/data/Synthesizer2/'

# On the remote, Synthesizer2 has max 900 frames.
# Try with 600, if it works, increase to 700.

lines = fileio.read_all_lines(fileio.join(root, "train.txt"))
out = []
pruned = 0
intact = 0
for line in lines:
    line = line.rstrip()
    audio_fname, mel_fname, embed_fname, *_ = line.split('|')
    mel = np.load(fileio.join(root, "mels", mel_fname))
    if len(mel) > hparams.max_mel_frames:
        pruned += 1
    else:
        intact += 1
        out.append(line)
        if intact % 100 == 0:
            print("Kept: %d / Discarded: %d   (%.1f%% discarded)" % 
                  (intact, pruned, (pruned / (intact + pruned)) * 100))
out.append('')        
fileio.write_all_lines(fileio.join(root, "train_max_frames_%d.txt" % hparams.max_mel_frames), out)
print("Kept: %d / Discarded: %d   (%.1f%% discarded)" %
      (intact, pruned, (pruned / (intact + pruned)) * 100))