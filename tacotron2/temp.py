from vlibs import fileio
import numpy as np

root = r'E:\Datasets\Synthesizer'

lines = fileio.read_all_lines(fileio.join(root, "train.txt"))
out = []
pruned = 0
intact = 0
for line in lines:
    line = line.rstrip()
    audio_fname, mel_fname, embed_fname, *_ = line.split('|')
    mel = np.load(fileio.join(root, "mels", mel_fname))
    if len(mel) > 900:
        fileio.remove(fileio.join(root, "audio", audio_fname))
        fileio.remove(fileio.join(root, "mels", mel_fname))
        fileio.remove(fileio.join(root, "embed", embed_fname))
        pruned += 1
    else:
        intact += 1
        out.append(line)
        if intact %100 == 0:
            print("%d / %d" % (intact, pruned))
out.append('')        
fileio.write_all_lines(fileio.join(root, "train2.txt"), out)
print("%d / %d" % (intact, pruned))
