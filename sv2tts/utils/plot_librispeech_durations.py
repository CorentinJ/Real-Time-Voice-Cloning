from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


fnames = ["_original", "_asr", "_asr_adjust"]
data = []
for fname in fnames:
    fpath = Path(r"E:\Datasets\SV2TTS\synthesizer\train%s.txt" % fname)
    with fpath.open("r") as metadata_file:
        metadata = [line.strip().split("|") for line in metadata_file]
    
    n_samples = np.array([int(line[3]) for line in metadata])
    n_samples = n_samples / 16000
    data.append(n_samples)


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
titles = ["Original", "Silence split", "Adjusted split + limits"]
for ax, title, datum in zip(axs, titles, data):
    ax.hist(datum, bins=range(21), alpha=0.5, histtype='bar', ec='black')
    hours = int(np.round(datum.sum() / 3600))
    ax.set_title(title + " (%d hours)" % hours)
    ax.set_xlim([-1, 21])
    ax.set_xticks(list(range(0, 25, 2)))
    ax.set_xlabel("Duration (seconds)")
    
    mean = np.mean(datum)
    ax.axvline(mean, color="red", label="mean (%.2fs)" % mean)
    median = np.median(datum)
    ax.axvline(median, color="green", label="median (%.2fs)" % median)

    ax.legend()
    
axs[0].set_ylabel("Number of utterances")
# fig.suptitle("LibriSpeech utterance durations", size=16)
plt.show()
