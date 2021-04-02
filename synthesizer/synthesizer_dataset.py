import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence
import sys
import math
import random
from synthesizer.utils.padding import pad_1D, pad_2D

np.set_printoptions(threshold=sys.maxsize)


class SynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path, duration_dir: Path, hparams):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s\n\t%s" %
              (metadata_fpath, mel_dir, embed_dir, duration_dir))

        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]

        mel_fnames = [x[1] for x in metadata if int(x[5])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[3] for x in metadata if int(x[5])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        duration_fnames = [x[2] for x in metadata if int(x[5])]
        duration_fpaths = [duration_dir.joinpath(
            fname) for fname in duration_fnames]
        self.samples_fpaths = list(
            zip(mel_fpaths, embed_fpaths, duration_fpaths))
        self.samples_phonemes = [x[7].strip() for x in metadata if int(x[5])]
        self.metadata = metadata
        self.hparams = hparams

        print("Found %d samples" % len(self.samples_fpaths))

    def __getitem__(self, index):
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        mel_path, embed_path, duration_path = self.samples_fpaths[index]
        # Load the mel spectrograms
        mel = np.load(mel_path).astype(np.float32)

        # mel = torch.from_numpy(mel)

        # Load the embed
        embed = np.load(embed_path).astype(np.float32)
        # embed = torch.from_numpy(embed)

        # Load the duration
        duration = np.load(duration_path)

        # Get the text and clean it
        text = np.array(text_to_sequence(
            self.samples_phonemes[index], self.hparams.tts_cleaner_names))

        return text, mel, embed, duration, index

    def __len__(self):
        return len(self.samples_fpaths)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step=1, sort=True):
        self.n_frames_per_step = n_frames_per_step
        self.sort = sort

    def __call__(self, batch):

        len_arr = np.array([x[0].shape[0] for x in batch])
        index_arr = np.argsort(-len_arr)

        batch_size = len(batch)
        real_batchsize = int(math.sqrt(batch_size))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(
                    np.arange(i*real_batchsize, (i+1)*real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        random.shuffle(output)
        return output

    def reprocess(self, batch, cut_list):
        ids = [batch[ind][-1] for ind in cut_list]

        texts = [batch[ind][0] for ind in cut_list]
        mel_targets = [batch[ind][1] for ind in cut_list]
        embeds = [batch[ind][2] for ind in cut_list]
        durations = [batch[ind][3] for ind in cut_list]

        for text, D, id_ in zip(texts, durations, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)

        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        durations = pad_1D(durations)
        embeds = pad_1D(embeds)
        mel_targets = pad_2D(mel_targets)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "durations": durations,
               "embeds": embeds,
               "src_len": length_text,
               "mel_len": length_mel}
        #    "log_D": log_Ds,
        #    "f0": f0s,
        #    "energy": energies,
        # if hp.use_spk_embed:
        #     out.update({"spk_ids": spk_ids})

        return out


# class TextMelCollate():
#     """ Zero-pads model inputs and targets based on number of frames per setep
#     """

#     def __init__(self, n_frames_per_step=1):
#         self.n_frames_per_step = n_frames_per_step

#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [text_normalized, mel_normalized]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]),
#             dim=0, descending=True)

#         max_input_len = input_lengths[0]

#         text_padded = torch.LongTensor(len(batch), max_input_len)
#         text_padded.zero_()

#         for i in range(len(ids_sorted_decreasing)):
#             text = batch[ids_sorted_decreasing[i]][0]
#             text_padded[i, :text.size(0)] = text

#         # Right zero-pad mel-spec
#         num_mels = batch[0][1].size(0)

#         max_target_len = max([x[1].size(1) for x in batch])

#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += self.n_frames_per_step - \
#                 max_target_len % self.n_frames_per_step
#             assert max_target_len % self.n_frames_per_step == 0

#         # include mel padded and gate padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         gate_padded = torch.FloatTensor(len(batch), max_target_len)
#         gate_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][1]
#             mel_padded[i, :, :mel.size(1)] = mel
#             gate_padded[i, mel.size(1)-1:] = 1
#             output_lengths[i] = mel.size(1)

#         embeds = torch.LongTensor([x[2] for x in batch])

#         return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, embeds
