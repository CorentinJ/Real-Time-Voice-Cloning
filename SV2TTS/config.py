from vlibs import fileio
import torch

project_root = fileio.abspath(fileio.leafdir(__file__))

librispeech_root = "E://Datasets/LibriSpeech"
librispeech_datasets = ["train-other-500"]
voxceleb1_root = "E://Datasets/VoxCeleb1"
voxceleb2_root = "E://Datasets/VoxCeleb2"
voxceleb_datasets = ["voxceleb1", "voxceleb2"]
anglophone_nationalites = ['australia', 'canada', 'ireland', 'uk', 'usa']
clean_data_root = "E://Datasets//SpeakerEncoder"
all_datasets = librispeech_datasets + voxceleb_datasets

model_dir = fileio.join(project_root, "saved_models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
