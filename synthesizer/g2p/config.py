import os
import json

import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfigEn(object):
    language = "EN"
    graphemes_path = f'synthesizer/g2p/resources/{language}/Graphemes.json'
    phonemes_path =  f'synthesizer/g2p/resources/{language}/Phonemes.json'
    lexicon_path =   f'synthesizer/g2p/resources/{language}/Lexicon.json'

class DataConfigRu(object):
    language = "RU"
    graphemes_path = f'synthesizer/g2p/resources/{language}/Graphemes.json'
    phonemes_path =  f'synthesizer/g2p/resources/{language}/Phonemes.json'
    lexicon_path =   f'synthesizer/g2p/resources/{language}/Lexicon.json'

class ModelConfigEn(object):
    with open(DataConfigEn.graphemes_path, encoding="utf8") as f:
        # print(f)
        graphemes_size = len(json.load(f))

    with open(DataConfigEn.phonemes_path) as f:
        phonemes_size = len(json.load(f))

    hidden_size = 128

class ModelConfigRu(object):
    with open(DataConfigRu.graphemes_path, encoding="utf8") as f:
        # print(f)
        graphemes_size = len(json.load(f))

    with open(DataConfigRu.phonemes_path) as f:
        phonemes_size = len(json.load(f))

    hidden_size = 128


class TrainConfigEn(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 3e-4
    batch_size = 128
    epochs = int(os.getenv('EPOCHS', '10'))
    log_path = f'log/{DataConfigEn.language}'

class TrainConfigRu(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 3e-4
    batch_size = 128
    epochs = int(os.getenv('EPOCHS', '10'))
    log_path = f'log/{DataConfigRu.language}'

class TestConfigEn(object):
    # device = cpu
    device = cpu
    encoder_model_path = f'synthesizer/g2p/models/{DataConfigEn.language}/encoder_e{TrainConfigEn.epochs:02}.pth'
    decoder_model_path = f'synthesizer/g2p/models/{DataConfigEn.language}/decoder_e{TrainConfigEn.epochs:02}.pth'

class TestConfigRu(object):
    # device = cpu
    device = cpu
    encoder_model_path = f'synthesizer/g2p/models/{DataConfigRu.language}/encoder_e{TrainConfigRu.epochs:02}.pth'
    decoder_model_path = f'synthesizer/g2p/models/{DataConfigRu.language}/decoder_e{TrainConfigRu.epochs:02}.pth'
