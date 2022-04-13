from .data import PersianLexicon
from .model import Encoder, Decoder
from .config import DataConfigEn, DataConfigRu, ModelConfigEn, ModelConfigRu, TestConfigEn, TestConfigRu

from alphabet_detector import AlphabetDetector

import torch


def load_model(model_path, model, lang):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage,
        loc: storage
    ))
    model.to(TestConfigEn.device if lang == "en" else TestConfigRu.device)
    model.eval()
    return model


class G2P(object):
    def __init__(self, lang):
        # data
        self.DataConfig = DataConfigEn if lang == "en" else DataConfigRu
        self.ModelConfig = ModelConfigEn if lang == "en" else ModelConfigRu
        self.ds = PersianLexicon(
            self.DataConfig.graphemes_path,
            self.DataConfig.phonemes_path,
            self.DataConfig.lexicon_path
        )

        # model
        self.encoder_model = Encoder(
            self.ModelConfig.graphemes_size,
            self.ModelConfig.hidden_size
        ).to(TestConfigEn.device if lang == "en" else TestConfigRu.device)
        load_model(TestConfigEn.encoder_model_path if lang == "en" else TestConfigRu.encoder_model_path,
                   self.encoder_model, lang)

        self.decoder_model = Decoder(
            self.ModelConfig.phonemes_size,
            self.ModelConfig.hidden_size
        ).to(TestConfigEn.device if lang == "en" else TestConfigRu.device)
        load_model(TestConfigEn.decoder_model_path if lang == "en" else TestConfigRu.decoder_model_path,
                   self.decoder_model, lang)
        self.lang = lang

    def __call__(self, word):
        x = [0] + [self.ds.g2idx[ch] for ch in word] + [1]
        x = torch.tensor(x).long().unsqueeze(1).to(TestConfigEn.device if self.lang == "en" else TestConfigRu.device)
        with torch.no_grad():
            enc = self.encoder_model(x)

        phonemes, att_weights = [], []
        x = torch.zeros(1, 1).long().to(TestConfigEn.device if self.lang == "en" else TestConfigRu.device)
        hidden = torch.ones(
            1,
            1,
            self.ModelConfig.hidden_size
        ).to(TestConfigEn.device if self.lang == "en" else TestConfigRu.device)
        t = 0
        while True:
            with torch.no_grad():
                # print(x.device, enc.device, hidden.device)
                out, hidden, att_weight = self.decoder_model(
                    x,
                    enc,
                    hidden
                )

            att_weights.append(att_weight.detach().cpu())
            max_index = out[0, 0].argmax()
            x = max_index.unsqueeze(0).unsqueeze(0)
            t += 1

            phonemes.append(self.ds.idx2p[max_index.item()])
            if max_index.item() == 1:
                break

        return phonemes
ru_g2p = G2P(lang="ru")
en_g2p = G2P(lang="en")
ad = AlphabetDetector()

def g2p_all(word):
    if ad.is_latin(word):
        ourg2p = en_g2p
        word = word.upper()
    else:  #elif ad.is_cyrillic(word):
        ourg2p = ru_g2p
    try:
        res = ourg2p(word)
    except Exception as e:  #weird shit, inspecting it rn
        print(e)
        print(word)
        try:
            res = ru_g2p("".join(s if s in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" else "" for s in word))
        except Exception as e1:
            print(e1)
            res = ourg2p("о")  # just so its not blank
            print("err_word: ", word)
            print("ourg2p: ", ourg2p)
    return res

def s2ids(sentence):
    words = ["".join(s if s in "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz" else "" for s in word)
             for word in sentence.split(" ")]
    return [g2p_all(word) for word in words]

def g2p_main(sentence):
    ids = s2ids(sentence)
    return [item for sublist in ids for item in sublist]