import re

import torch
import threading
import _thread

from alphabet_detector import AlphabetDetector
from contextlib import contextmanager

from .config import DataConfigEn, DataConfigRu, ModelConfigEn, ModelConfigRu, TestConfigEn, TestConfigRu
from .data import PersianLexicon
from .model import Encoder, Decoder


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


def normalize_repetitions(word):
    chars = [""]
    for ch in word:
        if chars[-1] != ch:
            chars.append(ch)
    return "".join(chars)


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def g2p_all(word, dl_logger):
    if ad.is_latin(word):
        ourg2p = en_g2p
        word = word.upper()
        word = normalize_repetitions(word)  # because of some words like чшшшшш
    else:  # elif ad.is_cyrillic(word):
        ourg2p = ru_g2p
    try:
        with time_limit(4):
            res = ourg2p(word)
    except TimeoutException:
        dl_logger.log("WARNING", data={
            "timed out": word
        })
        res = ourg2p(word[:2])  # will do for some noises
    except Exception:  #
        syms = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя!.,:-" if ourg2p == ru_g2p else "abcdefghijklmnopqrstuvwxyz!.,:-"
        res = ourg2p("".join(s if s in syms else "" for s in word))
        dl_logger.log("WARNING", data={
            "fixed the word": word
        })
    return res


def s2ids(sentence, dl_logger):
    words = ["".join(s if s in "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz!.,:-" else "" for s in word)
             for word in re.split(regexPattern, sentence)]
    return [g2p_all(word, dl_logger) if word not in delims else word for word in words]


class ShortLogger:
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        print(args, kwargs)


inited = False


def g2p_main(sentence):
    if not inited:
        init()
    ids = s2ids(sentence, dl_logger)
    return [item for sublist in ids for item in sublist]


def init(dl_logger_=None):
    global dl_logger, inited, delims, regexPattern
    delims = [",", ".", " ", "!", ":", "-"]
    regexPattern = '|'.join('(?={})'.format(re.escape(delim)) for delim in delims)
    if dl_logger_ is None:
        try:
            from synthesizer.models.tacotron.train import dl_logger
        except Exception as e:
            print(e)
            dl_logger = ShortLogger()
    else:
        dl_logger = dl_logger_
    inited = True

