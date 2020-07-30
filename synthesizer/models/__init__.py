from .tacotron import Tacotron


def create_model(name, hparams):
  if name == "Tacotron":
    return Tacotron(hparams)
  else:
    raise Exception("Unknown model: " + name)
