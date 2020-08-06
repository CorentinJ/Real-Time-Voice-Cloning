from collections import namedtuple

Utterance = namedtuple("Utterance", "name speaker_name wav spec embed partial_embeds synth")
Utterance.__eq__ = lambda x, y: x.name == y.name
Utterance.__hash__ = lambda x: hash(x.name)
