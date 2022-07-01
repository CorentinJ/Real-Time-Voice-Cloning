from synthesizer.utils.symbols import symbols
from synthesizer.g2p import g2p_main

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S T AH0 N} Street."

      Args:
        text: array of one- or two-char strings resembling the
        syllables, for ex: ['o0', 'b', 'r', 'y1', 'j', '<eos>', 'vj', 'e0', 'ch', 'e0', 'r']

      Returns:
        List of integers corresponding to the symbols in the text
    """
    text = g2p_main(text)
    sequence = [_symbol_to_id[symbol] for symbol in text if symbol in symbols]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    # print(sequence)
    return "".join([_id_to_symbol[symbol] for symbol in sequence])\
        .replace("<eos>", " ").strip().capitalize().replace("_", "")
