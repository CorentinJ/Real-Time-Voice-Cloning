from .symbols import symbols
from . import cleaners

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []
    sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))

    # Append EOS token
    sequence.append(_symbol_to_id["~"])
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
    if not cleaner:
        raise Exception("Unknown cleaner: %s" % name)
    text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s not in ("_", "~")
