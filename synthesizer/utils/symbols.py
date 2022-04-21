_pad = "_"
_eos = "<eos>"
_characters = 'abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = [
    'aa', 'aa0', 'aa1', 'aa2', 'ae', 'ae0', 'ae1', 'ae2', 'ah', 'ah0', 'ah1', 'ah2', 'ao', 'ao0', 'ao1', 'ao2', 'aw',
    'aw0', 'aw1', 'aw2', 'ay', 'ay0', 'ay1', 'ay2', 'b', 'ch', 'd', 'dh', 'eh', 'eh0', 'eh1', 'eh2', 'er', 'er0', 'er1',
    'er2', 'ey', 'ey0', 'ey1', 'ey2', 'f', 'g', 'hh', 'ih', 'ih0', 'ih1', 'ih2', 'iy', 'iy0', 'iy1', 'iy2', 'jh', 'k',
    'l',
    'm', 'n', 'ng', 'ow', 'ow0', 'ow1', 'ow2', 'oy', 'oy0', 'oy1', 'oy2', 'p', 'r', 's', 'sh', 't', 'th', 'uh',
    'uh0', 'uh1', 'uh2', 'uw', 'uw0', 'uw1', 'uw2', 'v', 'w', 'y', 'z', 'zh', 'vj', 'a0', 'a1', 'bj', 'c', 'dj', 'e0',
    'e1', 'fj', 'gj', 'h', 'hj', 'i0', 'i1', 'j', 'kj', 'lj', 'mj', 'nj', 'o0', 'o1', 'pj', 'rj', 'sch', 'sj', 'tj',
    'u0',
    'u1', 'y0', 'y1', 'zj', "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y",
    "Z", "ZH"]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet
