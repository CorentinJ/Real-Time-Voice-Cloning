import os
import sys

# Ensure the repository root is on sys.path for imports like `import encoder`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_third_party_imports():
    import librosa  # noqa: F401
    import numpy  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401


def test_project_imports():
    import encoder  # noqa: F401
    import synthesizer  # noqa: F401
    import vocoder  # noqa: F401
