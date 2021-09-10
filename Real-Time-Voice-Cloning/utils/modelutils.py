from pathlib import Path

def check_model_paths(encoder_path: Path, synthesizer_path: Path, vocoder_path: Path):
    # This function tests the model paths and makes sure at least one is valid.
    if encoder_path.is_file() or encoder_path.is_dir():
        return
    if synthesizer_path.is_file() or synthesizer_path.is_dir():
        return
    if vocoder_path.is_file() or vocoder_path.is_dir():
        return

    # If none of the paths exist, remind the user to download models if needed
    print("********************************************************************************")
    print("Error: Model files not found. Follow these instructions to get and install the models:")
    print("https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models")
    print("********************************************************************************\n")
    quit(-1)
