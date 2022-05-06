import json
import os
import shutil

from tqdm import tqdm


# This is the parser for the GOLOS dataset
# Change the dataroot path so that it leads to train_opus dir
# it will reorganize the structure so that you can then process the ds to the standard SV2TTS format

if __name__ == '__main__':
    dataroot = "D:/datasets/golos/train_opus/"
    with open(dataroot + 'manifest.jsonl') as f:
        for i, line in enumerate(tqdm(f, total=1103799)):
            k = i // 10000 % 10
            filepath, textpath, text = json.loads(line)["audio_filepath"], \
                                       json.loads(line)["audio_filepath"].replace("opus", "txt"), \
                                       json.loads(line)["text"]
            if not os.path.isfile(dataroot + filepath.replace("crowd", f"crowd/{k}")):
                shutil.move(dataroot + filepath, dataroot + filepath.replace("crowd", f"crowd/{k}"))
                f = open(dataroot + textpath.replace("crowd", f"crowd/{k}"), "a", encoding="utf8")
                f.write(text)
                f.close()

