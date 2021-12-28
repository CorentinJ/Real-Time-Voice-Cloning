import urllib.request
from pathlib import Path
from threading import Thread

from tqdm import tqdm


default_models = {
    "encoder": ("https://drive.google.com/uc?export=download&id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1", 17090379),
    # Too large to put on google drive with a direct link...
    "synthesizer": ("https://download1075.mediafire.com/qo9z9gv56uwg/02w4p210tuudu3u/pretrained.pt", 370554559),
    "vocoder": ("https://drive.google.com/uc?export=download&id=1cf2NO6FtI0jDuy8AV3Xgn6leO6dHjIgu", 53845290),
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, target: Path, bar_pos=0):
    # Ensure the directory exists
    target.parent.mkdir(exist_ok=True, parents=True)

    desc = f"Downloading {target}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)


def ensure_default_models(models_dir: Path):
    # Define download tasks
    threads = []
    for model_name, (url, size) in default_models.items():
        target_path = models_dir / "default" / f"{model_name}.pt"
        if target_path.exists():
            if target_path.stat().st_size != size:
                print(f"File {target_path} is not of expected size, redownloading...")
            else:
                continue

        thread = Thread(target=download, args=(url, target_path, len(threads)))
        thread.start()
        threads.append(thread)

    # Run and join threads
    for thread in threads:
        thread.join()
