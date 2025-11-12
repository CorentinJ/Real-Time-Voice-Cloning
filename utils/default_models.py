import urllib.request
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

from tqdm import tqdm


default_models = {
    "encoder": ("https://drive.google.com/uc?export=download&id=1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1", 17090379),
    "synthesizer": ("https://drive.google.com/u/0/uc?id=1EqFMIbvxffxtjiVrtykroF6_mUh-5Z3s&export=download&confirm=t", 370554559),
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

    desc = f"Downloading {target.name}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        try:
            # Handle Google Drive large file downloads
            # Google Drive may return an HTML confirmation page for large files
            import re
            import http.cookiejar
            
            # Create a cookie jar to maintain session
            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
            
            # Create a request with headers to avoid being blocked
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
            
            with opener.open(req) as response:
                # Check if we got an HTML page (confirmation page)
                content = response.read()
                if b'<html' in content or b'<HTML' in content:
                    # Extract the form action URL and parameters
                    html_content = content.decode('utf-8', errors='ignore')
                    
                    # Try to find the form action URL (newer Google Drive format)
                    form_match = re.search(r'action="([^"]+)"', html_content)
                    if form_match:
                        action_url = form_match.group(1)
                        # Extract form parameters
                        id_match = re.search(r'name="id"\s+value="([^"]+)"', html_content)
                        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html_content)
                        
                        if id_match:
                            file_id = id_match.group(1)
                            confirm = confirm_match.group(1) if confirm_match else 't'
                            # Construct the actual download URL
                            actual_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"
                            urllib.request.urlretrieve(actual_url, filename=target, reporthook=t.update_to)
                        else:
                            # Fallback: use the action URL directly
                            urllib.request.urlretrieve(action_url, filename=target, reporthook=t.update_to)
                    else:
                        # Try older format: extract from href
                        match = re.search(r'href="(/uc\?[^"]+)"', html_content)
                        if match:
                            actual_url = 'https://drive.google.com' + match.group(1)
                            urllib.request.urlretrieve(actual_url, filename=target, reporthook=t.update_to)
                        else:
                            raise Exception("Could not extract download URL from Google Drive confirmation page")
                else:
                    # Direct download, save the content
                    with open(target, 'wb') as f:
                        f.write(content)
                        t.update(len(content))
        except (HTTPError, Exception) as e:
            print(f"Error downloading {target.name}: {e}")
            # Remove incomplete file
            if target.exists():
                target.unlink()
            return


def ensure_default_models(models_dir: Path):
    # Define download tasks
    jobs = []
    for model_name, (url, size) in default_models.items():
        target_path = models_dir / "default" / f"{model_name}.pt"
        if target_path.exists():
            if target_path.stat().st_size != size:
                print(f"File {target_path} is not of expected size, redownloading...")
            else:
                continue

        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    # Run and join threads
    for thread, target_path, size in jobs:
        thread.join()

        assert target_path.exists() and target_path.stat().st_size == size, \
            f"Download for {target_path.name} failed. You may download models manually instead.\n" \
            f"https://drive.google.com/drive/folders/1fU6umc5uQAVR2udZdHX-lDgXYzTyqG_j"
