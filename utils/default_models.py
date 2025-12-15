from pathlib import Path

from huggingface_hub import hf_hub_download

HUGGINGFACE_REPO = "CorentinJ/SV2TTS"

default_models = {
    "encoder": 17090379,
    "synthesizer": 370554559,
    "vocoder": 53845290,
}


def _download_model(model_name: str, target_dir: Path):
    hf_hub_download(
        repo_id=HUGGINGFACE_REPO,
        revision="main",
        filename=f"{model_name}.pt",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )


def ensure_default_models(models_dir: Path):
    target_dir = models_dir / "default"
    target_dir.mkdir(parents=True, exist_ok=True)

    for model_name, expected_size in default_models.items():
        target_path = target_dir / f"{model_name}.pt"

        if target_path.exists():
            if target_path.stat().st_size == expected_size:
                continue
            print(f"File {target_path} is not of expected size, redownloading...")

        _download_model(model_name, target_dir)

        assert target_path.exists() and target_path.stat().st_size == expected_size, (
            f"Download for {target_path.name} failed. You may download models manually instead.\n"
            f"https://huggingface.co/{HUGGINGFACE_REPO}"
        )
