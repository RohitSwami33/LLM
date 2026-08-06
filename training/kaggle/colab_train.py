#!/usr/bin/env python3
"""Colab driver: resume research-v2 training from the Kaggle checkpoint dataset.

Run this as a script (not imported): `!python colab_train.py` in a Colab cell.
Uses the user's own Kaggle account credentials (upload kaggle.json to /content),
so it can reach the private corpus and checkpoint datasets and push new
checkpoints back. Colab GPU is separate from Kaggle's weekly quota.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    for pkg in ("kaggle", "sentencepiece"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    kaggle_json = Path("/content/kaggle.json")
    if not kaggle_json.exists():
        from google.colab import files
        uploaded = files.upload()
        kaggle_json = Path("/content") / next(iter(uploaded))
    home = Path.home()
    kdir = home / ".kaggle"
    kdir.mkdir(exist_ok=True)
    shutil.copy(kaggle_json, kdir / "kaggle.json")
    os.chmod(kdir / "kaggle.json", 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kdir)

    corpus = Path("/content/datasets_dl/corpus.jsonl")
    if not corpus.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("tomiokasan/research-v2-corpus",
                                   path="/content/datasets_dl", unzip=True, quiet=True)
        if not corpus.exists():
            raise RuntimeError("corpus download failed")

    sys.path.insert(0, "/content")
    import kaggle_pipeline
    kaggle_pipeline.run_pretokenize()
    kaggle_pipeline.run_training()


if __name__ == "__main__":
    main()
