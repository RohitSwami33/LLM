#!/usr/bin/env python3
"""Push the research_hybrid package to Kaggle as the research-moe-code dataset.

The v2 training kernel (training/kaggle/kaggle_pipeline_v2.py) mounts this
dataset and assembles it back into a research_hybrid/ package at boot.

Usage (from the repo root):
    .venv/bin/python tools/push_code_dataset.py

Notes:
    - Files are uploaded FLAT (dataset root), because the kaggle CLI fails on
      folder uploads ("Skipping folder"); the kernel reassembles the package.
    - kaggle 2.2.4's `_upload_file` never sets UploadFile.description (the
      dataset-side file name), which makes the server reject dataset creation
      with "Dataset url's dataset slugs and hashlink are all null"; we
      monkeypatch `_upload_file` to set description = file name.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from kaggle.api import kaggle_api_extended as K
from kaggle.api.kaggle_api_extended import KaggleApi

REPO = Path(__file__).resolve().parent.parent
OWNER = "tomiokasan"
SLUG = "research-moe-code"
DATASET_ID = f"{OWNER}/{SLUG}"


def patch_upload_file_name():
    orig = K.KaggleApi._upload_file

    def patched(self, file_name, full_path, blob_type, upload_context, quiet,
                resources, content_type=None):
        uf = orig(self, file_name, full_path, blob_type, upload_context, quiet,
                  resources, content_type)
        if uf is not None and not getattr(uf, "description", None):
            uf.description = file_name
        return uf

    K.KaggleApi._upload_file = patched


def main():
    patch_upload_file_name()

    tmp = Path(tempfile.mkdtemp(prefix="rhmoe_code_"))
    for f in sorted((REPO / "research_hybrid").glob("*.py")):
        shutil.copyfile(f, tmp / f.name)
    with open(tmp / "dataset-metadata.json", "w") as fh:
        json.dump({
            "id": DATASET_ID,
            "title": SLUG,
            "subtitle": "Hybrid-MoE-70M v2 research_hybrid package source",
            "description": "Python package for the research-v2-70m training kernel: "
                           "config, model, optim (MuonClip), train, attention, moe, "
                           "mamba2, mhc, mla, smoke_test, audit.",
            "licenses": [{"name": "other"}],
        }, fh, indent=2)

    api = KaggleApi()
    api.authenticate()
    print(f"Creating/updating dataset {DATASET_ID} from {tmp}")
    r = api.dataset_create_new(str(tmp), quiet=True, dir_mode="skip", convert_to_csv=False)
    err = getattr(r, "error", None)
    if err and "already in use" in err:
        print(f"Dataset exists; pushing a new version instead ({err})")
        r = api.dataset_create_version(str(tmp), version_notes="research_hybrid package update",
                                       quiet=True, dir_mode="skip", convert_to_csv=False)
        err = getattr(r, "error", None)
    if err:
        raise SystemExit(f"FAILED: {err}")
    print(f"OK: dataset {DATASET_ID} ready")


if __name__ == "__main__":
    main()
