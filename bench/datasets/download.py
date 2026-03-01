"""Download LoCoMo and LongMemEval benchmark datasets."""

import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASETS_DIR = Path(__file__).parent


def download_locomo() -> Path:
    """Download LoCoMo dataset from GitHub (snap-research/locomo)."""
    dest = DATASETS_DIR / "locomo10.json"
    if dest.exists():
        print(f"LoCoMo already downloaded: {dest}")
        return dest
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"Downloading LoCoMo from {url} ...")
    urllib.request.urlretrieve(url, dest)
    with open(dest) as f:
        data = json.load(f)
    qa_count = sum(len(s["qa"]) for s in data)
    print(f"LoCoMo: {len(data)} conversations, {qa_count} QA pairs")
    return dest


def download_longmemeval_s() -> Path:
    """Download LongMemEval-S (115K tokens/question) from HuggingFace."""
    dest = DATASETS_DIR / "longmemeval_s.json"
    if dest.exists():
        print(f"LongMemEval-S already downloaded: {dest}")
        return dest
    print("Downloading LongMemEval-S from HuggingFace ...")
    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_s_cleaned.json",
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )
    # Rename to consistent name
    Path(path).rename(dest)
    with open(dest) as f:
        data = json.load(f)
    print(f"LongMemEval-S: {len(data)} questions")
    return dest


def download_longmemeval_oracle() -> Path:
    """Download LongMemEval-Oracle (small, evidence-only) from HuggingFace."""
    dest = DATASETS_DIR / "longmemeval_oracle.json"
    if dest.exists():
        print(f"LongMemEval-Oracle already downloaded: {dest}")
        return dest
    print("Downloading LongMemEval-Oracle from HuggingFace ...")
    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_oracle.json",
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )
    Path(path).rename(dest)
    with open(dest) as f:
        data = json.load(f)
    print(f"LongMemEval-Oracle: {len(data)} questions")
    return dest


if __name__ == "__main__":
    download_locomo()
    download_longmemeval_oracle()
    download_longmemeval_s()
    print("All datasets downloaded.")
