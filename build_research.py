#!/usr/bin/env python3
"""Build research dataset by downloading from HuggingFace.

Downloads FineWeb + Wikipedia and samples ~400K docs.
Shows live progress throughout.
"""

import os, sys, json, time, random, gzip
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def download_fineweb(target: int, rng: random.Random, output_dir: Path):
    """Download FineWeb sample-10BT, sample target docs."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print(f"\n{'='*50}")
    print(f"FINE WEB — target: {target:,} docs")
    print(f"{'='*50}")

    # Try streaming first
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        collected = []
        pbar = tqdm(total=target, desc="FineWeb", unit="doc")
        for i, ex in enumerate(ds):
            text = ex.get("text", "")
            if len(text) < 50:
                continue
            collected.append({"text": text})
            pbar.update(1)
            if len(collected) >= target:
                break
        pbar.close()
        return collected
    except Exception as e:
        print(f"  Streaming failed: {e}")
        print("  Trying direct download...")

    # Fallback: download parquet files directly
    try:
        repo = "HuggingFaceFW/fineweb"
        config = "sample-10BT"
        files = [
            "sample/10BT/001_00000.parquet",
            "sample/10BT/001_00001.parquet",
            "sample/10BT/001_00002.parquet",
            "sample/10BT/001_00003.parquet",
        ]
        collected = []
        for fname in files:
            print(f"  Downloading {fname}...")
            path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            for i in range(len(table)):
                text = table[i]["text"].as_py()
                if text and len(text) >= 50:
                    collected.append({"text": text})
            print(f"    Got {len(collected):,} docs so far")
            if len(collected) >= target:
                break

        # Sample
        if len(collected) > target:
            collected = rng.sample(collected, target)
        print(f"  Total: {len(collected):,} docs")
        return collected
    except Exception as e:
        print(f"  Direct download failed: {e}")
        return []


def download_wikipedia(target: int, rng: random.Random, output_dir: Path):
    """Download English Wikipedia."""
    print(f"\n{'='*50}")
    print(f"WIKIPEDIA — target: {target:,} docs")
    print(f"{'='*50}")

    from datasets import load_dataset

    for ds_name, config_name in [
        ("culturax/processed_wikipedia", "en"),
        ("wikimedia/wikipedia", "20231101.en"),
    ]:
        try:
            ds = load_dataset(
                ds_name,
                name=config_name,
                split="train",
                streaming=True,
            )
            collected = []
            pbar = tqdm(total=target, desc="Wikipedia", unit="doc")
            for i, ex in enumerate(ds):
                text = ex.get("text", "")
                if not text or len(text) < 50:
                    continue
                collected.append({"text": text})
                pbar.update(1)
                if len(collected) >= target:
                    break
            pbar.close()
            print(f"  Total: {len(collected):,} docs")
            return collected
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
            continue

    print("  All Wikipedia sources failed")
    return []


def build(fineweb_docs=320000, wikipedia_docs=80000, output_dir="datasets/research", seed=42):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print("=" * 60)
    print("BUILDING RESEARCH DATASET")
    print("=" * 60)
    print(f"FineWeb:   {fineweb_docs:,} docs")
    print(f"Wikipedia: {wikipedia_docs:,} docs")
    print()

    # Download FineWeb
    fineweb = download_fineweb(fineweb_docs, rng, output_path)

    # Download Wikipedia
    wikipedia = download_wikipedia(wikipedia_docs, rng, output_path)

    # Combine
    all_docs = fineweb + wikipedia
    print(f"\nTotal before shuffle: {len(all_docs):,}")

    rng.shuffle(all_docs)
    print(f"Shuffled: {len(all_docs):,}")

    # Write
    corpus_path = output_path / "corpus.jsonl"
    print(f"\nWriting {corpus_path}...")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(all_docs, desc="Writing", unit="doc"):
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    # Metadata
    total_chars = sum(len(d.get("text", "")) for d in all_docs)
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_documents": len(all_docs),
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
        "sources": {
            "fineweb": len(fineweb),
            "wikipedia": len(wikipedia),
        },
        "seed": seed,
    }
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESEARCH DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"  FineWeb:   {len(fineweb):,} docs")
    print(f"  Wikipedia: {len(wikipedia):,} docs")
    print(f"  Total:     {len(all_docs):,} docs")
    print(f"  Chars:     {total_chars:,}")
    print(f"  Output:    {corpus_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb", type=int, default=320000)
    parser.add_argument("--wikipedia", type=int, default=80000)
    parser.add_argument("--output", default="datasets/research")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.fineweb, args.wikipedia, args.output, args.seed)