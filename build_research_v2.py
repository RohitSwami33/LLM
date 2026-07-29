#!/usr/bin/env python3
import os, sys, json, time, random, yaml, traceback
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SourceReport:
    def __init__(self, name):
        self.name = name
        self.status = "pending"
        self.docs = 0
        self.failure_reason = ""
        self.time_seconds = 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "documents": self.docs,
            "failure_reason": self.failure_reason,
            "time_seconds": round(self.time_seconds, 2),
        }


def download_with_fallback(source_cfg, target, rng, report, cache_dir):
    name = source_cfg.get("name", "unknown")
    cache_file = cache_dir / f"{name}.jsonl"
    if cache_file.exists():
        docs = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        if docs:
            report.status = "cached"
            report.docs = len(docs)
            print(f"  {name}: cached {len(docs):,} docs")
            return docs
    try:
        docs = _download_source(source_cfg, target, rng, report)
        if docs:
            with open(cache_file, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            report.status = "ok"
            report.docs = len(docs)
            return docs
        report.status = "empty"
        report.failure_reason = "No documents found"
        return []
    except Exception as e:
        report.status = "failed"
        report.failure_reason = str(e)
        print(f"  {name} FAILED: {e}")
        traceback.print_exc()
        return []


def _download_source(source_cfg, target, rng, report):
    from datasets import load_dataset
    from tqdm import tqdm
    name = source_cfg["name"]
    hf_path = source_cfg["hf_path"]
    hf_config = source_cfg.get("hf_config")
    split = source_cfg.get("split", "train")
    text_key = source_cfg.get("text_key", "text")
    start = time.time()
    print(f"\n{'='*60}")
    print(f"{name} - target: {target:,}")
    print(f"{'='*60}")
    if source_cfg.get("multi_config"):
        docs = []
        for cfg_name in source_cfg["multi_config"]:
            if len(docs) >= target:
                break
            try:
                ds = load_dataset(hf_path, name=cfg_name, split=split, streaming=True)
                pbar = tqdm(total=target, desc=f"  {cfg_name}", unit="doc")
                for ex in ds:
                    text = ex.get(text_key, "")
                    if text and len(text.strip()) >= 50:
                        docs.append({"text": text, "source": name})
                        pbar.update(1)
                        if len(docs) >= target:
                            break
                pbar.close()
            except Exception as e:
                print(f"  {cfg_name}: {e}")
        report.time_seconds = time.time() - start
        return docs[:target]
    else:
        ds = load_dataset(hf_path, name=hf_config, split=split, streaming=True)
        docs = []
        pbar = tqdm(total=target, desc=name, unit="doc")
        for ex in ds:
            text = ex.get(text_key, "")
            if text and len(text.strip()) >= 50:
                docs.append({"text": text, "source": name})
                pbar.update(1)
                if len(docs) >= target:
                    break
        pbar.close()
        report.time_seconds = time.time() - start
        return docs[:target]

SOURCES = [
    {
        "name": "fineweb",
        "hf_path": "HuggingFaceFW/fineweb",
        "hf_config": "sample-10BT",
        "split": "train",
        "text_key": "text",
        "target": 250000,
    },
    {
        "name": "wikipedia",
        "hf_path": "wikimedia/wikipedia",
        "hf_config": "20231101.en",
        "split": "train",
        "text_key": "text",
        "target": 70000,
    },
    {
        "name": "codesearchnet",
        "hf_path": "code-search-net/code_search_net",
        "multi_config": ["python", "javascript", "java"],
        "split": "train",
        "text_key": "code",
        "target": 40000,
    },
    {
        "name": "code_python",
        "hf_path": "bigcode/the-stack-dedup",
        "hf_config": "python",
        "split": "train",
        "text_key": "content",
        "target": 40000,
        "fallback_for": "codesearchnet",
    },
    {
        "name": "openstax",
        "hf_path": "HuggingFaceTB/openstax_paragraphs",
        "split": "train",
        "text_key": "text",
        "target": 15000,
    },
    {
        "name": "arxiv",
        "hf_path": "open-index/open-arxiv",
        "split": "train",
        "text_key": "abstract",
        "target": 10000,
    },
    {
        "name": "finemath",
        "hf_path": "HuggingFaceTB/finemath",
        "hf_config": "finemath-3plus",
        "split": "train",
        "text_key": "text",
        "target": 15000,
    },
]

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/research_v2.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rng = random.Random(cfg.get("seed", 42))
    out_dir = Path(cfg["output"]["path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RESEARCH CORPUS v2 - FAULT-TOLERANT BUILD")
    print("=" * 60)

    reports = []
    downloaded = {}

    for src in SOURCES:
        name = src["name"]
        target = src.get("target", 10000)
        report = SourceReport(name)
        fallback = src.get("fallback_for")
        if fallback:
            if fallback in downloaded and downloaded[fallback]:
                print(f"\n  Skipping {name} - fallback already satisfied by {fallback}")
                report.status = "skipped"
                report.failure_reason = f"Already have {fallback}"
                reports.append(report)
                continue
        docs = download_with_fallback(src, target, rng, report, cache_dir)
        if docs:
            downloaded[name] = docs
        reports.append(report)

    all_docs = []
    for name, docs in downloaded.items():
        all_docs.extend(docs)

    rng.shuffle(all_docs)

    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_chars = sum(len(d.get("text", "")) for d in all_docs)
    total_words = sum(len(d.get("text", "").split()) for d in all_docs)

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_documents": len(all_docs),
        "total_chars": total_chars,
        "total_words": total_words,
        "estimated_tokens": total_chars // 4,
        "avg_doc_len_chars": round(total_chars / max(len(all_docs), 1)),
        "avg_doc_len_words": round(total_words / max(len(all_docs), 1)),
        "sources": {name: len(docs) for name, docs in downloaded.items()},
        "seed": cfg.get("seed", 42),
        "corpus_version": "v2",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report_data = {
        "created_at": metadata["created_at"],
        "total_documents": len(all_docs),
        "sources": [r.to_dict() for r in reports],
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{'='*60}")
    print("BUILD REPORT")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'Status':<12} {'Docs':>10} {'Reason'}")
    print("-" * 70)
    for r in reports:
        reason = r.failure_reason if r.failure_reason else ""
        print(f"{r.name:<25} {r.status:<12} {r.docs:>10,} {reason}")
    print("-" * 70)
    print(f"{'TOTAL':<25} {'':12} {len(all_docs):>10,}")
    print(f"{'='*60}")

    print(f"\nCorpus: {corpus_path}")
    print(f"Chars: {total_chars:,} | Words: {total_words:,} | Tokens ~{total_chars//4:,}")

    from training.tokenizer.tokenizer import train_tokenizer
    tok_dir = out_dir / "tokenizer_data"
    tok_dir.mkdir(exist_ok=True)
    vocab = cfg.get("tokenizer", {}).get("vocab_size", 32768)
    print(f"\nTraining tokenizer (vocab={vocab})...")
    train_tokenizer(
        data_path=str(corpus_path),
        output_path=str(tok_dir / "tokenizer"),
        vocab_size=vocab,
        tokenizer_type="sentencepiece",
        model_type="bpe",
    )
    print("Done!")


if __name__ == "__main__":
    main()
