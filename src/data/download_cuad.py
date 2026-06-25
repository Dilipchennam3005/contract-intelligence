"""
Download and inspect the CUAD (Contract Understanding Atticus Dataset).
CUAD contains 510 contracts with 41 clause-type labels, formatted as SQuAD-style QA.
Source: https://github.com/TheAtticusProject/cuad
"""
import io
import json
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Official data zip from TheAtticusProject GitHub (discovered from cuad-qa.py loader)
CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

# Map of split name → filename inside the zip
SPLIT_FILES = {
    "train": "train_separate_questions.json",
    "test":  "test.json",
}


def download_cuad():
    zip_dest = DATA_DIR / "cuad_data.zip"

    if not zip_dest.exists():
        print(f"Downloading CUAD zip from GitHub ({CUAD_ZIP_URL})...")
        resp = requests.get(CUAD_ZIP_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="data.zip") as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  Saved {zip_dest}")
    else:
        print(f"  Zip already cached at {zip_dest}")

    print("Extracting splits...")
    paths = {}
    with zipfile.ZipFile(zip_dest) as zf:
        print(f"  Files in zip: {zf.namelist()[:10]}")
        for split, fname in SPLIT_FILES.items():
            dest = DATA_DIR / f"cuad_{split}.json"
            if dest.exists():
                print(f"  {split} already extracted")
            else:
                # File may be at root or inside a subdir
                matches = [n for n in zf.namelist() if n.endswith(fname)]
                if not matches:
                    print(f"  WARNING: {fname} not found in zip")
                    continue
                with zf.open(matches[0]) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted {split} -> {dest}")
            paths[split] = dest
    return paths


def load_squad_json(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                samples.append({
                    "id":       qa["id"],
                    "title":    article.get("title", ""),
                    "context":  ctx,
                    "question": qa["question"],
                    "answers":  qa["answers"],
                })
    return samples


def inspect_cuad(paths: dict):
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    all_splits = {}
    for split, path in paths.items():
        samples = load_squad_json(path)
        all_splits[split] = samples
        print(f"\n[{split.upper()}]  {len(samples):,} QA samples  (file: {path.name})")
        print(f"  Fields in each sample: {list(samples[0].keys())}")

    # Inspect first training sample
    print("\n--- First training sample ---")
    s = all_splits["train"][0]
    print(f"  id:       {s['id']}")
    print(f"  title:    {s['title']}")
    print(f"  question: {s['question']}")
    print(f"  context:  {repr(s['context'][:300])}...")
    print(f"  answers:  {s['answers']}")

    # Clause-type distribution
    print("\n--- Clause-type label distribution (train) ---")
    q_stats = {}
    for s in all_splits["train"]:
        q = s["question"]
        has_ans = bool(s["answers"])
        q_stats.setdefault(q, {"total": 0, "positive": 0})
        q_stats[q]["total"] += 1
        if has_ans:
            q_stats[q]["positive"] += 1

    print(f"  Unique clause-type questions (labels): {len(q_stats)}")
    print("\n  Top clause types by name and positive rate (% of contracts containing it):")
    for q, st in list(q_stats.items())[:15]:
        name = q.split("?")[0].replace("Highlight the parts", "").replace("that answer the question:", "").strip()
        name = name[:60] + "..." if len(name) > 60 else name
        rate = st["positive"] / st["total"] * 100
        print(f"    {rate:5.1f}%  pos | {name}")


if __name__ == "__main__":
    paths = download_cuad()
    inspect_cuad(paths)
    print("\nStep 1 complete. Data is in data/raw/")
