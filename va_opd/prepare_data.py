# Copyright 2026 VA-OPD authors. Licensed under Apache-2.0.
"""Prepare image-embedded parquet files with a recorded, disjoint validation split."""

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASETS = {
    "geometry3k": "hiyouga/geometry3k",
    "virl39k": "PAPOGalaxy/PAPO_ViRL39K_train",
}


def validation_indices(size, count=200, seed=42):
    if size < count:
        raise ValueError(f"Need at least {count} held-out examples, found {size}.")
    return sorted(random.Random(seed).sample(range(size), count))


def normalize_rows(table, dataset, split):
    """Keep questions unchanged; embed images so parquet has no host-specific paths."""
    from io import BytesIO

    from PIL import Image

    for index, row in enumerate(table):
        question = row.get("problem", row.get("question"))
        answer = row.get("answer")
        images = row.get("images", row.get("image"))
        if not isinstance(question, str) or answer is None or images is None:
            raise ValueError(f"Row {index} requires problem/question, answer, and image(s).")
        images = images if isinstance(images, list) else [images]
        encoded = []
        for image in images:
            if isinstance(image, dict) and image.get("bytes") is not None:
                encoded.append({"bytes": image["bytes"], "path": None})
            else:
                if isinstance(image, dict):
                    image = image.get("path")
                if isinstance(image, (str, Path)):
                    image = Image.open(image)
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported image payload at row {index}: {type(image).__name__}")
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
                encoded.append({"bytes": buffer.getvalue(), "path": None})
        if not encoded:
            raise ValueError(f"Row {index} has no images.")
        yield {
            "problem": question,
            "answer": str(answer),
            "images": encoded,
            "sample_id": f"{dataset}:{split}:{index}",
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        }


def write_dataset(output, train, validation=None, seed=42, source=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = list(train)
    if not train:
        raise ValueError("The training set is empty.")

    # Hash question + image bytes to detect cross-split duplicates even if their IDs differ.
    def identity(row):
        digest = hashlib.sha256(row["problem"].encode())
        for image in row["images"]:
            digest.update(image["bytes"])
        return digest.hexdigest()

    train_ids = [row["sample_id"] for row in train]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicate training sample IDs.")
    manifest = {
        "dataset": "geometry3k" if validation is not None else "virl39k",
        "source": source,
        "seed": seed,
        "train_ids": train_ids,
        "train_rows": len(train),
        "validation_ids": [],
    }
    manifest["train_content_sha256"] = [identity(row) for row in train]
    heldout = None
    if validation is not None:
        validation = list(validation)
        selected = validation_indices(len(validation), seed=seed)
        heldout = [validation[i] for i in selected]
        heldout_ids = [row["sample_id"] for row in heldout]
        if set(train_ids) & set(heldout_ids):
            raise ValueError("Training/validation sample IDs overlap.")
        train_hashes = {identity(row) for row in train}
        if train_hashes & {identity(row) for row in heldout}:
            raise ValueError("Training/validation examples overlap by question and image content.")
        manifest.update(
            validation_ids=heldout_ids,
            validation_rows=len(heldout),
            validation_source_split="validation",
            split_note="200 from the official validation split; historical paper IDs were not supplied.",
        )
        manifest["validation_content_sha256"] = [identity(row) for row in heldout]
    # Validate before writing; a failed split must not leave a seemingly ready dataset.
    pq.write_table(pa.Table.from_pylist(train), output / "train.parquet")
    if heldout is not None:
        pq.write_table(pa.Table.from_pylist(heldout), output / "validation200.parquet")
    manifest["files"] = {}
    for file in output.glob("*.parquet"):
        manifest["files"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="geometry3k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision recorded in the manifest.")
    parser.add_argument("--train-file", type=Path, help="Local source parquet instead of downloading.")
    parser.add_argument("--validation-file", type=Path, help="Local official validation parquet (Geometry3K).")
    args = parser.parse_args()
    from datasets import load_dataset

    source = {"repo_id": DATASETS[args.dataset], "revision": args.revision}
    if args.train_file:
        files = {"train": str(args.train_file)}
        if args.validation_file:
            files["validation"] = str(args.validation_file)
        raw = load_dataset("parquet", data_files=files)
        source.update(local_train=str(args.train_file.resolve()))
    else:
        raw = load_dataset(DATASETS[args.dataset], revision=args.revision)
    source["fingerprints"] = {name: data._fingerprint for name, data in raw.items()}
    validation = None
    if args.dataset == "geometry3k":
        if "validation" not in raw:
            raise ValueError("Geometry3K requires the official validation split; never select on its test split.")
        validation = normalize_rows(raw["validation"], args.dataset, "validation")
    manifest = write_dataset(
        args.output_dir or Path("data") / args.dataset,
        normalize_rows(raw["train"], args.dataset, "train"),
        validation,
        seed=args.seed,
        source=source,
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "train_rows": manifest["train_rows"],
                "validation_rows": len(manifest["validation_ids"]),
                "output_dir": str(args.output_dir or Path("data") / args.dataset),
            },
            indent=2,
        )
    )
    if args.dataset == "virl39k":
        print("Also prepare Geometry3K: its validation200.parquet selects the checkpoint for both datasets.")


if __name__ == "__main__":
    main()
