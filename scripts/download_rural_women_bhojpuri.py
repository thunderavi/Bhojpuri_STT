from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_DATASET = "ai4bharat/Rural_Women_Bhojpuri"
DEFAULT_OUTPUT_DIR = Path("data") / "rural_women_bhojpuri"
DEFAULT_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache the AI4Bharat Rural Women Bhojpuri dataset locally."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the downloaded dataset (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional path to export the loaded dataset metadata as CSV after download.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Where Hugging Face should cache dataset files (default: {DEFAULT_CACHE_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(args.cache_dir)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")

    # Import after setting cache locations so Hugging Face picks up the F: drive.
    from datasets import load_dataset, load_from_disk

    if args.output_dir.exists():
        print(f"Loading existing dataset from disk: {args.output_dir}")
        dataset = load_from_disk(str(args.output_dir))
    else:
        print(f"Downloading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset, cache_dir=str(args.cache_dir / "datasets"))
        print(f"Saving dataset to: {args.output_dir}")
        dataset.save_to_disk(str(args.output_dir))

    print(dataset)
    for split_name, split in dataset.items():
        print(f"\nSplit: {split_name}")
        print(split)
        print(split.select(range(1)).remove_columns(["audio"])[0] if "audio" in split.column_names else split[0])

    if args.export_csv is not None:
        export_path = args.export_csv
        export_path.parent.mkdir(parents=True, exist_ok=True)
        first_split_name = next(iter(dataset.keys()))
        print(f"Exporting split '{first_split_name}' to CSV: {export_path}")
        dataset[first_split_name].to_csv(str(export_path))

    print(f"\nLocal dataset ready at: {args.output_dir.resolve()}")
    print(f"Hugging Face cache: {args.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
