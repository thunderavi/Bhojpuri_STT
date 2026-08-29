from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk


DEFAULT_AI4BHARAT_DIR = Path("data") / "rural_women_bhojpuri"
DEFAULT_LOCAL_CSV = Path("outputs") / "wav_transcripts.csv"
DEFAULT_LOCAL_WAV_DIR = Path("wav")
DEFAULT_OUTPUT_DIR = Path("data") / "merged_bhojpuri"
TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge AI4Bharat Bhojpuri data with local wav transcripts."
    )
    parser.add_argument(
        "--ai4bharat-dir",
        type=Path,
        default=DEFAULT_AI4BHARAT_DIR,
        help=f"Saved AI4Bharat dataset directory (default: {DEFAULT_AI4BHARAT_DIR}).",
    )
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=DEFAULT_LOCAL_CSV,
        help=f"CSV from wav transcription (default: {DEFAULT_LOCAL_CSV}).",
    )
    parser.add_argument(
        "--local-wav-dir",
        type=Path,
        default=DEFAULT_LOCAL_WAV_DIR,
        help=f"Directory containing local wav files (default: {DEFAULT_LOCAL_WAV_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the merged dataset (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--local-eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of local wav data to reserve for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the local train/eval split.",
    )
    return parser.parse_args()


def read_local_transcripts(csv_path: Path, wav_dir: Path) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError(f"Local transcript CSV not found: {csv_path}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"Local wav directory not found: {wav_dir}")

    audio_paths: list[str] = []
    texts: list[str] = []
    sources: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("file") or "").strip()
            transcript = (row.get("transcript") or "").strip()
            if not file_name or not transcript:
                continue

            audio_path = wav_dir / file_name
            if not audio_path.exists():
                continue

            audio_paths.append(str(audio_path.resolve()))
            texts.append(transcript)
            sources.append("local_wav")

    if not audio_paths:
        raise ValueError("No usable local wav rows were found in the transcript CSV.")

    dataset = Dataset.from_dict(
        {
            "audio": audio_paths,
            "text": texts,
            "source": sources,
        }
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    return dataset


def standardize_ai4bharat_split(split, source_name: str) -> Dataset:
    columns = list(split.column_names)
    audio_column = "audio" if "audio" in columns else None
    text_column = "text" if "text" in columns else None

    if audio_column is None or text_column is None:
        raise ValueError(f"Expected audio/text columns in split, got: {columns}")

    source_values = [source_name] * len(split)
    standardized = split.remove_columns([column for column in columns if column not in {"audio", "text"}])
    standardized = standardized.add_column("source", source_values)
    return standardized


def main() -> None:
    args = parse_args()

    if not args.ai4bharat_dir.exists():
        raise FileNotFoundError(f"AI4Bharat dataset folder not found: {args.ai4bharat_dir}")

    print(f"Loading AI4Bharat dataset from: {args.ai4bharat_dir}")
    ai4bharat = load_from_disk(str(args.ai4bharat_dir))

    train_parts = []
    eval_parts = []

    local_dataset = read_local_transcripts(args.local_csv, args.local_wav_dir)
    local_split = local_dataset.train_test_split(test_size=args.local_eval_fraction, seed=args.seed)

    # Put local WAV rows first so small smoke-test slices hit files we know are present locally.
    train_parts.append(local_split["train"])
    eval_parts.append(local_split["test"])

    if "train_real" in ai4bharat:
        train_parts.append(standardize_ai4bharat_split(ai4bharat["train_real"], "ai4bharat_train_real"))
    if "train_synthetic" in ai4bharat:
        train_parts.append(standardize_ai4bharat_split(ai4bharat["train_synthetic"], "ai4bharat_train_synthetic"))
    if "benchmark" in ai4bharat:
        eval_parts.append(standardize_ai4bharat_split(ai4bharat["benchmark"], "ai4bharat_benchmark"))

    merged_train = concatenate_datasets(train_parts)
    merged_eval = concatenate_datasets(eval_parts)

    merged = DatasetDict({"train": merged_train, "eval": merged_eval})

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged dataset to: {args.output_dir}")
    merged.save_to_disk(str(args.output_dir))

    print(merged)
    print(f"Train rows: {len(merged['train'])}")
    print(f"Eval rows: {len(merged['eval'])}")
    print(f"Merged dataset ready at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
