"""Prepare the two local IoT CSV datasets from ZIP files or Kaggle.

Usage: python scripts/prepare_data.py --ton-archive PATH [--cic-archive PATH]
If --cic-archive is omitted, the public CICIoT2023 Kaggle ZIP is downloaded.
"""

import argparse
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[1]
KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/himadri07/ciciot2023"


def extract_one(archive_path: Path, member: str, destination: Path) -> None:
    """Extract only an expected CSV member and atomically replace the destination."""
    if destination.is_file():
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with ZipFile(archive_path) as archive, archive.open(member) as source, temporary.open("wb") as output:
            copyfileobj(source, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Extracted: {destination} ({destination.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ton-archive", type=Path, help="First ZIP, containing train_test_network.csv")
    parser.add_argument("--cic-archive", type=Path, help="Second ZIP; omit to download the public Kaggle copy")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    args = parser.parse_args()
    ton_target = args.data_dir / "ton_iot" / "train_test_network.csv"
    if not ton_target.is_file():
        if args.ton_archive is None or not args.ton_archive.is_file():
            parser.error("Provide --ton-archive with the first ZIP file")
        extract_one(args.ton_archive, "train_test_network.csv", ton_target)

    expected = {split: args.data_dir / "ciciot2023" / split / f"{split}.csv"
                for split in ("train", "test", "validation")}
    if all(path.is_file() for path in expected.values()):
        print("CICIoT2023 CSVs already present")
        return
    downloaded = None
    archive_path = args.cic_archive
    if archive_path is None:
        downloaded = args.data_dir / "ciciot2023_download.zip"
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        temporary = downloaded.with_suffix(".zip.part")
        print("Downloading CICIoT2023 from Kaggle (about 500 MB)...", flush=True)
        try:
            with urlopen(KAGGLE_DOWNLOAD, timeout=60) as source, temporary.open("wb") as output:
                copyfileobj(source, output)
            temporary.replace(downloaded)
        finally:
            temporary.unlink(missing_ok=True)
        archive_path = downloaded
    if not archive_path.is_file():
        parser.error(f"CICIoT2023 ZIP not found: {archive_path}")
    try:
        for split, destination in expected.items():
            extract_one(archive_path, f"CICIOT23/{split}/{split}.csv", destination)
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
