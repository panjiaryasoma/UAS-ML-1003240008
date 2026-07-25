from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLUMNS = ["text", "label"]
VALID_LABELS = {"positive", "neutral", "negative"}

DATA_FILES = {
    "train_preprocess.tsv": {
        "url": (
            "https://raw.githubusercontent.com/IndoNLP/indonlu/master/"
            "dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv"
        ),
        "git_blob_sha1": "05fa9e062b81bc57e72a4bd19f14ad14b7b0de8c",
    },
    "valid_preprocess.tsv": {
        "url": (
            "https://raw.githubusercontent.com/IndoNLP/indonlu/master/"
            "dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv"
        ),
        "git_blob_sha1": "4bbd081c3a96aba2ff2cacbddb4cb7b29b261c08",
    },
}


def git_blob_sha1(content: bytes) -> str:
    """Menghitung identitas blob Git untuk verifikasi versi dataset."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(
        header + content,
        usedforsecurity=False,
    ).hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_sha1: str,
) -> None:
    if destination.exists():
        current_sha1 = git_blob_sha1(destination.read_bytes())

        if current_sha1 == expected_sha1:
            print(f"[OK] Sudah tersedia: {destination.name}")
            return

        print(f"[INFO] Versi lokal berubah, mengunduh ulang: {destination.name}")

    print(f"[DOWNLOAD] {destination.name}")

    with urlopen(url, timeout=60) as response:
        content = response.read()

    actual_sha1 = git_blob_sha1(content)

    if actual_sha1 != expected_sha1:
        raise ValueError(
            f"Checksum {destination.name} tidak cocok. "
            "Dataset sumber mungkin telah berubah."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )
    temporary_path.write_bytes(content)
    temporary_path.replace(destination)


def download_dataset() -> dict[str, Path]:
    paths = {}

    for filename, source in DATA_FILES.items():
        destination = RAW_DATA_DIR / filename

        download_file(
            url=source["url"],
            destination=destination,
            expected_sha1=source["git_blob_sha1"],
        )
        paths[filename] = destination

    return paths


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=EXPECTED_COLUMNS,
        dtype="string",
        on_bad_lines="error",
    )


def validate_dataset(df: pd.DataFrame) -> None:
    if df.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Kolom harus {EXPECTED_COLUMNS}, "
            f"tetapi ditemukan {df.columns.tolist()}."
        )

    observed_labels = set(df["label"].dropna().unique())
    invalid_labels = observed_labels - VALID_LABELS

    if invalid_labels:
        raise ValueError(
            f"Label tidak valid ditemukan: {sorted(invalid_labels)}"
        )

    if df["label"].isna().any():
        raise ValueError("Dataset memiliki label kosong.")

    missing_labels = VALID_LABELS - observed_labels

    if missing_labels:
        raise ValueError(
            f"Kelas sentimen tidak lengkap: {sorted(missing_labels)}"
        )

    if len(df) < 1_000:
        raise ValueError(
            f"Dataset hanya memiliki {len(df)} baris; minimum 1.000."
        )


def load_dataset() -> pd.DataFrame:
    paths = download_dataset()

    train_df = read_split(paths["train_preprocess.tsv"])
    valid_df = read_split(paths["valid_preprocess.tsv"])

    df = pd.concat(
        [train_df, valid_df],
        ignore_index=True,
    )

    validate_dataset(df)
    return df


def print_profile(df: pd.DataFrame) -> None:
    print("\n=== PROFIL DATASET SMSA ===")
    print(f"Jumlah baris : {df.shape[0]:,}")
    print(f"Jumlah kolom : {df.shape[1]}")

    print("\nTipe setiap kolom:")
    print(df.dtypes.to_string())

    print("\nJumlah nilai hilang:")
    print(df.isna().sum().to_string())

    print("\nDistribusi label:")
    print(df["label"].value_counts(dropna=False).to_string())

    print(f"\nDuplikat identik: {df.duplicated().sum():,}")


def main() -> None:
    dataset = load_dataset()
    print_profile(dataset)


if __name__ == "__main__":
    main()