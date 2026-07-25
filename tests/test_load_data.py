import pandas as pd
import pytest

from src.load_data import read_split, validate_dataset


def test_read_split_parses_tsv(tmp_path):
    sample_path = tmp_path / "sample.tsv"
    sample_path.write_text(
        "produk ini bagus\tpositive\n"
        "biasa saja\tneutral\n"
        "sangat buruk\tnegative\n",
        encoding="utf-8",
    )

    df = read_split(sample_path)

    assert df.shape == (3, 2)
    assert df.columns.tolist() == ["text", "label"]
    assert df["label"].tolist() == ["positive", "neutral", "negative"]


def test_validate_dataset_rejects_unknown_label():
    df = pd.DataFrame({
        "text": ["produknya lumayan"],
        "label": ["mixed"],
    })

    with pytest.raises(ValueError, match="Label tidak valid"):
        validate_dataset(df)