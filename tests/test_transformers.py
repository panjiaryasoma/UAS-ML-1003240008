import pickle

import joblib
import numpy as np
import pandas as pd
import pytest

from src.transformers import IndonesianTextNormalizer


def test_normalizer_lowercases_trims_and_collapses_whitespace():
    normalizer = IndonesianTextNormalizer()

    result = normalizer.fit_transform([
        "  MAKANAN   INI\tENAK  ",
    ])

    assert result == ["makanan ini enak"]


def test_normalizer_handles_placeholders_and_elongated_words():
    normalizer = IndonesianTextNormalizer()

    result = normalizer.fit_transform([
        "__laugh__ mantapppp",
        "__sad__ sediiih sekali",
    ])

    assert result == [
        "laugh mantap",
        "sad sedih sekali",
    ]


def test_normalizer_maps_informal_negation_and_preserves_negation():
    normalizer = IndonesianTextNormalizer()

    result = normalizer.fit_transform([
        "gak bagus, tapi tidak buruk",
        "enggak mengecewakan",
        "bukan jelek dan belum selesai",
    ])

    assert result == [
        "tidak bagus, tapi tidak buruk",
        "tidak mengecewakan",
        "bukan jelek dan belum selesai",
    ]


def test_normalizer_handles_missing_values_without_mutating_input():
    series = pd.Series([
        "  BAGUS  ",
        None,
        pd.NA,
    ])
    original = series.copy(deep=True)
    normalizer = IndonesianTextNormalizer()

    result = normalizer.fit_transform(series)

    assert result == ["bagus", "", ""]
    pd.testing.assert_series_equal(series, original)


def test_normalizer_can_be_pickled_for_sklearn_pipeline():
    normalizer = IndonesianTextNormalizer()
    restored = pickle.loads(pickle.dumps(normalizer))

    result = restored.fit_transform([
        "__laugh__ ENAAAK",
    ])

    assert result == ["laugh enak"]


def test_normalizer_accepts_single_column_dataframe():
    normalizer = IndonesianTextNormalizer()

    df = pd.DataFrame({
        "text": [
            "  GAK   BAGUS  ",
            "__laugh__ ENAAAK",
        ],
    })

    result = normalizer.fit_transform(df)

    assert result == [
        "tidak bagus",
        "laugh enak",
    ]


def test_normalizer_accepts_single_column_numpy_array():
    normalizer = IndonesianTextNormalizer()

    values = np.array([
        ["  BELUM   SELESAI  "],
        ["mantapppp"],
    ], dtype=object)

    result = normalizer.fit_transform(values)

    assert result == [
        "belum selesai",
        "mantap",
    ]


def test_normalizer_rejects_dataframe_with_multiple_columns():
    normalizer = IndonesianTextNormalizer()
    df = pd.DataFrame({
        "text": ["bagus"],
        "extra": ["data"],
    })

    with pytest.raises(
        ValueError,
        match="hanya menerima DataFrame satu kolom",
    ):
        normalizer.fit_transform(df)


def test_normalizer_round_trip_with_joblib(tmp_path):
    normalizer = IndonesianTextNormalizer()
    path = tmp_path / "normalizer.joblib"

    joblib.dump(normalizer, path)
    restored = joblib.load(path)

    assert restored.transform(["GAK ENAAAK"]) == ["tidak enak"]

