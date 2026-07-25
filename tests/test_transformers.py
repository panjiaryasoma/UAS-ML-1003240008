import pickle

import pandas as pd

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
