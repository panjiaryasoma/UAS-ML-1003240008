import pandas as pd

from src.eda import (
    build_quality_summary,
    prepare_modeling_data,
    save_eda_plots,
    split_modeling_data,
)


def test_quality_summary_detects_hidden_issues():
    df = pd.DataFrame({
        "text": [
            "bagus",
            "bagus",
            "ambigu",
            "ambigu",
            "   ",
            "__laugh__ mantapppp",
            "enaaak sekali",
            "bagus banget 😍",
        ],
        "label": [
            "positive",
            "positive",
            "positive",
            "negative",
            "neutral",
            "positive",
            "positive",
            "positive",
        ],
    })

    summary = build_quality_summary(df)
    counts = summary.set_index("issue")["count"].to_dict()

    assert counts["exact_duplicate_rows"] == 1
    assert counts["duplicate_text_rows"] == 4
    assert counts["conflicting_text_groups"] == 1
    assert counts["conflicting_label_rows"] == 2
    assert counts["blank_or_whitespace_text"] == 1
    assert counts["placeholder_token"] == 1
    assert counts["elongated_word_rows"] == 2
    assert counts["contains_emoji"] == 1


def test_prepare_modeling_data_removes_exact_duplicates():
    df = pd.DataFrame({
        "text": ["bagus", "bagus", "buruk"],
        "label": ["positive", "positive", "negative"],
    })

    result = prepare_modeling_data(df)

    assert len(result) == 2
    assert result.duplicated(subset=["text", "label"]).sum() == 0


def test_split_modeling_data_is_reproducible_and_stratified():
    df = pd.DataFrame({
        "text": [f"teks unik {i}" for i in range(30)],
        "label": (
            ["positive"] * 10
            + ["neutral"] * 10
            + ["negative"] * 10
        ),
    })

    train_a, test_a = split_modeling_data(df)
    train_b, test_b = split_modeling_data(df)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)

    assert len(train_a) == 24
    assert len(test_a) == 6
    assert test_a["label"].value_counts().to_dict() == {
        "positive": 2,
        "neutral": 2,
        "negative": 2,
    }


def test_split_modeling_data_prevents_duplicates_across_partitions():
    df = pd.DataFrame({
        "text": (
            [f"positive {i}" for i in range(10)]
            + [f"neutral {i}" for i in range(10)]
            + [f"negative {i}" for i in range(10)]
            + ["positive 0", "neutral 0", "negative 0"]
        ),
        "label": (
            ["positive"] * 10
            + ["neutral"] * 10
            + ["negative"] * 10
            + ["positive", "neutral", "negative"]
        ),
    })

    train_df, test_df = split_modeling_data(df)

    train_pairs = set(zip(train_df["text"], train_df["label"]))
    test_pairs = set(zip(test_df["text"], test_df["label"]))

    assert train_pairs.isdisjoint(test_pairs)
    assert len(train_df) + len(test_df) == 30


def test_save_eda_plots_creates_required_png_files(tmp_path):
    df = pd.DataFrame({
        "text": ["bagus sekali", "biasa saja", "buruk sekali"] * 2,
        "label": ["positive", "neutral", "negative"] * 2,
    })

    paths = save_eda_plots(df, tmp_path)

    assert [path.name for path in paths] == [
        "label_distribution.png",
        "missing_values_by_column.png",
        "text_length_by_label.png",
        "top_terms_by_label.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
