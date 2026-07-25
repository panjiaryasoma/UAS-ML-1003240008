import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import (
    build_evaluation_summary,
    evaluate_model,
    prepare_test_partition,
    save_evaluation_artifacts,
    select_prediction_examples,
)
from src.transformers import IndonesianTextNormalizer


def make_balanced_text_df(rows_per_class: int = 10) -> pd.DataFrame:
    texts = []
    labels = []

    for index in range(rows_per_class):
        texts.append(f"bagus enak mantap pelayanan ramah nomor {index}")
        labels.append("positive")

        texts.append(f"buruk kecewa tidak enak pelayanan lambat nomor {index}")
        labels.append("negative")

        texts.append(f"informasi berita jadwal acara hari ini nomor {index}")
        labels.append("neutral")

    return pd.DataFrame({
        "text": texts,
        "label": labels,
    })


def build_fitted_model(df: pd.DataFrame) -> Pipeline:
    model = Pipeline([
        ("normalizer", IndonesianTextNormalizer()),
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=False,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                random_state=42,
            ),
        ),
    ])
    model.fit(df["text"], df["label"])
    return model


def test_prepare_test_partition_reproduces_locked_split():
    df = make_balanced_text_df(rows_per_class=10)

    X_test, y_test, split_info = prepare_test_partition(df)

    assert len(X_test) == 6
    assert len(y_test) == 6
    assert split_info == {
        "modeling_rows": 30,
        "train_rows": 24,
        "test_rows": 6,
    }
    assert X_test.name == "text"
    assert y_test.name == "label"


def test_evaluate_model_returns_metrics_report_and_probabilities():
    df = make_balanced_text_df(rows_per_class=10)
    model = build_fitted_model(df)
    X_test = pd.Series([
        "pelayanan bagus dan ramah",
        "pelayanan buruk dan lambat",
        "informasi jadwal hari ini",
    ], name="text")
    y_test = pd.Series([
        "positive",
        "negative",
        "neutral",
    ], name="label")

    metrics, report_df, predictions_df, matrix, labels = evaluate_model(
        model,
        X_test,
        y_test,
    )

    assert set(labels) == {"positive", "negative", "neutral"}
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1
    assert 0 <= metrics["f1_weighted"] <= 1
    assert metrics["test_rows"] == 3
    assert metrics["probabilities_available"] is True
    assert "log_loss" in metrics

    assert set(report_df["label"]).issuperset(set(labels))
    assert matrix.shape == (3, 3)
    assert len(predictions_df) == 3
    assert {
        "text",
        "true_label",
        "predicted_label",
        "confidence",
        "is_correct",
    }.issubset(predictions_df.columns)

    probability_columns = [
        column
        for column in predictions_df.columns
        if column.startswith("prob_")
    ]
    np.testing.assert_allclose(
        predictions_df[probability_columns].sum(axis=1),
        np.ones(3),
        atol=1e-6,
    )


def test_select_prediction_examples_limits_and_sorts_errors():
    predictions_df = pd.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f"],
        "true_label": [
            "positive",
            "positive",
            "negative",
            "negative",
            "neutral",
            "neutral",
        ],
        "predicted_label": [
            "positive",
            "negative",
            "negative",
            "positive",
            "neutral",
            "positive",
        ],
        "confidence": [0.7, 0.95, 0.8, 0.6, 0.9, 0.85],
        "is_correct": [True, False, True, False, True, False],
    })

    correct, incorrect = select_prediction_examples(
        predictions_df,
        correct_per_class=1,
        incorrect_limit=2,
    )

    assert len(correct) == 3
    assert len(incorrect) == 2
    assert incorrect["confidence"].tolist() == [0.95, 0.85]


def test_save_evaluation_artifacts_creates_required_files(tmp_path):
    df = make_balanced_text_df(rows_per_class=10)
    model = build_fitted_model(df)
    X_test = pd.Series([
        "pelayanan bagus",
        "pelayanan buruk",
        "informasi jadwal",
    ], name="text")
    y_test = pd.Series([
        "positive",
        "negative",
        "neutral",
    ], name="label")

    metrics, report_df, predictions_df, matrix, labels = evaluate_model(
        model,
        X_test,
        y_test,
    )
    metadata = {
        "model_version": "test-model",
        "dataset": {"name": "synthetic"},
        "model_selection": {
            "selected_model": "logistic_regression",
        },
        "runtime": {
            "python": "test",
            "scikit_learn": "test",
            "pandas": "test",
        },
    }
    summary = build_evaluation_summary(
        metrics=metrics,
        split_info={
            "modeling_rows": 30,
            "train_rows": 24,
            "test_rows": 6,
        },
        labels=labels,
        metadata=metadata,
    )

    paths = save_evaluation_artifacts(
        summary=summary,
        report_df=report_df,
        predictions_df=predictions_df,
        matrix=matrix,
        labels=labels,
        reports_dir=tmp_path,
    )

    assert set(paths) == {
        "summary",
        "classification_report",
        "confusion_matrix",
        "test_predictions",
        "correct_examples",
        "error_analysis",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())

    saved_summary = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    )
    assert saved_summary["model_version"] == "test-model"
    assert saved_summary["split"]["test_set_status"] == "evaluated_once"

    saved_report = pd.read_csv(paths["classification_report"])
    assert "label" in saved_report.columns

    saved_predictions = pd.read_csv(paths["test_predictions"])
    assert len(saved_predictions) == 3
