from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.eda import RANDOM_STATE, split_modeling_data
from src.load_data import load_dataset


matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def prepare_test_partition(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Mereproduksi split resmi dan hanya mengembalikan partisi test."""
    train_df, test_df = split_modeling_data(df)

    X_test = test_df["text"].copy()
    y_test = test_df["label"].copy()

    split_info = {
        "modeling_rows": int(len(train_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    return X_test, y_test, split_info


def load_model_and_metadata(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Memuat pipeline final dan metadata training."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. Jalankan python -m src.train."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {metadata_path}."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata_classes = metadata.get("model_selection", {}).get("classes", [])
    model_classes = [str(label) for label in getattr(model, "classes_", [])]

    if metadata_classes and model_classes != metadata_classes:
        raise ValueError(
            "Urutan kelas model tidak cocok dengan metadata: "
            f"model={model_classes}, metadata={metadata_classes}"
        )

    return model, metadata


def evaluate_model(
    model: Any,
    X_test: pd.Series,
    y_test: pd.Series,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    list[str],
]:
    """Menghitung metrik final dan tabel prediksi pada test set."""
    predictions = np.asarray(model.predict(X_test))
    labels = [str(label) for label in getattr(model, "classes_", [])]

    if not labels:
        labels = sorted({str(value) for value in y_test} | {
            str(value) for value in predictions
        })

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
        if probabilities.shape != (len(X_test), len(labels)):
            raise ValueError(
                "Bentuk probabilitas tidak cocok dengan jumlah data atau kelas."
            )
        confidence = probabilities.max(axis=1)
        probabilities_available = True
    else:
        probabilities = np.full(
            shape=(len(X_test), len(labels)),
            fill_value=np.nan,
            dtype=float,
        )
        confidence = np.full(len(X_test), np.nan, dtype=float)
        probabilities_available = False

    report_dict = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    predictions_df = pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "true_label": y_test.reset_index(drop=True),
        "predicted_label": predictions,
        "confidence": confidence,
    })
    predictions_df["is_correct"] = (
        predictions_df["true_label"]
        == predictions_df["predicted_label"]
    )

    for class_index, label in enumerate(labels):
        predictions_df[f"prob_{label}"] = probabilities[:, class_index]

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(
                y_test,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "test_rows": int(len(y_test)),
        "correct_rows": int(predictions_df["is_correct"].sum()),
        "incorrect_rows": int((~predictions_df["is_correct"]).sum()),
        "probabilities_available": probabilities_available,
        "prediction_distribution": {
            str(label): int(count)
            for label, count in predictions_df[
                "predicted_label"
            ].value_counts().sort_index().items()
        },
    }

    if probabilities_available:
        metrics["log_loss"] = float(
            log_loss(
                y_test,
                probabilities,
                labels=labels,
            )
        )

    return metrics, report_df, predictions_df, matrix, labels


def select_prediction_examples(
    predictions_df: pd.DataFrame,
    *,
    correct_per_class: int = 5,
    incorrect_limit: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Memilih contoh benar per kelas dan kesalahan paling percaya diri."""
    correct = predictions_df.loc[predictions_df["is_correct"]].copy()
    correct_examples = (
        correct.sort_values(
            by=["true_label", "confidence"],
            ascending=[True, False],
        )
        .groupby("true_label", group_keys=False)
        .head(correct_per_class)
        .reset_index(drop=True)
    )

    incorrect_examples = (
        predictions_df.loc[~predictions_df["is_correct"]]
        .sort_values(
            by=["confidence", "true_label", "predicted_label"],
            ascending=[False, True, True],
        )
        .head(incorrect_limit)
        .reset_index(drop=True)
    )

    return correct_examples, incorrect_examples


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> Path:
    """Menyimpan confusion matrix sebagai PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )
    axis.set_title("Confusion Matrix pada Test Set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path


def build_evaluation_summary(
    *,
    metrics: dict[str, Any],
    split_info: dict[str, int],
    labels: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Membangun ringkasan evaluasi final yang dapat direproduksi."""
    evaluated_at = datetime.now(timezone.utc).replace(microsecond=0)

    return {
        "model_version": metadata.get("model_version"),
        "evaluated_at_utc": evaluated_at.isoformat(),
        "dataset": metadata.get("dataset"),
        "selected_model": metadata.get(
            "model_selection",
            {},
        ).get("selected_model"),
        "labels": labels,
        "split": {
            **split_info,
            "random_state": RANDOM_STATE,
            "test_set_status": "evaluated_once",
        },
        "metrics": _to_json_safe(metrics),
        "training_runtime": metadata.get("runtime"),
    }


def save_evaluation_artifacts(
    *,
    summary: dict[str, Any],
    report_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    matrix: np.ndarray,
    labels: list[str],
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Menyimpan seluruh artefak evaluasi final."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "evaluation_summary.json"
    report_path = reports_dir / "classification_report.csv"
    matrix_path = reports_dir / "confusion_matrix.png"
    predictions_path = reports_dir / "test_predictions.csv"
    correct_path = reports_dir / "correct_prediction_examples.csv"
    errors_path = reports_dir / "error_analysis.csv"

    correct_examples, incorrect_examples = select_prediction_examples(
        predictions_df,
    )

    summary_path.write_text(
        json.dumps(
            _to_json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_df.to_csv(report_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    correct_examples.to_csv(correct_path, index=False)
    incorrect_examples.to_csv(errors_path, index=False)
    save_confusion_matrix(matrix, labels, matrix_path)

    return {
        "summary": summary_path,
        "classification_report": report_path,
        "confusion_matrix": matrix_path,
        "test_predictions": predictions_path,
        "correct_examples": correct_path,
        "error_analysis": errors_path,
    }


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    model, metadata = load_model_and_metadata()
    df = load_dataset()
    X_test, y_test, split_info = prepare_test_partition(df)

    expected_test_rows = (
        metadata
        .get("split", {})
        .get("locked_test_rows")
    )
    if (
        expected_test_rows is not None
        and int(expected_test_rows) != len(X_test)
    ):
        raise ValueError(
            "Jumlah test set tidak cocok dengan metadata training: "
            f"aktual={len(X_test)}, metadata={expected_test_rows}"
        )

    print("\n=== FINAL TEST EVALUATION ===")
    print(f"Model version: {metadata.get('model_version')}")
    print(f"Test rows dibuka sekali: {len(X_test)}")

    metrics, report_df, predictions_df, matrix, labels = evaluate_model(
        model,
        X_test,
        y_test,
    )
    summary = build_evaluation_summary(
        metrics=metrics,
        split_info=split_info,
        labels=labels,
        metadata=metadata,
    )
    paths = save_evaluation_artifacts(
        summary=summary,
        report_df=report_df,
        predictions_df=predictions_df,
        matrix=matrix,
        labels=labels,
    )

    print("\n=== METRIK FINAL ===")
    print(f"Accuracy    : {metrics['accuracy']:.6f}")
    print(f"F1-macro    : {metrics['f1_macro']:.6f}")
    print(f"F1-weighted : {metrics['f1_weighted']:.6f}")
    if "log_loss" in metrics:
        print(f"Log loss    : {metrics['log_loss']:.6f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report_df.to_string(index=False))

    print("\nArtefak evaluasi:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
