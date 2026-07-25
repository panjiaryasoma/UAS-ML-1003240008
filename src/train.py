from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.eda import RANDOM_STATE, split_modeling_data
from src.load_data import load_dataset
from src.transformers import IndonesianTextNormalizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CV_SPLITS = 5
CALIBRATION_CV = 3
SCORING = "f1_macro"


def build_text_pipeline(classifier: Any) -> Pipeline:
    """Membuat pipeline normalizer, TF-IDF, dan classifier."""
    return Pipeline([
        ("normalizer", IndonesianTextNormalizer()),
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=False,
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


def build_candidate_searches(
    *,
    cv_splits: int = CV_SPLITS,
    calibration_cv: int = CALIBRATION_CV,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict[str, GridSearchCV]:
    """Membuat tiga pencarian CV yang seluruhnya memberi probabilitas kelas."""
    outer_cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    nb_pipeline = build_text_pipeline(
        MultinomialNB(),
    )
    logistic_pipeline = build_text_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )

    svc_pipeline = build_text_pipeline(
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    calibrated_svc = CalibratedClassifierCV(
        estimator=svc_pipeline,
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=1,
    )

    return {
        "multinomial_nb": GridSearchCV(
            estimator=nb_pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "classifier__alpha": [0.5, 1.0],
            },
            scoring=SCORING,
            cv=outer_cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False,
        ),
        "logistic_regression": GridSearchCV(
            estimator=logistic_pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "classifier__C": [1.0, 2.0],
            },
            scoring=SCORING,
            cv=outer_cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False,
        ),
        "calibrated_linear_svc": GridSearchCV(
            estimator=calibrated_svc,
            param_grid={
                "estimator__tfidf__ngram_range": [(1, 1), (1, 2)],
                "estimator__classifier__C": [0.5, 1.0],
            },
            scoring=SCORING,
            cv=outer_cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False,
        ),
    }


def prepare_training_partition(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Mengembalikan fitur dan target train tanpa membuka isi test set."""
    train_df, test_df = split_modeling_data(df)

    X_train = train_df["text"].copy()
    y_train = train_df["label"].copy()

    split_info = {
        "modeling_rows": int(len(train_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "locked_test_rows": int(len(test_df)),
    }
    return X_train, y_train, split_info


def run_candidate_searches(
    X_train: pd.Series,
    y_train: pd.Series,
    searches: dict[str, GridSearchCV],
) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:
    """Menjalankan pencarian CV pada train set dan merangkum hasil terbaik."""
    rows: list[dict[str, Any]] = []
    fitted_searches: dict[str, GridSearchCV] = {}

    for name, search in searches.items():
        print(f"\n=== TRAINING: {name} ===")
        started = perf_counter()
        search.fit(X_train, y_train)
        elapsed = perf_counter() - started

        best_index = int(search.best_index_)
        rows.append({
            "model": name,
            "mean_cv_f1_macro": float(search.best_score_),
            "std_cv_f1_macro": float(
                search.cv_results_["std_test_score"][best_index]
            ),
            "mean_fit_time_seconds": float(
                search.cv_results_["mean_fit_time"][best_index]
            ),
            "total_search_seconds": float(elapsed),
            "best_params": _to_json_safe(search.best_params_),
        })
        fitted_searches[name] = search

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            by=["mean_cv_f1_macro", "std_cv_f1_macro", "model"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    return summary, fitted_searches


def select_best_model(
    summary: pd.DataFrame,
    fitted_searches: dict[str, GridSearchCV],
) -> tuple[str, Any]:
    """Memilih skor rata-rata tertinggi, lalu simpangan baku terendah."""
    if summary.empty:
        raise ValueError("Ringkasan CV kosong; tidak ada model yang dapat dipilih.")

    best_name = str(summary.iloc[0]["model"])
    if best_name not in fitted_searches:
        raise KeyError(f"Hasil pencarian untuk model '{best_name}' tidak tersedia.")

    return best_name, fitted_searches[best_name].best_estimator_


def build_training_metadata(
    *,
    best_name: str,
    best_model: Any,
    summary: pd.DataFrame,
    split_info: dict[str, int],
) -> dict[str, Any]:
    """Membangun metadata reproduksi tanpa memakai prediksi test set."""
    generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    classes = [
        str(label)
        for label in getattr(best_model, "classes_", [])
    ]

    return {
        "model_version": f"sentimenid-{generated_at:%Y%m%dT%H%M%SZ}",
        "generated_at_utc": generated_at.isoformat(),
        "dataset": {
            "name": "IndoNLU SmSA",
            "files": [
                "train_preprocess.tsv",
                "valid_preprocess.tsv",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "split": {
            **split_info,
            "test_size": 0.2,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "test_set_status": "locked_until_evaluate.py",
        },
        "model_selection": {
            "scoring": SCORING,
            "cv": {
                "type": "StratifiedKFold",
                "n_splits": CV_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "calibration_cv": CALIBRATION_CV,
            "selected_model": best_name,
            "classes": classes,
            "candidate_results": _to_json_safe(
                summary.to_dict(orient="records")
            ),
        },
        "pipeline": {
            "normalizer": "IndonesianTextNormalizer",
            "vectorizer": "TfidfVectorizer",
            "artifact_contains_full_preprocessing": True,
        },
    }


def save_training_artifacts(
    *,
    best_model: Any,
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Menyimpan estimator final, metadata, dan ringkasan CV."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    metadata_path = models_dir / "metadata.json"
    cv_results_path = reports_dir / "model_selection_results.csv"

    joblib.dump(best_model, model_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_summary = summary.copy()
    if "best_params" in csv_summary.columns:
        csv_summary["best_params"] = csv_summary["best_params"].map(
            lambda value: json.dumps(
                _to_json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    csv_summary.to_csv(cv_results_path, index=False)

    return {
        "model": model_path,
        "metadata": metadata_path,
        "cv_results": cv_results_path,
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
    df = load_dataset()
    X_train, y_train, split_info = prepare_training_partition(df)

    print("\n=== TRAIN PARTITION ===")
    print(f"X_train rows: {len(X_train)}")
    print(f"y_train rows: {len(y_train)}")
    print(f"Test rows tetap terkunci: {split_info['locked_test_rows']}")

    searches = build_candidate_searches()
    summary, fitted_searches = run_candidate_searches(
        X_train,
        y_train,
        searches,
    )

    best_name, best_model = select_best_model(
        summary,
        fitted_searches,
    )
    metadata = build_training_metadata(
        best_name=best_name,
        best_model=best_model,
        summary=summary,
        split_info=split_info,
    )
    paths = save_training_artifacts(
        best_model=best_model,
        metadata=metadata,
        summary=summary,
    )

    print("\n=== HASIL CROSS-VALIDATION ===")
    print(
        summary[
            [
                "model",
                "mean_cv_f1_macro",
                "std_cv_f1_macro",
                "total_search_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nModel terpilih: {best_name}")
    print("\nArtefak training:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
