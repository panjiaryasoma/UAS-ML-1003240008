import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.train import (
    build_candidate_searches,
    build_training_metadata,
    prepare_training_partition,
    run_candidate_searches,
    save_training_artifacts,
    select_best_model,
)


def make_balanced_text_df(rows_per_class: int = 12) -> pd.DataFrame:
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


def shrink_param_grid(search: GridSearchCV) -> None:
    search.param_grid = {
        key: [values[0]]
        for key, values in search.param_grid.items()
    }


def test_prepare_training_partition_separates_text_and_target():
    df = make_balanced_text_df(rows_per_class=10)

    X_train, y_train, split_info = prepare_training_partition(df)

    assert len(X_train) == 24
    assert len(y_train) == 24
    assert split_info == {
        "modeling_rows": 30,
        "train_rows": 24,
        "locked_test_rows": 6,
    }
    assert X_train.name == "text"
    assert y_train.name == "label"
    assert set(y_train.unique()) == {"positive", "negative", "neutral"}


def test_build_candidate_searches_has_required_models_and_cv_contract():
    searches = build_candidate_searches(
        cv_splits=5,
        calibration_cv=3,
        n_jobs=1,
        verbose=0,
    )

    assert set(searches) == {
        "multinomial_nb",
        "logistic_regression",
        "calibrated_linear_svc",
    }

    for search in searches.values():
        assert isinstance(search, GridSearchCV)
        assert search.scoring == "f1_macro"
        assert search.refit is True
        assert search.cv.n_splits == 5
        assert search.cv.shuffle is True
        assert search.cv.random_state == 42

    assert isinstance(
        searches["multinomial_nb"].estimator,
        Pipeline,
    )
    assert isinstance(
        searches["logistic_regression"].estimator,
        Pipeline,
    )

    calibrated = searches["calibrated_linear_svc"].estimator
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert isinstance(calibrated.estimator, Pipeline)
    assert calibrated.method == "sigmoid"
    assert calibrated.cv == 3


def test_all_candidates_fit_and_expose_probabilities():
    df = make_balanced_text_df(rows_per_class=12)
    X = df["text"]
    y = df["label"]

    searches = build_candidate_searches(
        cv_splits=2,
        calibration_cv=2,
        n_jobs=1,
        verbose=0,
    )
    for search in searches.values():
        shrink_param_grid(search)

    summary, fitted = run_candidate_searches(X, y, searches)

    assert set(summary["model"]) == set(searches)
    assert summary["mean_cv_f1_macro"].between(0, 1).all()
    assert summary["std_cv_f1_macro"].ge(0).all()

    samples = [
        "pelayanan sangat bagus dan ramah",
        "pelayanan tidak bagus dan lambat",
        "informasi jadwal acara hari ini",
    ]

    for search in fitted.values():
        probabilities = search.best_estimator_.predict_proba(samples)

        assert probabilities.shape == (3, 3)
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(3),
            atol=1e-6,
        )


def test_select_best_model_uses_mean_then_standard_deviation():
    summary = pd.DataFrame([
        {
            "model": "model_b",
            "mean_cv_f1_macro": 0.80,
            "std_cv_f1_macro": 0.04,
        },
        {
            "model": "model_a",
            "mean_cv_f1_macro": 0.80,
            "std_cv_f1_macro": 0.02,
        },
    ]).sort_values(
        by=["mean_cv_f1_macro", "std_cv_f1_macro", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    class FakeSearch:
        def __init__(self, estimator):
            self.best_estimator_ = estimator

    fitted = {
        "model_a": FakeSearch("estimator_a"),
        "model_b": FakeSearch("estimator_b"),
    }

    name, estimator = select_best_model(summary, fitted)

    assert name == "model_a"
    assert estimator == "estimator_a"


def test_training_artifacts_write_joblib_metadata_and_cv_results(tmp_path):
    df = make_balanced_text_df(rows_per_class=8)
    X = df["text"]
    y = df["label"]

    searches = build_candidate_searches(
        cv_splits=2,
        calibration_cv=2,
        n_jobs=1,
        verbose=0,
    )
    search = searches["multinomial_nb"]
    shrink_param_grid(search)
    search.fit(X, y)

    summary = pd.DataFrame([{
        "model": "multinomial_nb",
        "mean_cv_f1_macro": float(search.best_score_),
        "std_cv_f1_macro": float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
        "mean_fit_time_seconds": float(
            search.cv_results_["mean_fit_time"][search.best_index_]
        ),
        "total_search_seconds": 0.1,
        "best_params": search.best_params_,
    }])

    split_info = {
        "modeling_rows": 30,
        "train_rows": 24,
        "locked_test_rows": 6,
    }
    metadata = build_training_metadata(
        best_name="multinomial_nb",
        best_model=search.best_estimator_,
        summary=summary,
        split_info=split_info,
    )

    paths = save_training_artifacts(
        best_model=search.best_estimator_,
        metadata=metadata,
        summary=summary,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    assert paths["model"].is_file()
    assert paths["metadata"].is_file()
    assert paths["cv_results"].is_file()

    restored = joblib.load(paths["model"])
    assert restored.predict(["pelayanan bagus"]).shape == (1,)

    saved_metadata = json.loads(
        paths["metadata"].read_text(encoding="utf-8")
    )
    assert saved_metadata["model_selection"]["selected_model"] == "multinomial_nb"
    assert saved_metadata["split"]["test_set_status"] == "locked_until_evaluate.py"

    saved_results = pd.read_csv(paths["cv_results"])
    assert saved_results.loc[0, "model"] == "multinomial_nb"
