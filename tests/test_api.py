from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class FakeSentimentModel:
    """Model kecil untuk menguji kontrak API tanpa membuka model proyek."""

    classes_ = np.array(["negative", "neutral", "positive"], dtype=object)

    def __init__(self) -> None:
        self.received_batches: list[list[str]] = []

    def predict(self, texts):
        batch = list(texts)
        self.received_batches.append(batch)

        predictions = []
        for text in batch:
            lowered = text.lower()
            if "tidak" in lowered or "buruk" in lowered:
                predictions.append("negative")
            elif "jadwal" in lowered or "informasi" in lowered:
                predictions.append("neutral")
            else:
                predictions.append("positive")

        return np.asarray(predictions, dtype=object)

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            if "tidak" in lowered or "buruk" in lowered:
                rows.append([0.85, 0.10, 0.05])
            elif "jadwal" in lowered or "informasi" in lowered:
                rows.append([0.10, 0.80, 0.10])
            else:
                rows.append([0.05, 0.05, 0.90])

        return np.asarray(rows, dtype=float)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    # File model hanya perlu ada. joblib.load diganti dengan fake model agar
    # unit test cepat dan tidak bergantung pada artefak training lokal.
    model_path.write_bytes(b"fake-model")
    metadata_path.write_text(
        """
        {
          "model_version": "sentimenid-test-v1",
          "model_selection": {
            "selected_model": "fake_model",
            "classes": ["negative", "neutral", "positive"]
          }
        }
        """,
        encoding="utf-8",
    )

    fake_model = FakeSentimentModel()
    load_calls: list[Path] = []

    def fake_joblib_load(path):
        load_calls.append(Path(path))
        return fake_model

    monkeypatch.setattr(
        "app.main.joblib.load",
        fake_joblib_load,
    )

    api = create_app(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    with TestClient(api) as client:
        yield client, fake_model, load_calls


def test_root_returns_service_information(api_client):
    client, _, _ = api_client

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "SentimenID API",
        "status": "ok",
        "docs": "/docs",
    }


def test_health_reports_loaded_model_and_version(api_client):
    client, _, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "sentimenid-test-v1",
    }


def test_predict_returns_prediction_confidence_and_probabilities(api_client):
    client, _, _ = api_client

    response = client.post(
        "/predict-teks",
        json={
            "text": "aplikasinya tidak bagus dan sering error",
            "language": "id",
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] == "negative"
    assert body["confidence"] == pytest.approx(0.85)
    assert body["model_version"] == "sentimenid-test-v1"
    assert body["probabilities"] == pytest.approx({
        "negative": 0.85,
        "neutral": 0.10,
        "positive": 0.05,
    })
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_api_passes_original_text_directly_to_saved_pipeline(api_client):
    client, fake_model, _ = api_client
    original_text = "  TIDAK   BAGUS  "

    response = client.post(
        "/predict-teks",
        json={
            "text": original_text,
            "language": "id",
        },
    )

    assert response.status_code == 200
    assert fake_model.received_batches[-1] == [original_text]


def test_model_is_loaded_only_once_during_lifespan(api_client):
    client, _, load_calls = api_client

    client.get("/health")
    client.post(
        "/predict-teks",
        json={
            "text": "pelayanannya bagus",
            "language": "id",
        },
    )
    client.post(
        "/predict-teks",
        json={
            "text": "jadwal acara hari ini",
            "language": "id",
        },
    )

    assert len(load_calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"language": "id"},
        {"text": "", "language": "id"},
        {"text": "   ", "language": "id"},
        {"text": "bagus", "language": "en"},
        {"text": "bagus"},
        {"text": "a" * 5001, "language": "id"},
        {
            "text": "bagus",
            "language": "id",
            "unexpected": "field",
        },
    ],
)
def test_invalid_requests_return_422(api_client, payload):
    client, _, _ = api_client

    response = client.post(
        "/predict-teks",
        json=payload,
    )

    assert response.status_code == 422
