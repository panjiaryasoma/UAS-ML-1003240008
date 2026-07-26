from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np

from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)


LOGGER = logging.getLogger("sentimenid.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


def _path_from_environment(
    variable_name: str,
    default: Path,
) -> Path:
    configured = os.getenv(variable_name)
    return Path(configured) if configured else default


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata model tidak ditemukan: {metadata_path}"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Metadata model bukan JSON yang valid: {metadata_path}"
        ) from exc

    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError(
            "metadata.json harus memiliki model_version berupa string."
        )

    return metadata


def _validate_loaded_model(
    model: Any,
    metadata: dict[str, Any],
) -> list[str]:
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Artefak model tidak memiliki method predict().")

    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(
            "Artefak model tidak memiliki predict_proba(). "
            "Gunakan classifier yang sudah dikalibrasi."
        )

    classes = [
        str(label)
        for label in getattr(model, "classes_", [])
    ]
    if not classes:
        raise ValueError("Artefak model tidak memiliki classes_.")

    metadata_classes = (
        metadata
        .get("model_selection", {})
        .get("classes", [])
    )
    if metadata_classes and classes != [
        str(label) for label in metadata_classes
    ]:
        raise ValueError(
            "Urutan kelas pada model tidak cocok dengan metadata."
        )

    return classes


def create_app(
    *,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> FastAPI:
    resolved_model_path = Path(
        model_path
        or _path_from_environment(
            "SENTIMENID_MODEL_PATH",
            DEFAULT_MODEL_PATH,
        )
    )
    resolved_metadata_path = Path(
        metadata_path
        or _path_from_environment(
            "SENTIMENID_METADATA_PATH",
            DEFAULT_METADATA_PATH,
        )
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {resolved_model_path}. "
                "Jalankan `python -m src.train` terlebih dahulu."
            )

        model = joblib.load(resolved_model_path)
        metadata = _load_metadata(resolved_metadata_path)
        classes = _validate_loaded_model(model, metadata)

        application.state.model = model
        application.state.metadata = metadata
        application.state.classes = classes
        application.state.model_loaded = True

        LOGGER.info(
            "model_loaded version=%s classes=%s",
            metadata["model_version"],
            ",".join(classes),
        )

        try:
            yield
        finally:
            application.state.model = None
            application.state.metadata = None
            application.state.classes = []
            application.state.model_loaded = False

    application = FastAPI(
        title="SentimenID API",
        version="1.0.0",
        description=(
            "REST API klasifikasi sentimen teks Bahasa Indonesia "
            "menggunakan pipeline machine learning yang telah disimpan."
        ),
        lifespan=lifespan,
    )

    @application.get(
        "/",
        response_model=RootResponse,
        tags=["system"],
    )
    def root() -> RootResponse:
        return RootResponse(
            service="SentimenID API",
            status="ok",
            docs="/docs",
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health(request: Request) -> HealthResponse:
        loaded = bool(
            getattr(request.app.state, "model_loaded", False)
        )
        metadata = getattr(
            request.app.state,
            "metadata",
            None,
        )
        model_version = (
            metadata.get("model_version")
            if isinstance(metadata, dict)
            else None
        )

        if not loaded:
            raise HTTPException(
                status_code=503,
                detail="Model belum dimuat.",
            )

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=model_version,
        )

    @application.post(
        "/predict-teks",
        response_model=PredictionResponse,
        tags=["prediction"],
    )
    def predict_text(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        start_time = perf_counter()

        model = getattr(request.app.state, "model", None)
        metadata = getattr(request.app.state, "metadata", None)
        classes = getattr(request.app.state, "classes", [])

        if (
            model is None
            or not isinstance(metadata, dict)
            or not classes
        ):
            raise HTTPException(
                status_code=503,
                detail="Model belum siap digunakan.",
            )

        try:
            # Tidak ada preprocessing manual di API. Teks mentah diteruskan
            # langsung ke pipeline yang sudah menyimpan normalizer dan TF-IDF.
            prediction = str(model.predict([payload.text])[0])
            probability_row = np.asarray(
                model.predict_proba([payload.text])[0],
                dtype=float,
            )
        except Exception as exc:
            LOGGER.exception(
                "prediction_failed text_length=%d",
                len(payload.text),
            )
            raise HTTPException(
                status_code=500,
                detail="Prediksi gagal diproses.",
            ) from exc

        if probability_row.shape != (len(classes),):
            raise HTTPException(
                status_code=500,
                detail="Bentuk probabilitas model tidak valid.",
            )

        if not np.isfinite(probability_row).all():
            raise HTTPException(
                status_code=500,
                detail="Model menghasilkan probabilitas tidak valid.",
            )

        if prediction not in classes:
            raise HTTPException(
                status_code=500,
                detail="Label prediksi tidak tersedia dalam classes_.",
            )

        probabilities = {
            label: float(probability_row[index])
            for index, label in enumerate(classes)
        }
        confidence = probabilities[prediction]

        latency_ms = (perf_counter() - start_time) * 1000
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        LOGGER.info(
            "prediction_completed "
            "timestamp_utc=%s "
            "latency_ms=%.3f "
            "text_length=%d "
            "language=%s "
            "prediction=%s "
            "confidence=%.6f "
            "model_version=%s",
            timestamp_utc,
            latency_ms,
            len(payload.text),
            payload.language.value,
            prediction,
            confidence,
            metadata["model_version"],
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            model_version=metadata["model_version"],
        )

    return application


app = create_app()