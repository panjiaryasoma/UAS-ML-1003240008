from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_TEXT_LENGTH = 5000


class Language(str, Enum):
    ID = "id"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Teks Bahasa Indonesia yang akan diklasifikasikan.",
        examples=["aplikasinya tidak bagus dan sering error"],
    )
    language: Language = Field(
        ...,
        description="Bahasa input. Versi API ini hanya mendukung Bahasa Indonesia.",
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text tidak boleh kosong atau hanya berisi whitespace")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None


class RootResponse(BaseModel):
    service: str
    status: Literal["ok"]
    docs: str
