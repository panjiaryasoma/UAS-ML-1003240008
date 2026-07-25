from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_SLANG_MAP = {
    "enggak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
}

PLACEHOLDER_PATTERN = re.compile(r"__([a-z_]+)__", flags=re.IGNORECASE)
ELONGATED_WORD_PATTERN = re.compile(r"([a-z])\1{2,}", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


class IndonesianTextNormalizer(BaseEstimator, TransformerMixin):
    """Normalizer teks Indonesia yang stateless dan kompatibel dengan sklearn."""

    def __init__(
        self,
        slang_map: Mapping[str, str] | None = None,
        normalize_placeholders: bool = True,
        normalize_elongation: bool = True,
    ) -> None:
        self.slang_map = slang_map
        self.normalize_placeholders = normalize_placeholders
        self.normalize_elongation = normalize_elongation

    def fit(
        self,
        X: Iterable[Any],
        y: Iterable[Any] | None = None,
    ) -> "IndonesianTextNormalizer":
        """Tidak mempelajari statistik apa pun; disediakan untuk kontrak sklearn."""
        return self

    def transform(self, X: Iterable[Any]) -> list[str]:
        """Mengembalikan teks yang telah dinormalisasi tanpa mengubah input."""
        values = self._as_values(X)
        return [self._normalize_one(value) for value in values]

    def _normalize_one(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""

        text = str(value).lower()

        if self.normalize_placeholders:
            text = PLACEHOLDER_PATTERN.sub(
                lambda match: match.group(1).replace("_", " "),
                text,
            )

        if self.normalize_elongation:
            text = ELONGATED_WORD_PATTERN.sub(r"\1", text)

        slang_map = (
            DEFAULT_SLANG_MAP
            if self.slang_map is None
            else {key.lower(): replacement for key, replacement in self.slang_map.items()}
        )
        text = self._replace_slang(text, slang_map)

        return WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def _replace_slang(text: str, slang_map: Mapping[str, str]) -> str:
        if not slang_map:
            return text

        keys = sorted(slang_map, key=len, reverse=True)
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(key) for key in keys) + r")\b",
            flags=re.IGNORECASE,
        )
        return pattern.sub(
            lambda match: slang_map[match.group(0).lower()],
            text,
        )

    @staticmethod
    def _as_values(X: Iterable[Any]) -> list[Any]:
        if isinstance(X, str):
            return [X]

        if isinstance(X, pd.Series):
            return X.tolist()

        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "IndonesianTextNormalizer hanya menerima DataFrame satu kolom."
                )
            return X.iloc[:, 0].tolist()

        if getattr(X, "ndim", None) == 2:
            if X.shape[1] != 1:
                raise ValueError(
                    "IndonesianTextNormalizer hanya menerima array dua dimensi "
                    "dengan satu kolom."
                )
            return [row[0] for row in X]

        return list(X)
