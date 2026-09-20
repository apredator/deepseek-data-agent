"""
Optional semantic tooling (Hugging Face / embeddings flavour).

Adds a genuinely ML-flavoured capability on top of the deterministic payroll
checks: fuzzy/semantic similarity over *text* fields (names, positions) that do
not match verbatim - exactly the use-case described in the specification
(section 5.1, "Semantic-поиск/дедуп").

Two backends, chosen automatically:

* ``sentence-transformers`` embeddings when the package is installed
  (``pip install -r requirements-optional.txt``), and
* a dependency-free :mod:`difflib` sequence-similarity fallback otherwise.

Enabled with ``ENABLE_SEMANTIC_TOOLS=true`` so the core demo keeps a tiny
dependency footprint.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from sap_hr_data import SAPHRDataset

logger = logging.getLogger(__name__)

SEMANTIC_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "find_similar_employee_names",
        "description": (
            "Find employee name/position records that are suspiciously similar to each "
            "other but not identical (e.g. duplicated master-data records written with "
            "slight spelling differences). Uses embeddings when available, otherwise "
            "fuzzy string similarity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold between 0 and 1 (default 0.9).",
                },
                "column": {
                    "type": "string",
                    "enum": ["Nachn", "Vorna", "Position"],
                    "description": "Which text field to compare (default 'Nachn').",
                },
            },
            "required": [],
        },
    },
]


def _embedding_backend_available() -> bool:
    try:  # pragma: no cover - depends on optional dependency
        import sentence_transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class SemanticToolbox:
    """Semantic similarity helpers over the dataset's text columns."""

    def __init__(self, dataset: SAPHRDataset, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dataset = dataset
        self.model_name = model_name
        self._model = None
        self._using_embeddings = False

    # -- backend ----------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None or not _embedding_backend_available():
            return
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._using_embeddings = True
            logger.info("Semantic tools using sentence-transformers model %s", self.model_name)
        except Exception as error:  # noqa: BLE001
            logger.warning("sentence-transformers unavailable (%s); using difflib fallback", error)
            self._model = None

    # -- similarity -------------------------------------------------------
    @staticmethod
    def _string_similarity(left: str, right: str) -> float:
        return SequenceMatcher(None, left.lower(), right.lower()).ratio()

    def _pair_similarity(self, left: str, right: str) -> float:
        self._ensure_model()
        if self._using_embeddings and self._model is not None:  # pragma: no cover
            vectors = self._model.encode([left, right], normalize_embeddings=True)
            return float(vectors[0] @ vectors[1])
        return self._string_similarity(left, right)

    # -- tool -------------------------------------------------------------
    def find_similar_employee_names(
        self,
        threshold: float = 0.9,
        column: str = "Nachn",
        max_pairs: int = 25,
    ) -> Dict[str, Any]:
        """Return pairs of employees whose ``column`` value is highly similar but not equal."""
        frame = self.dataset.pa0001
        if column not in frame.columns:
            return {
                "tool": "find_similar_employee_names", "ok": False, "count": 0,
                "records": [], "truncated": False,
                "summary": f"Column '{column}' is not available in PA0001_org.",
                "error": "unknown_column",
            }

        unique = (
            frame[["Pernr", column, "Orgeh", "Position"]]
            .dropna(subset=[column])
            .drop_duplicates(subset=[column])
            .sort_values(column)
            .to_dict(orient="records")
        )

        pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for index, left in enumerate(unique):
            for right in unique[index + 1:]:
                score = self._pair_similarity(str(left[column]), str(right[column]))
                if threshold <= score < 1.0:
                    pairs.append((score, left, right))

        pairs.sort(key=lambda item: -item[0])
        records = [
            {
                "similarity": round(score, 4),
                "column": column,
                "Pernr_a": left["Pernr"],
                "value_a": left[column],
                "org_unit_a": left.get("Orgeh"),
                "Pernr_b": right["Pernr"],
                "value_b": right[column],
                "org_unit_b": right.get("Orgeh"),
            }
            for score, left, right in pairs[:max_pairs]
        ]

        backend = "sentence-transformers" if self._using_embeddings else "difflib"
        summary = (
            f"Found {len(records)} pair(s) of PA0001 '{column}' values with similarity "
            f">= {threshold} ({backend} backend)."
        )
        return {
            "tool": "find_similar_employee_names", "ok": True,
            "count": len(records), "records": records,
            "truncated": len(pairs) > max_pairs, "summary": summary,
            "backend": backend,
        }


def build_semantic_registry_extras(dataset: SAPHRDataset) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (extra tool schemas, extra bound functions) to merge into a registry."""
    toolbox = SemanticToolbox(dataset)
    return SEMANTIC_TOOL_SPECS, {
        "find_similar_employee_names": toolbox.find_similar_employee_names,
    }


__all__ = ["SEMANTIC_TOOL_SPECS", "SemanticToolbox", "build_semantic_registry_extras"]
