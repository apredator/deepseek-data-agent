"""
Central configuration for the HR/payroll AI agent.

All tunables (paths, anomaly thresholds, LLM provider, guardrail limits) are read
from environment variables / ``.env`` instead of being hard-coded, as required by
the project specification (section 6, "Конфигурация").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# ``override=False`` keeps real environment variables (CI, Docker, shell) winning
# over values coming from a local ``.env`` file.
load_dotenv(BASE_DIR / ".env", override=False)

import os  # noqa: E402  (imported after load_dotenv on purpose)

#: Default dataset year used by the synthetic data generator.
DATASET_YEAR = 2024

#: Models known to the DeepSeek public API + the internal naming used by DSH.
KNOWN_DEEPSEEK_MODELS: List[str] = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the HR/payroll agent."""

    # --- paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    output_dir: Path = BASE_DIR / "outputs"
    log_dir: Path = BASE_DIR / "logs"

    # --- logging ---------------------------------------------------------
    log_level: str = "INFO"
    debug: bool = False

    # --- LLM -------------------------------------------------------------
    llm_provider: str = "deepseek"  # deepseek | hf | mock
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    request_timeout_seconds: int = 60

    hf_api_token: str = ""
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_api_base: str = "https://router.huggingface.co/v1"
    enable_hf_fallback: bool = False

    # --- agent loop / guardrails ----------------------------------------
    max_tool_iterations: int = 5
    allow_react_fallback: bool = True
    read_only: bool = True
    enable_semantic_tools: bool = False

    # --- anomaly thresholds ---------------------------------------------
    salary_jump_threshold_pct: float = 40.0

    # --- synthetic dataset ----------------------------------------------
    dataset_dir_name: str = "data"
    num_employees: int = 600
    num_periods: int = 12
    closed_periods: int = 9
    seed: int = 42
    injection_probe: bool = True

    # --- Hugging Face Datasets Hub (optional) ---------------------------
    hf_dataset_repo: str = ""

    def ensure_dirs(self) -> None:
        """Create the working directories if they do not exist yet."""
        for path in (self.data_dir, self.output_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current environment."""
        base_dir = Path(_env_str("AGENT_BASE_DIR", str(BASE_DIR))).resolve()
        data_dir = Path(_env_str("DATA_DIR", str(base_dir / "data"))).resolve()
        output_dir = Path(_env_str("OUTPUT_DIR", str(base_dir / "outputs"))).resolve()
        log_dir = Path(_env_str("LOG_DIR", str(base_dir / "logs"))).resolve()

        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            log_dir=log_dir,
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
            debug=_env_bool("DEBUG", False),
            llm_provider=_env_str("LLM_PROVIDER", "deepseek").lower(),
            deepseek_api_key=_env_str("DEEPSEEK_API_KEY", ""),
            deepseek_model=_env_str("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_api_base=_env_str("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip("/"),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 60),
            hf_api_token=_env_str("HF_TOKEN", _env_str("HUGGINGFACE_API_TOKEN", "")),
            hf_model=_env_str("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            hf_api_base=_env_str("HF_API_BASE", "https://router.huggingface.co/v1").rstrip("/"),
            enable_hf_fallback=_env_bool("ENABLE_HF_FALLBACK", False),
            max_tool_iterations=max(1, _env_int("MAX_TOOL_ITERATIONS", 5)),
            allow_react_fallback=_env_bool("ALLOW_REACT_FALLBACK", True),
            read_only=_env_bool("AGENT_READ_ONLY", True),
            enable_semantic_tools=_env_bool("ENABLE_SEMANTIC_TOOLS", False),
            salary_jump_threshold_pct=_env_float("SALARY_JUMP_THRESHOLD_PCT", 40.0),
            num_employees=max(10, _env_int("SYNTHETIC_NUM_EMPLOYEES", 600)),
            num_periods=max(2, _env_int("SYNTHETIC_NUM_PERIODS", 12)),
            closed_periods=max(0, _env_int("SYNTHETIC_CLOSED_PERIODS", 9)),
            seed=_env_int("SYNTHETIC_SEED", 42),
            injection_probe=_env_bool("SYNTHETIC_INJECTION_PROBE", True),
            hf_dataset_repo=_env_str("HF_DATASET_REPO", ""),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings.from_env()


def reload_settings() -> Settings:
    """Clear the settings cache and rebuild from the environment (used in tests)."""
    get_settings.cache_clear()
    return get_settings()


def configure_logging(level: str | None = None, log_to_file: bool = False) -> None:
    """Configure root logging once, using ``LOG_LEVEL`` from settings by default."""
    settings = get_settings()
    resolved = (level or settings.log_level or "INFO").upper()

    root = logging.getLogger()
    if not root.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        if log_to_file:
            try:
                settings.ensure_dirs()
                file_handler = logging.FileHandler(settings.log_dir / "hr_agent.log", encoding="utf-8")
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError:  # pragma: no cover - filesystem edge case
                pass

    root.setLevel(getattr(logging, resolved, logging.INFO))


__all__ = [
    "BASE_DIR",
    "DATASET_YEAR",
    "KNOWN_DEEPSEEK_MODELS",
    "Settings",
    "get_settings",
    "reload_settings",
    "configure_logging",
]
