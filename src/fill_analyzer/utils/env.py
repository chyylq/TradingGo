"""
Environment and configuration helpers for LLM credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def load_openai_credentials(config_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load OpenAI credential settings from environment variables or a config file.

    The function first checks the current process environment for the keys:
        - OPENAI_API_KEY (required)
        - OPENAI_API_BASE (optional)
        - OPENAI_MODEL (optional)

    If values are missing and `config_path` is provided, a simple key=value file
    is parsed to populate the same keys. The file format matches the template
    provided in `configs/openai_env.example`.
    """

    credentials: Dict[str, str] = {}
    for key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL"):
        value = os.getenv(key)
        if value:
            credentials[key] = value

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" not in stripped:
                        continue
                    key, _, raw_value = stripped.partition("=")
                    key = key.strip()
                    value = raw_value.strip()
                    if key and value and key not in credentials:
                        credentials[key] = value

    if "OPENAI_API_KEY" not in credentials:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in the environment "
            "or provide a config file with the key."
        )

    return credentials

