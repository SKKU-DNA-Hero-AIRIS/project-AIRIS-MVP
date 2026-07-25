"""Local environment-file helpers for demo-only secrets."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def read_env_value(name: str, path: Path = ENV_PATH) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{name}="
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("\"'")
    return ""


def save_env_value(name: str, value: str, path: Path = ENV_PATH) -> None:
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("유효한 API Key를 입력하세요.")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    prefix = f"{name}="
    replacement = f"{name}={value}"
    updated = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            updated.append(replacement)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(replacement)

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)
