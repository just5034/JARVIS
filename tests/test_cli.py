"""Tests for the JARVIS CLI entry point."""

from __future__ import annotations

import subprocess
import sys


def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "JARVIS" in result.stdout


def test_version_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_validate_configs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "validate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "validation passed" in result.stdout


def test_serve_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--load-model" in result.stdout
