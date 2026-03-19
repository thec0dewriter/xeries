"""Tests for example notebooks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def get_notebook_files() -> list[Path]:
    """Get all notebook files in the examples directory."""
    if not EXAMPLES_DIR.exists():
        return []
    return list(EXAMPLES_DIR.glob("*.ipynb"))


@pytest.fixture
def notebook_executor():
    """Fixture to execute notebooks."""

    def _execute(notebook_path: Path, timeout: int = 300) -> tuple[bool, str]:
        """Execute a notebook and return success status and output."""
        cmd = [
            sys.executable,
            "-m",
            "nbclient",
            str(notebook_path),
            "--execute",
            "--timeout",
            str(timeout),
            "--allow-errors",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output

    return _execute


@pytest.mark.notebook
@pytest.mark.slow
class TestNotebooks:
    """Test suite for example notebooks."""

    @pytest.mark.parametrize(
        "notebook_name",
        ["01_quickstart.ipynb"],
    )
    def test_quickstart_notebook(
        self,
        notebook_name: str,
        notebook_executor,
    ) -> None:
        """Test that the quickstart notebook executes without errors."""
        notebook_path = EXAMPLES_DIR / notebook_name
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        success, output = notebook_executor(notebook_path)

        if not success:
            pytest.fail(f"Notebook execution failed:\n{output}")

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "notebook_name",
        ["02_skforecast_integration.ipynb"],
    )
    def test_skforecast_notebook(
        self,
        notebook_name: str,
        notebook_executor,
    ) -> None:
        """Test skforecast integration notebook (requires skforecast)."""
        pytest.importorskip("skforecast")

        notebook_path = EXAMPLES_DIR / notebook_name
        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        success, output = notebook_executor(notebook_path)

        if not success:
            pytest.fail(f"Notebook execution failed:\n{output}")


@pytest.mark.notebook
def test_notebooks_are_valid_json() -> None:
    """Test that all notebooks are valid JSON."""
    import json

    notebooks = get_notebook_files()
    if not notebooks:
        pytest.skip("No notebooks found")

    for nb_path in notebooks:
        try:
            with open(nb_path, encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {nb_path.name}: {e}")


@pytest.mark.notebook
def test_notebooks_have_valid_structure() -> None:
    """Test that all notebooks have valid nbformat structure."""
    import json

    notebooks = get_notebook_files()
    if not notebooks:
        pytest.skip("No notebooks found")

    for nb_path in notebooks:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb, f"Missing 'cells' in {nb_path.name}"
        assert "metadata" in nb, f"Missing 'metadata' in {nb_path.name}"
        assert "nbformat" in nb, f"Missing 'nbformat' in {nb_path.name}"
        assert nb["nbformat"] >= 4, f"nbformat too old in {nb_path.name}"
