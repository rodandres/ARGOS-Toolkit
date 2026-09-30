from pathlib import Path
import ast
import py_compile
import runpy

import nbformat
from nbclient import NotebookClient


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[1]

EXAMPLES_DIR = PROJECT_ROOT / "examples"
PY_EXAMPLES_DIR = EXAMPLES_DIR / "py"
NOTEBOOK_EXAMPLES_DIR = EXAMPLES_DIR / "ipynb"


# ---------------------------------------------------------------------------
# Python examples
# ---------------------------------------------------------------------------


def test_python_examples_compile():
    """Check that all Python examples are syntactically valid."""
    paths = list(PY_EXAMPLES_DIR.rglob("*.py"))

    assert paths, f"No Python examples found in {PY_EXAMPLES_DIR}"

    for path in paths:
        py_compile.compile(
            path,
            doraise=True,
        )


def test_python_examples_execute():
    """Execute all Python examples and verify that they complete successfully."""
    paths = list(PY_EXAMPLES_DIR.rglob("*.py"))

    assert paths, f"No Python examples found in {PY_EXAMPLES_DIR}"

    for path in paths:
        runpy.run_path(
            str(path),
            run_name="__main__",
        )


# ---------------------------------------------------------------------------
# Jupyter notebooks
# ---------------------------------------------------------------------------


def test_notebook_examples_compile():
    """Check that all notebook code cells contain valid Python syntax."""
    paths = list(NOTEBOOK_EXAMPLES_DIR.rglob("*.ipynb"))

    assert paths, f"No Jupyter notebooks found in {NOTEBOOK_EXAMPLES_DIR}"

    for path in paths:
        notebook = nbformat.read(
            path,
            as_version=4,
        )

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue

            ast.parse(
                cell.source,
                filename=str(path),
            )


def test_notebook_examples_execute():
    """Execute all Jupyter notebooks and verify that they complete successfully."""
    paths = list(NOTEBOOK_EXAMPLES_DIR.rglob("*.ipynb"))

    assert paths, f"No Jupyter notebooks found in {NOTEBOOK_EXAMPLES_DIR}"

    for path in paths:
        notebook = nbformat.read(
            path,
            as_version=4,
        )

        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
        )

        client.execute()