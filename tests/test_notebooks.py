"""Smoke checks that run in CI without data, GPUs or re-training.

* every notebook is valid nbformat v4;
* the headline numbers quoted in README.md are present in the saved
  notebook outputs (so the README cannot drift from the evidence);
* every image referenced from README.md exists in the repository.
"""
import pathlib
import re

import nbformat
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p for p in ROOT.rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts and ".venv" not in p.parts
)

# (notebook path relative to the repo root, text that must appear in its saved outputs)
EXPECTED_OUTPUTS = [('KNN.ipynb', 'Training data shape: (800, 10)'),
 ('KNN.ipynb', 'Accuracy of KNN model: 34.00%'),
 ('KNN.ipynb', 'Accuracy of KNN model (scaled): 50.00%'),
 ('KNN.ipynb', 'Accuracy of Best KNN model: 52.50%'),
 ('KNN.ipynb',
  "Best Hyperparameters: {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}"),
 ('KNN.ipynb', 'Accuracy of Best KNN model: 59.00000000000001%'),
 ('KNN.ipynb', 'Accuracy: 0.5'),
 ('KNN.ipynb', '   macro avg       0.53      0.53      0.52       200'),
 ('CNN.ipynb', 'Test accuracy: 0.5350')]


def _outputs_text(path: pathlib.Path) -> str:
    nb = nbformat.read(path, as_version=4)
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if "text" in out:
                chunks.append(out["text"])
            data = out.get("data", {})
            if "text/plain" in data:
                chunks.append(data["text/plain"])
    return "\n".join(chunks)


def test_notebooks_found():
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_is_valid_nbformat(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("notebook,snippet", EXPECTED_OUTPUTS)
def test_reported_result_is_in_notebook_output(notebook, snippet):
    assert snippet in _outputs_text(ROOT / notebook), f"{snippet!r} not found in outputs of {notebook}"


def test_readme_images_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    refs = re.findall(r"\]\((docs/images/[^)\s]+)\)", readme) + re.findall(r'src="(docs/images/[^"]+)"', readme)
    assert refs, "README.md should show at least one image from docs/images/"
    missing = [r for r in refs if not (ROOT / r).is_file()]
    assert not missing, f"missing images: {missing}"
