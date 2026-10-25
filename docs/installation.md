# Installation

ReactEA requires **Python 3.11 or newer** (3.11 to 3.14 are supported).
Its dependencies (RDKit, jMetalPy, NumPy, pandas, ChEMBL Structure Pipeline, ...) are installed automatically.

## From PyPI

```bash
pip install reactea
```

or add it to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add reactea
```

## From source

The project is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/BioSystemsUM/ReactEA.git
cd ReactEA
uv sync
```

`uv sync` creates a virtual environment (`.venv`) with ReactEA and the pinned dependencies from `uv.lock`, and
downloads a compatible Python if needed (choose one with, e.g., `uv sync --python 3.13`).
Without uv, `pip install .` also works.

## Optional dependencies

Some built-in evaluation functions need tools that are not installed with ReactEA:

| Evaluation function | Requires |
|---|---|
| `Docking` | [DOCKSTRING](https://github.com/dockstring/dockstring) |
| `SweetnessPredictionDeepSweet`, `PenalizedSweetness` | [DeepSweet](https://github.com/BioSystemsUM/DeepSweet) |

## Development

`uv sync` also installs the development tools (pytest, flake8, mypy).

```bash
uv run pytest            # run the tests
uv run flake8 src tests  # lint
```

To build this documentation:

```bash
uv sync --group docs
uv run sphinx-build docs docs/_build/html
```