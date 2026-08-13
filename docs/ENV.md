# Environment

Build/development environment for micro-lab.

## Development machine (this build)
- **OS:** macOS 26.5 (Apple Silicon, arm64)
- **Python:** 3.12.13 (project venv at `.venv/`)
- **git:** 2.50.1
- **gh:** 2.93.0

## CI
- **Runner:** ubuntu-latest (GitHub Actions), Python 3.12
- Runs `ruff check`, `ruff format --check`, and `pytest -m "not slow"` with
  coverage, gated at **>= 90%** line coverage on `src/`.
- CI executes only on **committed derived fixtures** (`data/fixtures/`). Raw
  LOBSTER files are gitignored and never downloaded in CI.

## Pinned dependencies
```
numpy==2.2.1
scipy==1.15.1
pandas==2.2.3
statsmodels==0.14.4
scikit-learn==1.6.1
matplotlib==3.10.0
hypothesis==6.123.2
pytest==8.3.4
coverage==7.6.10
ruff==0.9.2
```

## Reproduce the environment
```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Get the data (not committed)
```bash
bash scripts/get_data.sh          # all five tickers, level 10
```
See `docs/DESIGN.md` for the data-source provenance note and `README.md` for
LOBSTER attribution.
