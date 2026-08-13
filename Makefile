# micro-lab — common developer tasks. Run `make help` for the list.
# Assumes the project venv is active (see docs/ENV.md): `pip install -e ".[dev]"`.

.PHONY: help install lint format test coverage verify data gate studies figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create/refresh the editable install with dev extras
	pip install -e ".[dev]"

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Apply ruff formatting
	ruff format .

test:  ## Run the CI test subset (no raw LOBSTER data needed)
	pytest -m "not slow"

coverage:  ## CI test subset with coverage + the >=90% gate
	coverage run -m pytest -m "not slow"
	coverage report --fail-under=90

verify:  ## Full CI verification: ruff + format-check + tests + coverage gate
	ruff check .
	ruff format --check .
	coverage run -m pytest -m "not slow"
	coverage report --fail-under=90

data:  ## Download the LOBSTER sample (not committed; needed for the slow tests + studies)
	bash scripts/get_data.sh

gate:  ## Print the  machinery-gate evidence report (needs data)
	python -m src.studies.g1_gate

studies:  ## Rerun the registered study family -> data/fixtures/*.csv (needs data)
	python -m src.studies.runner

figures:  ## Regenerate all showcase figures into docs/figures/ (from committed fixtures)
	python scripts/make_figures.py

clean:  ## Remove caches and build artifacts (keeps fixtures + figures)
	rm -rf .pytest_cache .ruff_cache .hypothesis .coverage coverage.xml \
	       htmlcov *.egg-info src/*.egg-info
