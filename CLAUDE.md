# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a personal "100 Days of Data Science" learning log. Content is organized into day-range
folders (`D_1_10`, `D_11_20`, `D_21_30`, ...) each containing Jupyter notebooks for that stage's
topics (data cleaning/EDA/vis, probability & stats, regression/classification/evaluation, and more
topics planned per `README.md`). There is no installable Python package, application code, or test
suite — the deliverable is the notebooks themselves.

## Environment / tooling commands

- Dev tooling deps: `pip install -r requirements-dev.txt` (ruff, black, isort, mypy, pre-commit, nbqa,
  pytest stack, mkdocs — most are not actively wired up beyond pre-commit/ruff/mypy).
- Full (heavy, Anaconda-derived) runtime deps for running the notebooks: `pip install -r requirements.txt`.
- Lint/format: `ruff check .` and `ruff format .` (config in `pyproject.toml`, line-length 120, py310,
  numpy-style docstrings). isort/black configs also exist but ruff is the hook that actually runs.
- Type check: `mypy .` (very permissive settings — most strictness flags are off).
- Run all pre-commit hooks (isort, nbqa-isort, ruff-format, ruff --fix, mypy, basic sanity checks):
  `pre-commit run --all-files`.
- nbQA applies isort to notebook cells specifically (`nbqa-isort` hook) since ruff/black hooks target
  `types_or: [python, pyi, jupyter]` directly.
- No test suite exists yet (mypy/ruff configs already exclude `test`/`tests`/`docs` in anticipation).

## Architecture: shared code and cross-notebook wiring

**`Imports_Functions.ipynb`** (repo root) is the shared library for every day-folder notebook. It is
not imported as a Python module — subsequent notebooks pull it in via the Jupyter magic:

```python
%run ../Imports_Functions.ipynb
```

(see the first cell of `D_21_30/Regression_models.ipynb`). This executes the notebook in place, so
all its imports (`numpy`, `pandas`, `sklearn`, `scipy.stats`, etc.) and function definitions become
available in the calling notebook's namespace. When adding a new shared helper, add it to
`Imports_Functions.ipynb`, not to an individual day notebook — otherwise it won't be visible to other
notebooks that also `%run` it.

Because notebooks live one directory below the root (`D_XX_YY/notebook.ipynb`), all relative paths
inside notebooks are written relative to that subfolder, e.g. `../Imports_Functions.ipynb`,
`../D_1_10/Data/laptopData_cleaned.csv`, `../Results/regressions.pkl`. Keep this `../`-relative
convention when adding new notebooks or new data/result paths.

### Results dict pattern

Modeling notebooks (currently `D_21_30/Regression_models.ipynb`) accumulate experiment results into a
single `results: dict` keyed by a descriptive `col_name` (e.g. `'lin_reg_Price_vs_Weight'`), rather
than persisting one model per file. Key helpers in `Imports_Functions.ipynb`:

- `results_func(data, results, params)` — dispatches on `deepcopy(model).__name__` to know how to fit
  and score a model (`linregress`, `OLS`, `LinearRegression`, `PolynomialFeatures` are currently
  handled), then writes a dict of fit stats (coefficients, r², MAE/MSE/RMSE, etc.) into
  `results[col_name]`. `params` is expected to look like:
  `{'X': ..., 'y': ..., 'Model': model, 'col_name': ..., 'pol_degree': ...}`.
- `linreg(...)` — an older/simpler single-purpose version of the same idea for `scipy.stats.linregress`
  only; `results_func` is the one being extended for new model types.
- `saving_results(results)` / `loading_results()` — pickle the whole results dict to/from
  `../Results/regressions.pkl`. Notebooks typically load existing results at the top (so state persists
  across kernel restarts) and save after adding new experiments.

When adding support for a new model type, extend the `if/elif` chain in `results_func` rather than
writing a parallel one-off function, to keep all experiment results going through the same
dict-keyed/pickle-persisted store.

### Data flow

Raw and cleaned datasets live under `D_1_10/Data/` (e.g. `laptopData.csv` →
`laptopData_cleaned.csv` produced by `D_1_10/Data_Cleaning.ipynb`) and are read by later notebooks via
relative paths as described above — there is no shared data-loading abstraction beyond
`pd.read_csv(...)`.

Note: `.gitignore` blanket-ignores `*.csv`, `*.json`, and `*.pkl`, but the data/results files currently
in the repo (`D_1_10/Data/laptopData.csv`, `Results/regressions.json`, `Results/regressions.pkl`) were
force-added and are tracked. Keep that in mind when regenerating or replacing them — a plain `git add`
on a new/regenerated file matching those patterns will silently be ignored unless you `git add -f`.
