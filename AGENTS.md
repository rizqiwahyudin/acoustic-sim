# AGENTS.md

## Cursor Cloud specific instructions

This is a Python scientific simulation project (DOA array geometry comparison) with no web server, database, or external services. It runs entirely offline.

### Dependencies

Three pip packages are required: `numpy`, `matplotlib`, `pyroomacoustics`. The system also needs `python3-dev` (C headers) for building `pyroomacoustics` from source.

### Running the application

- **Quick validation (~20s):** `python run_comparison.py --test`
- **Regenerate plots from existing CSV:** `python run_comparison.py --plot-only`
- **Full parameter sweep (can take hours):** `python run_comparison.py`
- **Smoke test:** Run `doa_smoke_test.py` inline with `matplotlib.use('Agg')` set before import, since `plt.show()` requires a display. Alternatively, set `MPLBACKEND=Agg` env var.

### Key gotchas

- `doa_smoke_test.py` calls `plt.show()` which blocks in headless environments. Use `MPLBACKEND=Agg` or patch the backend before running.
- `pyroomacoustics` requires C++ compilation (`python3-dev` and `g++` must be installed). The wheel is cached after first install.
- Results are written to `results/` directory; the `--test` and full modes share the same `metrics.csv` file. Back up before re-running if you need to preserve existing data.
- No linter, formatter, or test framework is configured in this repo.
