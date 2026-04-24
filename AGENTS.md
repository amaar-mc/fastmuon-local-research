# Repository Guidelines

## Project Structure & Module Organization

This repository is a local PyTorch research harness for Muon-family optimizer experiments. `optim_lab/optimizers` contains the optimizer factory, Muon variants, Newton-Schulz utilities, and matrix/fallback parameter routing. `optim_lab/models` holds the TinyGPT probe model, and `optim_lab/data` provides deterministic token streams plus optional byte-level local text loading. Experiment entrypoints live in `optim_lab/experiments`; result summarizers and plotting utilities live in `optim_lab/analysis`. Reproducible experiment definitions are JSON files under `configs`, and human-readable findings belong in `reports` or `RESEARCH_LOG.md`.

Generated run directories such as `results/`, `results_fast/`, and `results_probe/` are intentionally ignored. Commit configs, reports, and code; do not commit transient metrics unless they have been distilled into a report.

## Build, Test, and Development Commands

Run the full test suite:

```bash
python3 -m pytest
```

Run the main reduced Newton-Schulz probe:

```bash
python3 -m optim_lab.experiments.sweep --config configs/fast_muon_probe.json
```

Summarize a completed run:

```bash
python3 -m optim_lab.analysis.summarize results/fast_muon_probe
```

Plot validation curves:

```bash
python3 -m optim_lab.analysis.plot_results results/fast_muon_probe --output results/fast_muon_probe/loss_curves.png
```

## Testing Guidelines

Tests are written with `pytest` and configured in `pyproject.toml`. Current coverage checks Newton-Schulz behavior, deterministic data generation, TinyGPT forward/loss behavior, optimizer parameter routing, and Pulse/Cached Pulse diagnostics. Add focused tests when changing optimizer routing, state initialization, or metric logging.

## Commit & PR Guidelines

There was no prior git history before this repository was initialized. Use concise research-style commit subjects that describe the unit of work, for example `Add reduced-depth Muon probe config` or `Document FastMuon first-cycle findings`. Keep generated experiment outputs out of commits unless a reviewer explicitly asks for raw artifacts.

