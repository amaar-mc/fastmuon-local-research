# FastMuon Local Research Lab

This repository is a small, reproducible PyTorch lab for testing Muon-family optimizers on local hardware. It was built to investigate a narrow question:

> Does Muon need five Newton-Schulz iterations every step, or can partial orthogonalization preserve most of the useful geometry at lower optimizer cost?

The first empirical signal is that **reduced-depth Muon**, provisionally called **FastMuon-NS3/NS4**, preserved most of Muon's small-language-model loss improvement in a controlled TinyGPT probe while using fewer Newton-Schulz iterations.

This is not a broad optimizer breakthrough claim. It is a local research finding with code, configs, tests, and a reproducible first experiment.

## Optimizer Idea

Muon updates hidden 2D weight matrices by applying momentum and then approximately orthogonalizing the matrix update:

$$
M_t = \mu M_{t-1} + G_t
$$

$$
\Delta W_t = -\eta\,\operatorname{NS}_k(M_t)
$$

where $\operatorname{NS}_k$ is $k$ steps of the Newton-Schulz polar-factor iteration. Standard Muon commonly uses $k=5$.

The FastMuon hypothesis tested here is simple:

$$
k \in \{3,4\}
$$

may be enough during early small-LLM training. In that regime, a moderate orthogonality error may still deliver useful spectral balancing, while avoiding part of the optimizer-side matrix multiplication cost.

The repository also includes two exploratory gating variants:

$$
\rho_t =
\operatorname{clip}\left(
\sigma(a_c(c_t-b_c))\,
\sigma(a_s(s_t-b_s)),
\rho_{\min},
\rho_{\max}
\right)
$$

$$
\Delta W_t =
-\eta\,
\operatorname{RMSNorm}\left((1-\rho_t)N_t + \rho_t O_t\right)
$$

where $N_t$ is RMS-normalized momentum, $O_t$ is the Newton-Schulz orthogonalized direction, $c_t$ is fast/slow momentum coherence, and $s_t$ is matrix energy anisotropy. These variants were useful negative controls, but they are not the lead result.

## What Was Observed

Clean run: `results_fast/fast_muon_probe`

Probe:

- Model: TinyGPT, 2 layers, width 64, 2 heads, context 32
- Data: deterministic synthetic token stream
- Budget: 100 steps, 3 seeds, 51,200 tokens per run
- Device: Apple MPS
- PyTorch: 2.6.0

| Setting | Median validation loss | Seed losses | Median step seconds | Median NS error |
|---|---:|---|---:|---:|
| AdamW, lr=0.001 | 4.71216 | 4.7094, 4.7122, 4.7151 | 0.01569 | NA |
| FastMuon-NS3, matrix lr=0.02 | 4.67666 | 4.6842, 4.6744, 4.6767 | 0.13453 | 0.06817 |
| FastMuon-NS4, matrix lr=0.02 | 4.66698 | 4.6705, 4.6670, 4.6664 | 0.14735 | 0.05333 |
| Muon-NS5, matrix lr=0.02 | 4.65868 | 4.6657, 4.6559, 4.6587 | 0.15923 | 0.04756 |
| cached-pulse Muon, matrix lr=0.02 | 4.71200 | 4.7120, 4.7185, 4.6923 | 0.14674 | 0.05148 |

Interpretation:

- FastMuon-NS3 and FastMuon-NS4 both beat AdamW on fixed-token validation loss in this probe.
- NS4 kept most of Muon-NS5's improvement while using fewer Newton-Schulz iterations.
- NS3 gave the sharper efficiency tradeoff: lower optimizer depth, still better loss than AdamW, but a larger gap to NS5.
- Pulse/cached-pulse variants reduced Newton-Schulz call frequency but lost too much quality to be the lead candidate.
- AdamW remains far faster wall-clock on this tiny local MPS setup. The current claim is **sample-efficiency and Muon-internal optimizer efficiency**, not end-to-end wall-clock superiority over AdamW.

See [reports/fast_muon_probe.md](reports/fast_muon_probe.md) for the first-cycle report.

## Repository Layout

- `optim_lab/optimizers`: AdamW/SGD factory, Muon, Coherence-Interpolated Muon, Pulse Muon, Cached Pulse Muon, and Newton-Schulz utilities.
- `optim_lab/models`: configurable TinyGPT with separate Q/K/V projections.
- `optim_lab/data`: deterministic synthetic token streams and optional byte-level local text loading.
- `optim_lab/experiments`: training, sweep, and matrix microbenchmark entrypoints.
- `optim_lab/analysis`: summary tables and plotting utilities.
- `configs`: reproducible experiment configs.
- `tests`: correctness tests for optimizer routing, Newton-Schulz behavior, data, and model execution.
- `reports`: human-readable research notes from completed experiment cycles.

## Reproduce The Main Probe

Run tests:

```bash
python3 -m pytest
```

Run the reduced Newton-Schulz sweep:

```bash
python3 -m optim_lab.experiments.sweep --config configs/fast_muon_probe.json
```

Summarize results:

```bash
python3 -m optim_lab.analysis.summarize results/fast_muon_probe
```

Plot validation curves:

```bash
python3 -m optim_lab.analysis.plot_results \
  results/fast_muon_probe \
  --output results/fast_muon_probe/loss_curves.png
```

Run the broader local cycle:

```bash
python3 -m optim_lab.experiments.sweep --config configs/local_llm_cycle.json
```

Run matrix microbenchmarks:

```bash
python3 -m optim_lab.experiments.microbench --config configs/microbench.json
python3 -m optim_lab.analysis.summarize_microbench results/microbench
```

## Research Caveats

The result is intentionally scoped. It has not yet been validated on a real text corpus, longer schedules, larger models, CUDA GPUs, or distributed training. A publication-grade claim would need longer runs, stronger AdamW and Muon tuning, real-data replication, and confidence intervals over more seeds.

The next serious experiment is a 500-step TinyGPT sweep over AdamW, Muon-NS3, Muon-NS4, and Muon-NS5, followed by the same comparison on a byte-level text corpus.

