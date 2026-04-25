# FastMuon Probe Report

Date: 2026-04-24

## Research Question

Muon's practical optimizer cost is dominated by repeated Newton-Schulz matrix multiplications on hidden 2D parameters. The standard implementation uses five iterations:

$$
\Delta W_t = -\eta\,\mathrm{NS}_5(M_t)
$$

This probe asks whether a lower-depth approximation,

$$
\Delta W_t = -\eta\,\mathrm{NS}_k(M_t), \quad k < 5,
$$

retains most of Muon's fixed-token training benefit.

## Experimental Setup

- Model: TinyGPT, 2 layers, 64 embedding width, 2 attention heads
- Context length: 32 tokens
- Parameters: 118,528
- Data: deterministic synthetic next-token stream
- Training budget: 100 steps, batch size 16
- Total tokens per run: 51,200
- Seeds: 0, 1, 2
- Device: Apple MPS
- PyTorch: 2.6.0

The command used for the clean reproduction run was:

```bash
python3 -m optim_lab.experiments.sweep \
  --config configs/fast_muon_probe.json \
  --output-root results_fast
```

## Results

| Setting | Median validation loss | Seed losses | Median step seconds | Median NS error |
|---|---:|---|---:|---:|
| AdamW, lr=0.001 | 4.71216 | 4.7094, 4.7122, 4.7151 | 0.01569 | NA |
| FastMuon-NS3, matrix lr=0.02 | 4.67666 | 4.6842, 4.6744, 4.6767 | 0.13453 | 0.06817 |
| FastMuon-NS4, matrix lr=0.02 | 4.66698 | 4.6705, 4.6670, 4.6664 | 0.14735 | 0.05333 |
| Muon-NS5, matrix lr=0.02 | 4.65868 | 4.6657, 4.6559, 4.6587 | 0.15923 | 0.04756 |
| cached-pulse Muon, matrix lr=0.02 | 4.71200 | 4.7120, 4.7185, 4.6923 | 0.14674 | 0.05148 |

## Interpretation

FastMuon-NS3 and FastMuon-NS4 both improved fixed-token validation loss over AdamW in this probe. NS5 remained the best loss result, but NS3 and NS4 preserved a large share of the improvement while using fewer Newton-Schulz iterations.

The most interesting result is not that NS3/NS4 beat NS5. They did not. The interesting result is that five iterations may be more precision than this training regime needs. A partially orthogonalized update with higher residual orthogonality error still improved the loss curve relative to AdamW.

The exploratory pulse variants were less compelling. They reduced Newton-Schulz call frequency, but loss regressed toward AdamW. That suggests that stale orthogonal directions are weaker than fresh but lower-depth orthogonalization.

## Candidate Claim

The current candidate finding is:

> For this local TinyGPT probe, reducing Muon's Newton-Schulz depth from 5 to 3 or 4 preserves most of Muon's sample-efficiency advantage over AdamW while reducing the optimizer-side orthogonalization depth.

This is a research lead, not a final claim.

## Next Validation

1. Extend the same sweep to 500 steps.
2. Repeat with a real byte-level text corpus.
3. Try an annealed schedule such as

$$
k(t) =
\begin{cases}
3, & t < T_1 \\
4, & T_1 \leq t < T_2 \\
5, & t \geq T_2
\end{cases}
$$

4. Measure wall-clock on CUDA, where matrix multiplication timing is more representative of the Muon deployment target.
