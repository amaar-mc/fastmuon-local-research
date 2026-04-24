# Optimizer Research Log

## 2026-04-24: First Local Discovery Cycle

Environment:

- Python 3.13.3
- PyTorch 2.6.0
- Apple MPS
- TinyGPT probe: 2 layers, 64 width, 2 heads, 32-token context, deterministic synthetic token stream
- Probe budget: 100 steps, 3 seeds, 51,200 tokens/run

## Candidate Ideas Tested

- `cimuon`: coherence/interpolated Muon using temporal coherence and matrix anisotropy to set `rho`.
- `pulse_muon`: skips Newton-Schulz on non-pulse steps and uses normalized momentum.
- `cached_pulse_muon`: skips Newton-Schulz on non-pulse steps but reuses the last orthogonalized direction.
- Reduced-depth Muon: regular Muon with fewer Newton-Schulz iterations (`ns_steps=3` or `4` instead of `5`).

## Main Finding

The strongest signal is **reduced-depth Muon**, not the coherence/pulse variants.

Clean run: `results_fast/fast_muon_probe`

| setting | median val loss | seed losses | median step sec | median NS error |
|---|---:|---|---:|---:|
| AdamW lr=0.001 | 4.71216 | 4.7094, 4.7122, 4.7151 | 0.01569 | NA |
| FastMuon-NS3 lr=0.02 | 4.67666 | 4.6842, 4.6744, 4.6767 | 0.13453 | 0.06817 |
| FastMuon-NS4 lr=0.02 | 4.66698 | 4.6705, 4.6670, 4.6664 | 0.14735 | 0.05333 |
| Muon-NS5 lr=0.02 | 4.65868 | 4.6657, 4.6559, 4.6587 | 0.15923 | 0.04756 |
| cached-pulse lr=0.02 | 4.71200 | 4.7120, 4.7185, 4.6923 | 0.14674 | 0.05148 |

Interpretation:

- NS3 and NS4 both beat AdamW on sample efficiency in this probe.
- NS4 keeps most of NS5's loss improvement while using fewer Newton-Schulz iterations.
- NS3 is the stronger efficiency tradeoff: clearly better than AdamW, materially cheaper than NS5, but with a larger loss gap to NS5.
- The pulse/cached-pulse variants reduce Newton-Schulz call frequency, but currently lose too much optimization quality to be the lead candidate.

## Current Hypothesis

Muon does not need five Newton-Schulz iterations at early tiny-LLM training scale. A partially orthogonalized update with moderate orthogonality error may preserve most of Muon's useful spectral balancing while reducing optimizer cost.

Working name: **FastMuon-NS3/NS4**.

## What Is Not Proven Yet

- This is not a research-grade discovery yet; it is a local, small-model signal.
- The result may depend on model size, batch size, MPS backend, synthetic data, or the short 100-step horizon.
- AdamW is much faster on this local tiny setting, so wall-clock claims against AdamW are not meaningful here.
- Need longer runs and a real text dataset before making a stronger claim.

## Next Tests

- Run 500-step TinyGPT sweep comparing AdamW, Muon-NS3, Muon-NS4, and Muon-NS5.
- Add a real byte-level text corpus option and repeat.
- Try an annealed schedule: NS3 early, NS4/NS5 later.
- Test whether NS3/NS4 remains stable with higher model width/depth.

