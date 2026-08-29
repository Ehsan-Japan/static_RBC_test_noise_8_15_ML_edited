# Ray-based sampling for transition-line extraction in DQD stability diagrams

E. Alizadeh Kashtiban, T. Fujita, A. Oiwa — Osaka University (SANKEN).

Recover the charge-transition lines of a double quantum dot from a small
fraction of the gate-voltage plane instead of a full raster scan. Current is
measured along a fan of rays from one corner of the (V₁, V₂) window; a small
U-Net turns those sparse traces into a dense transition-line map. The study
question is: **how little of the grid can we measure and still recover every
line?**

`README.md` is the human-facing overview; `scripts/README.md` is the
step-by-step. This file is the orientation for an agent.

## Layout

```
scripts/          the programs you run — a settings block and a few lines each
src/dqd/          the library.  Nothing here has a command line.
    config/       parameter space, paths, figure house style
    simulation/   the device model, and the DQD acceptance test
    ml/           the ray cutting, the U-Net, training, metrics
    study/        the four stages, the sweep, and the figure gallery
    visualization/ drawing measurements and predictions over a diagram
training_data/    one folder per configuration (+ the cached device pool)
results/          cross-configuration tables, figures and the deck
```

Entry points: `run_0_full_sweep.py` runs the whole study; `run_1`…`run_7`
are the stages. `benchmark.py` is the geometry ablation.

## The claims this repo supports, and the file that makes each one true

- **Devices are simulated once and shared.** Changing the number of rays
  changes how a device is measured, never which device it is.
  (`src/dqd/study/dataset.py`, `device_split.py`)
- **The split is on the device, not the image**, made once with a fixed seed
  and stored with the device pool. No device lands on both sides in any
  budget.
- **Every diagram passes an automated DQD acceptance test** on the simulated
  charge configuration `n(V₁,V₂)` before entering the dataset:
  both dots exchange charge with the reservoir; interdot transitions must be
  present (Δn₁+Δn₂ = 0, Δn₁ ≠ 0 — the signature that separates a true double
  dot from two independent dots); 6–90 charge states so the honeycomb is
  resolvable at 100×100; the sensor responds.
  (`src/dqd/simulation/dqd_validator.py`)
- **Training constants are constants**, not settings — `src/dqd/ml/grid_train.py`,
  so every cell of a sweep trains identically. The architecture is fixed at
  1,949,409 params for all budgets, so accuracy differences come from the
  measurement, not from capacity.
- **Nothing is tuned on the test devices.** The binarisation threshold is
  chosen on a validation split carved out of the *training* devices and
  stored inside the checkpoint. (`src/dqd/study/threshold_report.py` documents
  this at length — read it before writing about the method.)

## Method, in one pass

- **Simulator** — QArray, constant-capacitance model. QD₁, QD₂ on plunger
  gates coupled through C_m; QD_s is the charge sensor. Capacitances sampled
  randomly per device. Ground truth is the exact charge-state boundary from
  the simulator, not an edge-detected image.
- **The voltage window is NOT [−1, 1].** `device_factory.generate` takes a
  *base* window of (−1, 1, −1, 1) and then shifts it per device by
  `offset_scale = 0.35` of the width, i.e. `off ~ U(−0.7, +0.7)` on each
  axis. Every device therefore spans exactly **2 × 2**, but its origin
  varies: across the 550-device pool `x_min` runs −1.695 … −0.301 and
  `y_min` −1.700 … −0.307. The per-device window is stored in each
  `sample_N/device.json` under `voltage_window`. Axes are labelled **mV**
  (`set_axis_labels(..., x_unit="mV", y_unit="mV")`). Say "a 2 × 2 mV window,
  randomly offset per device" — never "[−1, 1]".
  **Noise-free**: `NoNoise()` in `dqd_simulator.py`, nothing added afterwards.
- **Measurement** — n rays × n points fired from one corner of the window.
  A ray crossing a transition line gives a local maximum in the sensor
  current. Rays are oblique, so one ray crosses both honeycomb families.
- **Network input** — 2 channels only: ch0 = raw sensor value where a ray
  passed, ch1 = visited mask (so "measured and low" is distinguishable from
  "never measured"). The peaks channel exists for figures; the network never
  sees it.
- **Network** — fully convolutional U-Net, depth 3 (32→64→128, bottleneck
  256), sigmoid head → **one probability per pixel**.
- **Loss** — BCEWithLogits with the positive class weighted by its rarity
  (capped at 8.0) + soft Dice. Adam 1e-3, batch 16, 50 epochs.
- **Metric** — tolerant **F1@τ**: a predicted pixel counts when a true line
  pixel lies within τ, and vice versa; distance-transform based, averaged per
  device. τ = 0,1,2,3 all reported, τ = 1 is the headline. Pixel accuracy is
  reported only to be dismissed (~93 % for predicting nothing).

## Headline numbers (results/4-5-6-7-8_rays_40-50-60_points_500_samples)

550 devices, 500 train / 50 test, 15 budgets (4–8 rays × 40–60 points).

| budget | coverage | F1@1 | P | R |
|---|---|---|---|---|
| 4 × 40 | 1.57 % | 0.672 | 0.601 | 0.769 |
| 8 × 40 | 3.11 % | 0.810 | 0.753 | 0.881 |
| 8 × 50 | 3.88 % | 0.835 | 0.801 | 0.875 |
| **8 × 60** | **4.66 %** | **0.849** | **0.818** | **0.885** |

τ-sweep at the best budget: F1@0 0.495 → F1@1 0.849 → F1@2 0.933 →
F1@3 0.967, so most residual error is sub-pixel placement of a line that was
found, not a missed line. Strict IoU 0.333 (one-pixel-wide lines are punished
hard by IoU — report it, don't hide it). Per-device F1@1 sd falls 0.128 →
0.078 as rays increase.

Chosen thresholds across the 15 budgets land at **0.4–0.6** and move with the
budget — never 0.5.

**Geometry ablation** (`benchmark_geometry_8_rays_60_points_500_samples`,
same 480-point budget, same devices, same network):

| arm | coverage | F1@1 |
|---|---|---|
| parallel diagonal | 3.82 % | 0.879 |
| corner fan (ours) | 4.66 % | 0.849 |
| horizontal cuts (Hernandes-style) | 4.80 % | 0.841 |
| vertical cuts | 4.80 % | 0.839 |

Parallel oblique lines beat the corner fan, from fewer pixels — fan rays
crowd near their common origin. The defensible claim is **oblique geometry
wins**: both oblique arms beat both axis-aligned arms, because an oblique
line crosses both families of honeycomb edges.

## Comparison papers (for the manuscript)

- **Hernandes et al., "Reconstructing Quantum Dot Charge Stability Diagrams
  with Diffusion Models"** (arXiv:2603.26432, TU Delft) — conditional
  diffusion reconstructs the full CSD image from a sparse mask (uniform grid
  or line cuts), ~4 % of the data, ~9000 training examples. Target is an
  *image*; lines still have to be extracted and no per-pixel confidence is
  attached. Their line-cut masking is our `hcuts` benchmark arm.
- **Muto et al., "Automatic detection of single-electron regime and virtual
  gate definition in quantum dots using U-Net and clustering"**
  (Sci. Rep., 2026; arXiv:2501.05878; Tohoku/RIKEN) — U-Net segments lines in
  *experimental* diagrams, then Hough transform + clustering give line
  angles, the virtual-gate matrix and the single-electron regime. Their U-Net
  consumes a **full raster scan** — acquisition cost untouched. Our
  contribution sits upstream of theirs and their Hough stage is compatible
  with our output.

**The gap we fill:** neither attaches a calibrated probability to a
transition line, and neither treats the binarisation threshold as a quantity
to be defined, chosen out-of-sample and reported.

## Limitations to state, not bury

Simulation only and noise-free; constant-capacitance model, double dot,
fixed 100×100 resolution and a fixed 2 × 2 window size (only its origin
varies); fixed ray origin; rays are non-adaptive; strict IoU is low.

## Conventions

- Reporting is **F1@τ**, τ = 0…3, τ = 1 as headline. Never quote pixel
  accuracy as a result.
- Don't tune anything on the test devices.
- `results/`, `training_data/`, `docs/`, `*.pptx`, `*.pdf`, `*.docx` are
  gitignored — they're reproducible from code, or they're paper material.
- The folder name says `noise_8_15` but **the simulation carries no noise**.
  Don't infer otherwise from the path.
- `README.md` advertises `run_9_make_slides.py`, which does not exist.
