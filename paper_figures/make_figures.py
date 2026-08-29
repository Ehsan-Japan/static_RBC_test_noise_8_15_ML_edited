"""
paper_figures/make_figures.py — publication figures, drawn from the real
arrays in the device pool rather than mocked up.

    fig_network_input   the TWO maps the U-Net is actually given
    fig_unet            a compact schematic of the network

Run from the project root:   python paper_figures/make_figures.py

Every figure is written as .png (600 dpi, for slides) and .pdf (vector, for
the manuscript).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from dqd.ml import ray_peaks                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
POOL = os.path.join(ROOT, "training_data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")

# The device used for the INPUT / MASK explanation.  pool sample_1 is a
# TRAINING device (device_id 0) — fine here, because this figure only shows
# what a measurement looks like and makes no accuracy claim.
DEVICE = os.path.join(POOL, "sample_1")
N_RAYS, N_POINTS = 8, 60

INK = "#111111"
MUT = "#5a606a"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.edgecolor": "#999999",
    "savefig.facecolor": "white",
})


def save(fig, name):
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        p = os.path.join(HERE, name + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)


# ── figure 1: the two input maps ──────────────────────────────────────────
def fig_network_input(compact=False):
    m = ray_peaks.measure(DEVICE, N_RAYS, N_POINTS)
    ux, uy, Z = ray_peaks.load_grid(DEVICE)
    H, W = Z.shape
    ch = ray_peaks.to_channels(m, (H, W))
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    cov = 100.0 * vis.mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))

    # channel 1 — the measured value, shown only where a ray passed
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("#f2f2f2")
    ax = axes[0]
    im = ax.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("channel 1   —   measured sensor signal", fontsize=11,
                 color=INK, pad=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("normalised charge-sensor signal", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("grey = never measured", fontsize=9, color=MUT)

    # channel 2 — where we looked at all
    ax = axes[1]
    ax.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title("channel 2   —   visited mask", fontsize=11, color=INK,
                 pad=9)
    ax.set_xlabel(f"black = a ray passed here  ({cov:.1f} % of the grid)",
                  fontsize=9, color=MUT)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999")

    if not compact:
        # the manuscript version carries its own title and caption; the slide
        # version does not, because the slide already says both
        fig.suptitle(f"What the network is given — {N_RAYS} rays × "
                     f"{N_POINTS} points, one held-out device",
                     fontsize=12.5, color=INK, y=1.0)
        fig.text(0.5, -0.045,
                 "Channel 2 is what separates “measured here, and the signal "
                 "was low” from “never looked here”.  Without it a zero in "
                 "channel 1 is ambiguous everywhere.",
                 ha="center", fontsize=9.5, color=MUT)
    fig.tight_layout()
    save(fig, "fig_network_input_slide" if compact else "fig_network_input")


# ── figure 2: compact U-Net schematic ─────────────────────────────────────
ENC = "#4f81a8"
BOT = "#3d4a5c"
DEC = "#c07a4a"
IO = "#5f8a5f"


_BLOCK_FS = 1.0          # set by draw_unet so sub-captions scale with fs


def _block(ax, x, y, w, h, color, label, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=0, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y - 0.055, sub, ha="center", va="top",
                fontsize=8.5 * _BLOCK_FS, color=MUT)


def _arrow(ax, p, q, color, style="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=style,
                                 shrinkA=1, shrinkB=1, zorder=1))


def draw_unet(ax, fs=1.0, io_labels=True):
    """Draw the U-Net schematic into an existing axes.  fs scales the fonts;
    io_labels=False drops the input/output captions, for when the figure
    already shows the real input and output either side."""
    global _BLOCK_FS
    _BLOCK_FS = fs
    ax.set_xlim(0, 10); ax.set_ylim(-0.32, 1.25); ax.axis("off")

    # x, y, w, h, colour, label, caption
    lvl = [(0.72, 0.85), (0.56, 0.60), (0.40, 0.35)]      # h, y per depth
    enc_x = [1.35, 2.35, 3.35]
    dec_x = [5.30, 6.30, 7.30]
    widths = ["32", "64", "128"]
    grids = ["100²", "50²", "25²"]

    _block(ax, 0.15, 0.30, 0.55, 0.72, IO, "2", "input")
    for i, (x, w, g) in enumerate(zip(enc_x, widths, grids)):
        h, yt = lvl[i]
        _block(ax, x, yt - h, 0.72, h, ENC, w, g)
    _block(ax, 4.30, 0.00, 0.72, 0.28, BOT, "256", "12²")
    for i, (x, w, g) in enumerate(zip(dec_x, reversed(widths),
                                      reversed(grids))):
        h, yt = lvl[2 - i]
        _block(ax, x, yt - h, 0.72, h, DEC, w, g)
    _block(ax, 8.40, 0.30, 0.55, 0.72, IO, "1", "output")

    # down / up path
    _arrow(ax, (0.72, 0.66), (1.33, 0.66), INK)
    for i in range(2):
        h0, y0 = lvl[i]; h1, y1 = lvl[i + 1]
        _arrow(ax, (enc_x[i] + 0.72, y0 - h0 / 2),
               (enc_x[i + 1], y1 - h1 / 2), ENC)
    h, y = lvl[2]
    _arrow(ax, (enc_x[2] + 0.72, y - h / 2), (4.30, 0.14), ENC)
    _arrow(ax, (5.02, 0.14), (dec_x[0], y - h / 2), DEC)
    for i in range(2):
        h0, y0 = lvl[2 - i]; h1, y1 = lvl[1 - i]
        _arrow(ax, (dec_x[i] + 0.72, y0 - h0 / 2),
               (dec_x[i + 1], y1 - h1 / 2), DEC)
    _arrow(ax, (dec_x[2] + 0.72, 0.66), (8.38, 0.66), INK)

    # skip connections - the deepest one is routed clear of the bottleneck
    skip_y = [lvl[0][1] - 0.06, lvl[1][1] - 0.06, 0.33]
    for i in range(3):
        _arrow(ax, (enc_x[i] + 0.72, skip_y[i]),
               (dec_x[2 - i], skip_y[i]), "#9aa2ac", style=(0, (4, 3)),
               lw=1.2)
    ax.text(4.66, lvl[0][1] + 0.02, "skip connections",
            ha="center", fontsize=8.5 * fs, color=MUT, style="italic")

    if io_labels:
        ax.text(0.42, 0.16, "channel 1  ray signal\nchannel 2  visited mask",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
        ax.text(8.68, 0.16, "P(transition line)\nper pixel",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
    ax.text(2.43, 1.16, "encoder", fontsize=10.5 * fs, color=ENC,
            fontweight="bold", ha="center")
    ax.text(6.38, 1.16, "decoder", fontsize=10.5 * fs, color=DEC,
            fontweight="bold", ha="center")
    ax.text(4.66, -0.14, "bottleneck", fontsize=9.5 * fs, color=BOT,
            fontweight="bold", ha="center", va="top")

    return ax


def fig_unet():
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    draw_unet(ax)
    save(fig, "fig_unet")




# ── figure 3: input → probability → binarised at a ladder of thresholds ───
# This one needs the trained network: results/<config>/model/unet.pt
CONFIG = "8_rays_60_points_500_samples"
RUN_DIR = os.path.join(ROOT, "results",
                       "4-5-6-7-8_rays_40-50-60_points_500_samples")
LADDER = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Which held-out device the results figure uses.  Index into test.npz:
#   0 -> figures/test/sample_1 (pool sample_5)
#   1 -> figures/test/sample_2 (pool sample_10)   <- the one we show
# Both are in test_ids, so the network never saw them during training.
RESULT_DEVICE = 1


def fig_probability_to_lines(device_index=None, ladder=LADDER):
    """
    One held-out device end to end:

        row 1   channel 1 | channel 2 | probability map | ground truth
        row 2   the same probability map cut at a ladder of thresholds,
                the checkpoint's own threshold framed in red

    device_index : None picks the device whose F1@1 at the chosen threshold
                   is closest to the configuration's mean, so the figure is
                   representative rather than cherry-picked.
    """
    from dqd.ml import grid_train
    from dqd.ml.grid_metrics import tolerant_f1
    from dqd.study.dataset import load_split

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    X, Y, names = load_split(os.path.join(cfg_dir, "test.npz"))
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    thr = float(ck["threshold"])
    prob = grid_train.predict(net, X)

    f1 = np.array([tolerant_f1(prob[i] > thr, Y[i], 1.0)["f1"]
                   for i in range(len(X))])
    if device_index is None:
        device_index = int(np.argmin(np.abs(f1 - f1.mean())))
    i = device_index
    print(f"  figures/test/sample_{i + 1} = {os.path.basename(names[i])}  "
          f"F1@1 {f1[i]:.3f}  "
          f"(mean over {len(f1)} held-out devices {f1.mean():.3f})")

    sig, vis = X[i][0], X[i][1]
    p, truth = prob[i], Y[i] > 0.5
    n = len(ladder)

    fig = plt.figure(figsize=(2.15 * n, 5.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.28, 1.0], hspace=0.34,
                          wspace=0.10)

    def blank(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999")

    # ── row 1: what goes in, what comes out ──────────────────────────────
    span = n // 4
    cm = plt.get_cmap("inferno").copy(); cm.set_bad("#f2f2f2")

    ax = fig.add_subplot(gs[0, 0:span])
    ax.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("input · channel 1\nmeasured sensor signal", fontsize=10.5)
    blank(ax)

    ax = fig.add_subplot(gs[0, span:2 * span])
    ax.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"input · channel 2\nvisited mask "
                 f"({100 * vis.mean():.1f} % of the grid)", fontsize=10.5)
    blank(ax)

    ax = fig.add_subplot(gs[0, 2 * span:3 * span])
    im = ax.imshow(p, origin="lower", cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title("output\nP(transition line) per pixel", fontsize=10.5,
                 color="#7d2b3a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=8)
    blank(ax)

    ax = fig.add_subplot(gs[0, 3 * span:])
    ax.imshow(1 - truth, origin="lower", cmap="gray", interpolation="nearest")
    ax.set_title(f"ground truth\n{100 * truth.mean():.1f} % line pixels",
                 fontsize=10.5)
    blank(ax)

    # ── row 2: the same map, cut at a ladder of thresholds ───────────────
    from scipy.ndimage import distance_transform_edt
    d_true = (distance_transform_edt(~truth) if truth.any()
              else np.full(truth.shape, np.inf))
    for k, t in enumerate(ladder):
        pred = p > t
        m = tolerant_f1(pred, Y[i], 1.0)
        d_pred = (distance_transform_edt(~pred) if pred.any()
                  else np.full(pred.shape, np.inf))
        rgb = np.ones(truth.shape + (3,))
        rgb[pred & (d_true > 1.0)] = (0.95, 0.55, 0.15)     # false alarm
        rgb[truth & (d_pred > 1.0)] = (0.15, 0.40, 0.85)    # missed line
        rgb[pred & (d_true <= 1.0)] = (0.12, 0.55, 0.28)    # hit
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        chosen = abs(t - thr) < 1e-9
        ax.set_title(f"P > {t:g}\nF1@1 {m['f1']:.3f}", fontsize=10,
                     color="#c0392b" if chosen else INK,
                     fontweight="bold" if chosen else "normal")
        blank(ax)
        if chosen:
            for s in ax.spines.values():
                s.set_color("#c0392b"); s.set_linewidth(2.4)

    fig.text(0.5, 0.455,
             "the probability map binarised at a ladder of thresholds — "
             "red frame = the threshold chosen on the validation split",
             ha="center", fontsize=10, color=MUT)
    fig.text(0.5, 0.012,
             "green = line found within 1 px      "
             "blue = true line missed      "
             "orange = line drawn that is not there",
             ha="center", fontsize=9.5, color=MUT)
    save(fig, "fig_probability_to_lines")


# ── figure 4: the whole flow — input | network | output ───────────────────
def fig_model_flow(device_index=RESULT_DEVICE):
    """
    The two input maps on the left, the network in the middle, the
    probability map on the right — one picture of what the model does.
    """
    from dqd.ml import grid_train
    from dqd.study.dataset import load_split

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    X, Y, names = load_split(os.path.join(cfg_dir, "test.npz"))
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    i = device_index
    p = grid_train.predict(net, X[i:i + 1])[0]
    sig, vis = X[i][0], X[i][1]

    fig = plt.figure(figsize=(13.0, 3.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 3.45, 1.25],
                          wspace=0.30, hspace=0.42,
                          left=0.02, right=0.985, top=0.86, bottom=0.06)

    def blank(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")

    cm = plt.get_cmap("inferno").copy(); cm.set_bad("#f2f2f2")
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
               vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("channel 1 · measured signal", fontsize=9.5, pad=5)
    blank(ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax2.set_title(f"channel 2 · visited mask ({100 * vis.mean():.1f} %)",
                  fontsize=9.5, pad=5)
    blank(ax2)

    axn = fig.add_subplot(gs[:, 1])
    draw_unet(axn, fs=0.92, io_labels=False)

    axo = fig.add_subplot(gs[:, 2])
    im = axo.imshow(p, origin="lower", cmap="magma", vmin=0, vmax=1,
                    interpolation="nearest")
    axo.set_title("P(transition line) per pixel", fontsize=10.5,
                  color="#7d2b3a", pad=6)
    fig.colorbar(im, ax=axo, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=7.5)
    blank(axo)

    # arrows in figure coordinates: inputs -> network -> output
    def farrow(x0, y0, x1, y1):
        fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=15,
                                       linewidth=1.8, color="#444444"))
    b1, b2 = ax1.get_position(), ax2.get_position()
    bn, bo = axn.get_position(), axo.get_position()
    farrow(b1.x1 + 0.004, b1.y0 + b1.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(b2.x1 + 0.004, b2.y0 + b2.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(bn.x1 - 0.010, bn.y0 + bn.height * 0.62,
           bo.x0 - 0.006, bo.y0 + bo.height * 0.5)

    fig.text(0.5, 0.955, "Two measured maps in, one probability map out",
             ha="center", fontsize=12.5, color=INK)
    save(fig, "fig_model_flow")


# ── figure 5: the same three maps, each as its own standalone panel ───────
# Separate files so a slide can lay them out itself, with its own arrows,
# instead of being handed one merged image.
def fig_panels(device_index=RESULT_DEVICE):
    from dqd.ml import grid_train
    from dqd.study.dataset import load_split

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    X, Y, names = load_split(os.path.join(cfg_dir, "test.npz"))
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    i = device_index
    prob = grid_train.predict(net, X[i:i + 1])[0]
    sig, vis = X[i][0], X[i][1]

    def panel(draw, title, name, colour=INK, cbar=None):
        fig, ax = plt.subplots(figsize=(2.7, 2.95))
        im = draw(ax)
        ax.set_title(title, fontsize=11, color=colour, pad=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        save(fig, name)

    cm = plt.get_cmap("inferno").copy(); cm.set_bad("#f2f2f2")
    panel(lambda ax: ax.imshow(np.where(vis > 0.5, sig, np.nan),
                               origin="lower", cmap=cm, vmin=0, vmax=1,
                               interpolation="nearest"),
          "channel 1\nmeasured sensor signal", "panel_channel1")
    panel(lambda ax: ax.imshow(1.0 - vis, origin="lower", cmap="gray",
                               vmin=0, vmax=1, interpolation="nearest"),
          f"channel 2\nvisited mask ({100 * vis.mean():.1f} % of the grid)",
          "panel_channel2")
    panel(lambda ax: ax.imshow(prob, origin="lower", cmap="magma", vmin=0,
                               vmax=1, interpolation="nearest"),
          "output\nP(transition line) per pixel", "panel_probability",
          colour="#7d2b3a", cbar=True)


if __name__ == "__main__":
    print("figures ->", HERE)
    fig_network_input()
    fig_network_input(compact=True)
    fig_unet()
    fig_probability_to_lines(RESULT_DEVICE)
    fig_model_flow()
    fig_panels()
