"""Draw the documentation figures from the logged results (no competition data needed).

Usage: python scripts/make_figures.py   (writes PNG files to docs/assets/)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneemil.constants import SLOTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BOX, BOX_EDGE = "#f1f0ec", "#c9c8c1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
POSITIVES = [24, 9, 26, 23, 15, 11, 21, 35, 27, 12, 19, 18]

# Results logged on Kaggle (58 gold studies)
LLM_V3 = [0.919, 0.884, 0.911, 0.825, 0.902, 0.853, 0.820, 0.771, 0.716, 0.894, 0.774, 0.804]
K4 = [0.936, 0.873, 0.960, 0.892, 0.969, 0.870, 0.834, 0.917, 0.787, 0.967, 0.923, 0.960]
ENSEMBLE = [0.940, 0.893, 0.954, 0.867, 0.966, 0.861, 0.834, 0.916, 0.769, 0.962, 0.924, 0.947]
K4_CURVE = [0.7638, 0.8427, 0.8681, 0.8939, 0.9030, 0.9053, 0.9075, 0.9074]
K4B_CURVE = [0.7758, 0.8438, 0.8704, 0.8833, 0.8887, 0.8920, 0.8917, 0.8904]


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, text, fc=BOX, ec=BOX_EDGE, color=INK, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=1.4, linestyle=ls, shrinkA=2, shrinkB=2))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.1, 4.35, "Labels from text (training only)", fontsize=10, color=INK2, weight="bold")
    box(ax, 0.1, 3.0, 2.2, 1.0, "4,349 radiology\nreports\n(multilingual)")
    box(ax, 2.9, 3.0, 2.5, 1.0, "02  Local LLM\nQwen2.5 7B Instruct\n5 certainty levels")
    box(ax, 6.0, 3.0, 2.3, 1.0, "Soft targets\n12 per study\n0.95 ... 0.03", fc="#e8f1fb", ec=BLUE)
    arrow(ax, 2.3, 3.5, 2.9, 3.5)
    arrow(ax, 5.4, 3.5, 6.0, 3.5)

    ax.text(0.1, 2.25, "Images (training and test)", fontsize=10, color=INK2, weight="bold")
    xs = [0.1, 2.1, 4.25, 6.25, 8.2, 10.2]
    labels = ["DICOM study\nseveral series", "01  Inventory\nplane, sequence,\nslice order",
              "03  Canonical\nvolume\n5 slots, 64 slices", "2.5D windows\n3 slices =\n3 channels",
              "04  ConvNeXt Tiny\n+ per-finding\nattention (MIL)", "05  12 probabilities\nper study"]
    for i, (x, t) in enumerate(zip(xs, labels)):
        fc, ec = ("#e8f1fb", BLUE) if i == 4 else (BOX, BOX_EDGE)
        box(ax, x, 0.75, 1.7, 1.15, t, fc=fc, ec=ec)
        if i:
            arrow(ax, xs[i - 1] + 1.7, 1.32, x, 1.32)
    arrow(ax, 7.15, 3.0, 9.05, 1.9, color=BLUE)
    ax.text(8.25, 2.55, "trains", color=BLUE, fontsize=9)
    box(ax, 9.3, 3.0, 2.6, 1.0, "58 gold studies\nexpert labels\nevaluation only", fc="#fdf0ea", ec=ORANGE)
    arrow(ax, 10.6, 3.0, 9.6, 1.9, color=ORANGE, ls="--")
    ax.text(10.5, 2.5, "never trained on", color=ORANGE, fontsize=9)
    ax.text(6.0, 0.25, "No report is available at test time: the final model reads images only.",
            fontsize=9, color=MUTED, ha="center")
    save(fig, "pipeline.png")


def canonical_volume():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(-0.5, 64.5)
    ax.set_ylim(-2.6, 3.2)
    ax.axis("off")
    colors = ["#2a78d6", "#7fb0ea", "#eb6834", "#f4ac8c", "#1baf7a"]
    start = 0
    for (name, plane, kind, k), c in zip(SLOTS, colors):
        empty = name == "cor_anat"
        for i in range(k):
            ax.add_patch(Rectangle((start + i + 0.08, 0), 0.84, 1.4, fc=SURFACE if empty else c,
                                   ec=c if empty else "none", lw=0.8, hatch="////" if empty else None))
        seq = {"fluid": "fat-sat (fluid)", "anat": "non-sat"}.get(kind, "any")
        ax.text(start + k / 2, 1.75, f"{plane}\n{seq}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(start + k / 2, -0.35, f"{k} slices", ha="center", va="top", fontsize=8.5, color=INK2)
        start += k
    ax.text(32, -2.35, "Example: the coronal non-saturated series is missing (about 1 study in 4). "
            "The slot stays empty and is masked.",
            ha="center", fontsize=9, color=MUTED)
    for c, col in [(21, INK)]:
        ax.add_patch(Rectangle((c - 1 + 0.02, -0.08), 3 - 0.04, 1.56, fc="none", ec=col, lw=2))
    ax.text(21.5, -1.35, "a 2.5D window: slices c-1, c, c+1 become the 3 colour channels (never across two slots)",
            ha="center", fontsize=9, color=INK)
    arrow(ax, 21.5, -1.05, 21.5, -0.12, color=INK)
    save(fig, "canonical_volume.png")


def mil_head():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    for i in range(5):
        y = 3.3 - i * 0.7
        box(ax, 0.1, y, 1.2, 0.5, f"window {i + 1}" if i < 4 else "window K", size=8.5)
        arrow(ax, 1.3, y + 0.25, 2.0, y + 0.25)
    box(ax, 2.0, 0.5, 1.6, 3.3, "Shared\nbackbone\nConvNeXt Tiny\n(ImageNet)\n\none feature\nvector per\nwindow")
    box(ax, 4.2, 0.5, 1.6, 3.3, "+ slice-position\nembedding\n\ntells the head\nwhich slot\na window\ncomes from")
    arrow(ax, 3.6, 2.15, 4.2, 2.15)
    heads = [("ACL", BLUE), ("Effusion", ORANGE), ("...", MUTED), ("Fracture", AQUA)]
    for i, (n, c) in enumerate(heads):
        y = 3.2 - i * 0.85
        box(ax, 6.5, y, 2.4, 0.6, f"attention for {n}" if n != "..." else "12 attention heads",
            fc=SURFACE, ec=c, size=8.5)
        arrow(ax, 5.8, 2.15, 6.5, y + 0.3)
        box(ax, 9.5, y, 1.8, 0.6, f"P({n})" if n != "..." else "...", fc=SURFACE, ec=c, size=8.5)
        arrow(ax, 8.9, y + 0.3, 9.5, y + 0.3)
    ax.text(7.7, 0.12, "softmax over windows: each finding chooses its own slices", ha="center",
            fontsize=9, color=MUTED)
    save(fig, "mil_head.png")


def llm_versions():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = ["v1  strict parser", "v2  tolerant parser\n+ glossary", "v3  finding words\n+ stricter prompt"]
    vals = [0.758, 0.832, 0.839]
    fails = ["19 / 58 unparsed", "1 / 58 unparsed", "0 / 58 unparsed"]
    ys = [2, 1, 0]
    ax.plot(vals, ys, color=GRID, lw=2, zorder=1)
    ax.scatter(vals, ys, color=BLUE, s=70, zorder=2)
    for v, y, f in zip(vals, ys, fails):
        ax.text(v + 0.005, y, f"{v:.3f}   ({f})", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0.74, 0.90)
    ax.set_xlabel("Gold macro AUC of the LLM labels (58 studies)")
    ax.grid(axis="y", visible=False)
    ax.set_title("LLM labelling, three iterations", loc="left", fontsize=11, color=INK)
    save(fig, "llm_versions.png")


def training_curves():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ep = list(range(1, 9))
    ax.plot(ep, K4_CURVE, color=BLUE, lw=2, marker="o", ms=5, label="Image model, seed 42")
    ax.plot(ep, K4B_CURVE, color=ORANGE, lw=2, marker="s", ms=5, label="Image model, seed 7")
    ax.axhline(0.839, color=MUTED, lw=1.5, ls="--", label="LLM labels it was trained on")
    ax.text(8.1, K4_CURVE[-1], "0.907", color=INK, va="center", fontsize=9)
    ax.text(8.1, K4B_CURVE[-1], "0.890", color=INK, va="center", fontsize=9)
    ax.text(8.1, 0.843, "0.839", color=INK2, va="bottom", fontsize=9)
    ax.set_xlim(0.7, 8.7)
    ax.set_ylim(0.74, 0.93)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gold macro AUC (58 studies)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("The student overtakes its teacher", loc="left", fontsize=11, color=INK)
    save(fig, "training_curves.png")


def per_finding():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    order = sorted(range(12), key=lambda i: K4[i])
    ys = range(12)
    for y, i in zip(ys, order):
        ax.plot([LLM_V3[i], K4[i]], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter([LLM_V3[i] for i in order], ys, color=MUTED, s=45, zorder=2, label="LLM labels (text)")
    ax.scatter([K4[i] for i in order], ys, color=BLUE, s=55, zorder=3, label="Image model (seed 42)")
    ax.scatter([ENSEMBLE[i] for i in order], ys, color=ORANGE, marker="D", s=35, zorder=4,
               label="Ensemble of 2 seeds")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{FINDINGS[i]}  (n={POSITIVES[i]})" for i in order])
    ax.set_xlim(0.68, 1.0)
    ax.set_xlabel("AUC on the 58 gold studies (n = positive cases)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Per finding: images recover what reports leave out", loc="left", fontsize=11, color=INK)
    save(fig, "per_finding.png")


if __name__ == "__main__":
    pipeline()
    canonical_volume()
    mil_head()
    llm_versions()
    training_curves()
    per_finding()
