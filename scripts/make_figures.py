"""Generate the teaching figures used on the site, in light and dark versions.

Usage:
    pip install matplotlib numpy
    python scripts/make_figures.py [--font-dir DIR]

Writes docs/assets/img/<name>.svg and docs/assets/img/<name>-dark.svg. Pages show
the right one with Material's #only-light / #only-dark image suffixes.

Every curve is computed from its formula, or from a seeded simulation where noted.
Figures marked "schematic" show shapes, not measurements. Text is converted to
paths, so the SVGs look the same everywhere. If --font-dir contains IBM Plex Sans
TTF files they are used (matching the site); otherwise matplotlib's default sans.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "img")


@dataclass(frozen=True)
class Theme:
    name: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    surface: str
    masked: str
    s1: str  # blue
    s2: str  # orange
    s3: str  # aqua
    s4: str  # yellow
    seq: tuple  # sequential ramp, low -> high magnitude


LIGHT = Theme(
    name="light", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
    axis="#c3c2b7", surface="#ffffff", masked="#f0efec",
    s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100",
    seq=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2e333d",
    axis="#4a505c", surface="#1f232b", masked="#2a2f38",
    s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500",
    seq=("#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)

LW = 1.8  # about 2px at the size figures are shown


def setup_fonts(font_dir: str | None) -> None:
    family = None
    if font_dir and os.path.isdir(font_dir):
        for fn in sorted(os.listdir(font_dir)):
            if fn.lower().endswith((".ttf", ".otf")):
                font_manager.fontManager.addfont(os.path.join(font_dir, fn))
                if "plex" in fn.lower():
                    family = "IBM Plex Sans"
    plt.rcParams.update(
        {
            "font.family": family or "DejaVu Sans",
            "font.size": 9,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "path",
            "svg.hashsalt": "aiml-map",
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def style(ax, t: Theme, grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t.axis, labelcolor=t.ink2, labelsize=8, length=3, width=0.8)
    ax.title.set_color(t.ink)
    ax.xaxis.label.set_color(t.ink2)
    ax.yaxis.label.set_color(t.ink2)
    if grid:
        ax.grid(True, color=t.grid, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)
    ax.set_facecolor("none")


def legend(ax, t: Theme, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=t.ink2, handlelength=1.6, **kw)
    return leg


def save(fig, name: str, t: Theme) -> None:
    os.makedirs(OUT, exist_ok=True)
    suffix = "" if t.name == "light" else "-dark"
    path = os.path.join(OUT, f"{name}{suffix}.svg")
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.06,
                metadata={"Date": None, "Creator": None})
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# ---------------------------------------------------------------- figures

def fig_bias_variance(t: Theme) -> None:
    """Schematic: shapes only."""
    x = np.linspace(0.3, 10, 400)
    bias2 = 0.85 * np.exp(-0.45 * x)
    var = 0.0042 * x ** 2.3
    noise = np.full_like(x, 0.12)
    total = bias2 + var + noise
    train = 0.88 * np.exp(-0.36 * x) + 0.02
    xm = x[np.argmin(total)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax in (a, b):
        style(ax, t, grid=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Model complexity →")
        ax.set_ylabel("Error")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 10.6)

    a.set_title("Training vs validation error")
    a.plot(x, train, color=t.s1, lw=LW, label="training error")
    a.plot(x, total, color=t.s2, lw=LW, label="validation error")
    a.axvline(xm, color=t.axis, lw=0.8)
    a.text(xm, 1.0, " best complexity", color=t.ink2, fontsize=8, va="top")
    a.text(0.5, 0.06, "underfitting\n(high bias)", color=t.ink2, fontsize=8)
    a.text(6.0, 0.72, "overfitting\n(high variance)", color=t.ink2, fontsize=8)
    a.text(10.1, train[-1] + 0.04, "training", color=t.ink2, fontsize=8, ha="right")
    a.text(10.1, total[-1] + 0.02, "validation", color=t.ink2, fontsize=8, ha="right")
    legend(a, t, loc="lower right", bbox_to_anchor=(1.0, 0.16))

    b.set_title("Where validation error comes from")
    b.plot(x, total, color=t.s2, lw=LW, label="total error")
    b.plot(x, bias2, color=t.s3, lw=LW, label="bias²")
    b.plot(x, var, color=t.s4, lw=LW, label="variance")
    b.plot(x, noise, color=t.muted, lw=1.0, ls=(0, (4, 3)), label="irreducible noise")
    b.text(0.55, bias2[8] + 0.03, "bias²", color=t.ink2, fontsize=8)
    b.text(9.0, var[-1] - 0.17, "variance", color=t.ink2, fontsize=8, ha="right")
    b.text(10.1, 0.15, "noise", color=t.ink2, fontsize=8, ha="right")
    legend(b, t, loc="upper center", bbox_to_anchor=(0.52, 1.0))
    fig.text(0.995, -0.02, "Schematic: shapes, not measurements", color=t.muted, fontsize=7, ha="right")
    fig.tight_layout(w_pad=2.5)
    save(fig, "bias-variance", t)


def _constrained_min(c, A, boundary):
    d = boundary - c
    losses = np.einsum("ni,ij,nj->n", d, A, d)
    i = int(np.argmin(losses))
    return boundary[i], losses[i]


def fig_l1_l2(t: Theme) -> None:
    theta = math.radians(-25)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    A = R @ np.diag([1.0, 4.0]) @ R.T
    c = np.array([0.55, 1.9])

    s = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    circle = np.stack([np.cos(s), np.sin(s)], axis=1)
    diamond = circle / np.abs(circle).sum(axis=1, keepdims=True)

    gx, gy = np.meshgrid(np.linspace(-1.8, 2.6, 160), np.linspace(-1.8, 2.8, 160))
    G = np.stack([gx - c[0], gy - c[1]], axis=-1)
    Z = np.einsum("...i,ij,...j->...", G, A, G)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    specs = [
        (axes[0], diamond, "L1 (lasso)", "touches a corner:\n$w_1 = 0$ exactly"),
        (axes[1], circle, "L2 (ridge)", "touches a smooth edge:\nboth weights shrink"),
    ]
    for ax, shape, title, note in specs:
        style(ax, t, grid=False)
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 2.6)
        ax.set_ylim(-1.6, 2.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.axhline(0, color=t.axis, lw=0.8)
        ax.axvline(0, color=t.axis, lw=0.8)
        ax.text(2.55, -0.18, "$w_1$", color=t.ink2, ha="right", va="top")
        ax.text(0.08, 2.75, "$w_2$", color=t.ink2, va="top")
        ax.set_title(title)
        ax.add_patch(Polygon(shape, closed=True, facecolor=t.s1, alpha=0.12, edgecolor="none"))
        ax.add_patch(Polygon(shape, closed=True, facecolor="none", edgecolor=t.s1, lw=LW))
        point, level = _constrained_min(c, A, shape)
        ax.contour(gx, gy, Z, levels=[level, level * 2.2, level * 3.8], colors=[t.s2], linewidths=1.2)
        ax.plot(*c, marker="o", ms=4, color=t.ink, mec=t.surface, mew=1.2)
        ax.text(c[0] + 0.12, c[1] + 0.05, "best fit with\nno penalty", color=t.ink2, fontsize=7.5)
        ax.plot(*point, marker="o", ms=7, color=t.ink, mec=t.surface, mew=1.5, zorder=5)
        ax.text(1.0, -1.15, note, color=t.ink2, fontsize=8)
    handles = [
        Patch(facecolor=t.s1, alpha=0.35, edgecolor=t.s1, label="allowed weights (constraint)"),
        plt.Line2D([], [], color=t.s2, lw=1.2, label="loss contours"),
        plt.Line2D([], [], color=t.ink, marker="o", lw=0, ms=6, label="regularized solution"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=t.ink2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    save(fig, "l1-l2", t)


def fig_roc_pr(t: Theme) -> None:
    """Seeded simulation: 1% positives, positive scores shifted by 1.8 standard deviations."""
    rng = np.random.default_rng(7)
    n_neg, n_pos = 99_000, 1_000
    scores = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(1.8, 1, n_pos)])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / n_pos
    fpr = fp / n_neg
    precision = tp / (tp + fp)
    auc = np.trapezoid(np.concatenate([[0], tpr]), np.concatenate([[0], fpr]))
    recall_prev = np.concatenate([[0], tpr[:-1]])
    ap = np.sum((tpr - recall_prev) * precision)
    p_at_half = precision[np.searchsorted(tpr, 0.5)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax in (a, b):
        style(ax, t)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    a.set_title(f"ROC curve · AUC = {auc:.2f}")
    a.plot([0, 1], [0, 1], color=t.muted, lw=1.0, ls=(0, (4, 3)))
    a.plot(fpr, tpr, color=t.s1, lw=LW)
    a.text(0.55, 0.42, "random\nclassifier", color=t.ink2, fontsize=8)
    a.text(0.25, 0.86, "looks strong", color=t.ink2, fontsize=8)
    a.set_xlabel("False positive rate")
    a.set_ylabel("True positive rate (recall)")

    b.set_title(f"Precision–recall · AP = {ap:.2f}")
    b.axhline(n_pos / (n_pos + n_neg), color=t.muted, lw=1.0, ls=(0, (4, 3)))
    b.plot(tpr, precision, color=t.s1, lw=LW)
    b.plot(0.5, p_at_half, marker="o", ms=6, color=t.s1, mec=t.surface, mew=1.5, zorder=5)
    b.annotate(f"at 50% recall, only {p_at_half:.0%}\nof flagged cases are fraud",
               xy=(0.5, p_at_half), xytext=(0.47, 0.62), color=t.ink2, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=t.axis, lw=0.8))
    b.text(0.03, 0.04, "random (1% base rate)", color=t.ink2, fontsize=8)
    b.set_xlabel("Recall")
    b.set_ylabel("Precision")
    fig.text(0.995, -0.03, "Simulated: 1,000 positives among 100,000 cases", color=t.muted,
             fontsize=7, ha="right")
    fig.tight_layout(w_pad=2.5)
    save(fig, "roc-pr", t)


def _gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def _gelu_grad(x):
    phi = np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return cdf + x * phi


def fig_activations(t: Theme) -> None:
    x = np.linspace(-4, 4, 801)
    sig = 1 / (1 + np.exp(-x))
    th = np.tanh(x)
    relu = np.maximum(0, x)
    panels = [
        ("Sigmoid", sig, sig * (1 - sig), (-0.15, 1.15), "derivative peaks at 0.25"),
        ("Tanh", th, 1 - th ** 2, (-1.2, 1.2), "derivative peaks at 1.0"),
        ("ReLU", relu, (x > 0).astype(float), (-0.6, 3.2), "gradient is exactly 1 for x > 0"),
        ("GELU", _gelu(x), _gelu_grad(x), (-0.6, 3.2), "smooth; slightly negative near −0.75"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    for ax, (name, f, df, ylim, note) in zip(axes.flat, panels):
        style(ax, t)
        ax.set_title(name)
        ax.axhline(0, color=t.axis, lw=0.8)
        ax.plot(x, f, color=t.s1, lw=LW, label="function")
        ax.plot(x, df, color=t.s2, lw=LW, label="derivative")
        ax.set_xlim(-4, 4)
        ax.set_ylim(*ylim)
        ax.text(-3.85, ylim[1] - (ylim[1] - ylim[0]) * 0.06, note, color=t.ink2, fontsize=7.5, va="top")
    sig_ax = axes.flat[0]
    sig_ax.text(3.9, 0.92, "f", color=t.ink2, fontsize=8, ha="right", va="top")
    sig_ax.text(0.25, 0.3, "f′", color=t.ink2, fontsize=8)
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False, fontsize=8,
               labelcolor=t.ink2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.5, w_pad=2)
    save(fig, "activations", t)


def _run(update, start, steps):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    state = {}
    for k in range(1, steps + 1):
        p = update(p, state, k)
        path.append(p.copy())
    return np.array(path)


def fig_optimizers(t: Theme) -> None:
    curv = np.array([1.0, 18.0])  # f(x, y) = 0.5 * (x^2 + 18 y^2): a long, narrow valley

    def grad(p):
        return curv * p

    def sgd(p, s, k, lr=0.1):
        return p - lr * grad(p)

    def momentum(p, s, k, lr=0.03, beta=0.85):
        v = s.get("v", np.zeros(2))
        v = beta * v + grad(p)
        s["v"] = v
        return p - lr * v

    def adam(p, s, k, lr=0.35, b1=0.9, b2=0.999, eps=1e-8):
        g = grad(p)
        m = b1 * s.get("m", np.zeros(2)) + (1 - b1) * g
        v = b2 * s.get("v", np.zeros(2)) + (1 - b2) * g * g
        s["m"], s["v"] = m, v
        mh, vh = m / (1 - b1 ** k), v / (1 - b2 ** k)
        return p - lr * mh / (np.sqrt(vh) + eps)

    start = (-4.6, 1.15)
    steps = 30
    runs = [("SGD", sgd, t.s1), ("SGD + momentum", momentum, t.s2), ("Adam", adam, t.s3)]
    gx, gy = np.meshgrid(np.linspace(-5, 1.2, 300), np.linspace(-1.5, 1.6, 300))
    Z = 0.5 * (curv[0] * gx ** 2 + curv[1] * gy ** 2)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    style(ax, t, grid=False)
    ax.contour(gx, gy, Z, levels=[0.5, 2, 4.5, 8, 12.5, 18, 24.5], colors=[t.grid if t.name == "dark" else t.axis],
               linewidths=0.8)
    for name, upd, col in runs:
        path = _run(upd, start, steps)
        ax.plot(path[:, 0], path[:, 1], color=col, lw=LW, label=name)
        ax.plot(path[-1, 0], path[-1, 1], marker="o", ms=6, color=col, mec=t.surface, mew=1.5, zorder=5)
    ax.plot(*start, marker="o", ms=6, color=t.ink, mec=t.surface, mew=1.5, zorder=6)
    ax.text(start[0] + 0.05, start[1] + 0.12, "start", color=t.ink2, fontsize=8)
    ax.plot(0, 0, marker="+", ms=9, color=t.ink, mew=1.4)
    ax.text(0.1, 0.22, "minimum", color=t.ink2, fontsize=8)
    ax.set_xlim(-5, 1.2)
    ax.set_ylim(-1.5, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("weight 1 (flat direction)")
    ax.set_ylabel("weight 2 (steep)")
    ax.set_title(f"{steps} steps on a long, narrow valley")
    legend(ax, t, loc="lower left")
    fig.tight_layout()
    save(fig, "optimizer-paths", t)


def fig_lr_schedule(t: Theme) -> None:
    total, warm, peak, floor = 100_000, 2_000, 3.0, 0.3  # in units of 1e-4
    s = np.linspace(0, total, 2001)
    lr = np.where(
        s < warm,
        peak * s / warm,
        floor + 0.5 * (peak - floor) * (1 + np.cos(np.pi * (s - warm) / (total - warm))),
    )
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    style(ax, t)
    ax.plot(s, lr, color=t.s1, lw=LW)
    ax.set_xlim(0, total)
    ax.set_ylim(0, 3.4)
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000], ["0", "25K", "50K", "75K", "100K"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Learning rate ($\times 10^{-4}$)")
    ax.set_title("Linear warmup, then cosine decay")
    ax.annotate("warmup: 0 → peak\nin 2K steps", xy=(warm, peak), xytext=(9_000, 3.05),
                color=t.ink2, fontsize=8, arrowprops=dict(arrowstyle="-", color=t.axis, lw=0.8))
    ax.text(50_000, 1.85, "cosine decay", color=t.ink2, fontsize=8)
    ax.text(99_000, floor + 0.38, "ends at 10% of peak", color=t.ink2, fontsize=8, ha="right")
    fig.tight_layout()
    save(fig, "lr-schedule", t)


def fig_causal_attention(t: Theme) -> None:
    """Illustrative weights: a seeded score matrix with a mild recency bias, then the causal mask."""
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 1.1, (n, n))
    scores[2, 1] += 2.2  # "sat" looks back at "cat"
    scores[5, 3] += 1.6  # "mat" looks back at "on"
    scores[4, 0] += 1.8  # second "the" looks back at the first
    scores += -0.35 * np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("seq", t.seq)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    style(ax, t, grid=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Causal self-attention, one head", pad=34)
    gap = 0.04
    for i in range(n):
        for j in range(n):
            color = t.masked if mask[i, j] else cmap(w[i, j])
            ax.add_patch(Rectangle((j + gap, n - 1 - i + gap), 1 - 2 * gap, 1 - 2 * gap,
                                   facecolor=color, edgecolor="none"))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5, tokens)
    ax.set_yticks(np.arange(n) + 0.5, tokens[::-1])
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Key: the token being looked at", labelpad=8)
    ax.set_ylabel("Query: the token doing the looking")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t.axis, labelcolor=t.ink2, labelsize=8, length=2)
    cb.set_label("attention weight (each row sums to 1)", color=t.ink2, fontsize=8)
    handles = [Patch(facecolor=t.masked, edgecolor="none", label="masked: future token (score set to −∞)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False,
              fontsize=8, labelcolor=t.ink2)
    fig.text(0.02, 0.005, "Illustrative weights", color=t.muted, fontsize=7)
    fig.tight_layout()
    save(fig, "causal-attention", t)


def fig_diffusion_schedules(t: Theme) -> None:
    T = 1000
    tt = np.arange(T + 1)
    betas = np.linspace(1e-4, 0.02, T)
    abar_lin = np.concatenate([[1.0], np.cumprod(1 - betas)])
    s = 0.008
    f = np.cos(((tt / T) + s) / (1 + s) * np.pi / 2) ** 2
    abar_cos = f / f[0]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    style(ax, t)
    ax.plot(tt, abar_lin, color=t.s1, lw=LW, label="linear β schedule (DDPM)")
    ax.plot(tt, abar_cos, color=t.s2, lw=LW, label="cosine schedule")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("ᾱ: share of signal variance left")
    ax.set_title("How fast each schedule destroys the image")
    ax.text(430, 0.28, "linear: most signal gone\nby the middle", color=t.ink2, fontsize=8)
    ax.text(560, abar_cos[560] + 0.08, "cosine: degrades more evenly", color=t.ink2, fontsize=8)
    legend(ax, t, loc="upper right")
    fig.tight_layout()
    save(fig, "diffusion-schedules", t)


def fig_temperature(t: Theme) -> None:
    words = ["mat", "sofa", "floor", "bed", "roof", "moon"]
    logits = np.array([3.0, 2.2, 1.8, 1.2, 0.3, -0.5])
    temps = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    x = np.arange(len(words))
    for ax, T in zip(axes, temps):
        p = np.exp(logits / T)
        p = p / p.sum()
        style(ax, t)
        ax.grid(True, axis="y", color=t.grid, linewidth=0.6)
        ax.grid(False, axis="x")
        ax.bar(x, p, width=0.55, color=t.s1, edgecolor="none")
        ax.set_xticks(x, words, rotation=0, fontsize=7.5)
        ax.set_ylim(0, 1.0)
        label = {0.5: "T = 0.5 · sharper", 1.0: "T = 1 · the model's distribution", 2.0: "T = 2 · flatter"}[T]
        ax.set_title(label, fontsize=9)
        ax.text(0, p[0] + 0.03, f"{p[0]:.0%}", color=t.ink2, fontsize=8, ha="center")
    axes[0].set_ylabel("Probability of next word")
    fig.text(0.01, -0.03, "Next word after “The cat sat on the …”, from the same logits divided by T",
             color=t.muted, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "temperature", t)


FIGURES = [
    fig_bias_variance, fig_l1_l2, fig_roc_pr, fig_activations, fig_optimizers,
    fig_lr_schedule, fig_causal_attention, fig_diffusion_schedules, fig_temperature,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font-dir", default=os.environ.get("AIML_FONT_DIR"))
    args = parser.parse_args()
    setup_fonts(args.font_dir)
    for theme in (LIGHT, DARK):
        for make in FIGURES:
            make(theme)


if __name__ == "__main__":
    main()
