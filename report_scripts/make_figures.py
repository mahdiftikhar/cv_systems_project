"""Draw the report's figures from report_scripts/figure_data.json.

Usage: python -I report_scripts/make_figures.py   (after make_data.py)
Writes PDF files to report/figures/. Every plotted value comes from figure_data.json, which
make_data.py builds from the saved result files. Plots are drawn at the width they print at
(a CVPR column is 3.25 in, the full text width 6.875 in) and included at scale 1.

The visual style follows the helpers of https://github.com/ppriyank/plot (style.py, line.py,
scatter.py, hist.py and plot_add.py): a light blue-grey grid, black left and bottom spines,
markers with a dark border, dash-dot reference lines, shaded bands with dash-dot edges, white
gaps between stacked bars and value labels on bars. Line widths and sizes are scaled down to
print size. The colours stay on the palette that passed the colour-vision check.
"""
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(HERE, "figure_data.json"), encoding="utf-8") as h:
    D = json.load(h)

COL, FULL = 3.25, 6.875
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#A8BAC4"          # grid colour of ppriyank/plot (style.grid_data)
SHADE = "black"           # shaded bands, alpha 0.15 as in plot_add.V_line_Section examples
AR, VIS, HIV = "#8a8985", "#2a78d6", "#eb6834"
AQUA, VIOLET = "#1baf7a", "#6250d6"
PREFILL = "#3b3a37"
BORDER = dict(edgecolor="black", linewidth=0.5)   # marker border, scatter_boder_color="black"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": "black",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)


def ref_h(ax, y):
    ax.axhline(y, color="black", linewidth=0.7, linestyle="-.", zorder=2)


def ref_v(ax, x):
    ax.axvline(x, color="black", linewidth=0.7, linestyle="-.", zorder=2)


def band_v(ax, x0, x1):
    ax.axvspan(x0, x1, color=SHADE, alpha=0.08, linewidth=0, zorder=0)
    for x in (x0, x1):
        ax.axvline(x, color=SHADE, alpha=0.35, linewidth=0.6, linestyle="-.", zorder=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# ------------------------------------------------------------------ teaser: where the time goes
def teaser():
    panels = [
        ("A short answer", D["teaser_single"]["short textvqa"]),
        ("A paragraph answer", D["teaser_single"]["long textvqa"]),
        ("A summary of five to six manga pages", D["teaser_pages"][sorted(D["teaser_pages"])[0]]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.05), gridspec_kw={"wspace": 0.42})
    names = [("ar", "Autoregressive", AR), ("vispec", "ViSpec", VIS), ("hivis", "HiViS", HIV)]
    for ax, (title, data) in zip(axes, panels):
        total_ar = data["ar"][1]
        for i, (key, label, color) in enumerate(names):
            ttft, lat = data[key]
            y = 2 - i
            ax.barh(y, ttft, height=0.66, color=PREFILL, edgecolor="white", linewidth=1.0, zorder=3)
            ax.barh(y, lat - ttft, left=ttft, height=0.66, color=color, edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3)
            text = f"{lat:.1f} s" if key == "ar" else f"{lat:.1f} s,  {total_ar / lat:.2f}$\\times$"
            ax.text(lat + 0.025 * total_ar, y, text, va="center", ha="left", color=INK, fontsize=6.5)
        ax.set_yticks([2, 1, 0], [n[1] for n in names])
        ax.set_xlim(0, total_ar * 1.38)
        ax.set_xlabel("Time per sample (s)")
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        grid(ax, "x")
    handles = [Rectangle((0, 0), 1, 1, color=PREFILL), Rectangle((0, 0), 1, 1, color=AR), Rectangle((0, 0), 1, 1, color=VIS), Rectangle((0, 0), 1, 1, color=HIV)]
    fig.legend(handles, ["Prompt processing, up to the first token", "Decoding, autoregressive", "Decoding, ViSpec", "Decoding, HiViS"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.27))
    save(fig, "teaser.pdf")


# ------------------------------------------------------------------ speedup against the prompt share
def amdahl():
    pts = D["amdahl"]
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    pages = [p["ar_ttft_share"] for p in pts if p["setting"] == "pages"]
    band_v(ax, min(pages) - 0.015, max(pages) + 0.015)
    f = np.linspace(0.0, 0.97, 200)
    for s_ in (1.5, 2.0, 2.5):
        ax.plot(f, 1 / (f + (1 - f) / s_), color=GRID, linewidth=1.3, alpha=0.8, zorder=1)
    for p in sorted(pts, key=lambda p: p["model"]):
        color = VIS if p["method"] == "ViSpec" else HIV
        marker = "o" if p["model"] == "3B" else "s"
        ax.scatter(p["ar_ttft_share"], p["speedup"], s=20, marker=marker, facecolor=color, alpha=0.85, zorder=3, **BORDER)
    ref_h(ax, 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.8, 2.65)
    ax.set_xlabel("Share of autoregressive time spent before the first token")
    ax.set_ylabel("Speedup over autoregressive decoding")
    grid(ax, "both")
    for x, y, text, ha in ((0.10, 2.06, "long answers", "left"), (0.265, 1.52, "short captions", "right"),
                           (0.335, 2.45, "manga windows", "center"), (0.875, 1.04, "short answers", "center")):
        ax.text(x, y, text, fontsize=6.5, color=INK, ha=ha)
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=VIS, markeredgecolor="black", markeredgewidth=0.5, markersize=4.5, label="ViSpec, 3B"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor=VIS, markeredgecolor="black", markeredgewidth=0.5, markersize=4.5, label="ViSpec, 7B"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor=HIV, markeredgecolor="black", markeredgewidth=0.5, markersize=4.5, label="HiViS, 7B"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "amdahl.pdf")


# ------------------------------------------------------------------ margins at the first difference
def margins():
    m = D["margins"]
    groups = [("3B", "3B, ViSpec", VIS), ("7B", "7B, single images", HIV), ("7Bpages", "7B, manga windows", AQUA)]
    values = sorted({round(x["bf16"], 4) for x in m})
    xs = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    cut = max(i for i, v in enumerate(values) if v <= 0.5) + 0.5
    ax.axvspan(cut, len(values) - 0.4, color=SHADE, alpha=0.08, linewidth=0, zorder=0)
    bottom = np.zeros(len(values))
    for key, label, color in groups:
        c = Counter(round(x["bf16"], 4) for x in m if x["group"] == key)
        h = np.array([c.get(v, 0) for v in values])
        keep = h > 0
        ax.bar(xs[keep], h[keep], width=0.74, bottom=bottom[keep], color=color, edgecolor="white", linewidth=0.8, alpha=0.95, label=label, zorder=3)
        bottom += h
    for x, b in zip(xs, bottom):
        ax.text(x, b + 5, f"{int(b)}", ha="center", va="bottom", fontsize=6, color=INK)
    ref_v(ax, cut)
    ax.text(cut + 0.12, bottom.max() * 0.42, "near-tie\nthreshold", fontsize=6, color=INK2, va="top")
    ax.set_xticks(xs, [f"{v:g}" for v in values])
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_xlabel("Top-two logit margin of the target in bf16 at the first differing token")
    ax.set_ylabel("First differences")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.tick_params(axis="x", length=0)
    grid(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.86, 1.0))
    save(fig, "margins.pdf")


# ------------------------------------------------------------------ visual sparsity
def sparsity():
    single = D["sparsity_single"]
    pages = D["sparsity_pages"]
    pol_color = {"spatial": VIOLET, "attention": AQUA}
    shape = {"draft": "^", "kv": "o", "combined": "D"}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.55), gridspec_kw={"wspace": 0.42, "width_ratios": [1.3, 1, 1, 1]})
    ax = axes[0]
    for r in single:
        ret = r["kv_retention"] if r["kv_retention"] is not None else r["draft_retention"]
        if ret >= 1.0:
            continue
        color = pol_color[r["kv_policy"] or r["draft_policy"]]
        ax.scatter(r["live_kv_arm_over_samepolicy_1_0"], r["latency_speedup_vs_samepolicy_1_0"], s=12, marker=shape[r["sweep"]],
                   facecolor=color, alpha=0.85, zorder=3, **BORDER)
    for r in pages:
        color = pol_color[r["kv_policy"] or r["draft_policy"] or "spatial"]
        ax.scatter(r["live_kv_mean_mib"] / r["ref_live_kv_mib"], r["latency_speedup_vs_vispec_ref"], s=16, marker=shape[r["sweep"]],
                   facecolor="white", edgecolor=color, linewidth=1.0, zorder=4)
    ref_h(ax, 1.0)
    ax.set_xlim(0.2, 1.08)
    ax.set_ylim(0.82, 1.16)
    ax.set_xlabel("Live KV cache, arm over full cache")
    ax.set_ylabel("Speedup over the full cache")
    ax.set_title("Memory and latency", loc="left")
    grid(ax, "both")

    panels = [("textvqa", "Short TextVQA answers"), ("coco_caption", "Short COCO captions"), (None, "Manga window summaries")]
    for ax, (bench, title) in zip(axes[1:], panels):
        for policy in ("spatial", "attention"):
            if bench is None:
                rows = [r for r in pages if r["sweep"] == "kv" and r["kv_policy"] == policy]
                ref = [r for r in pages if r["sweep"] == "kv"][0]
                base = ref["quality"] - ref["quality_delta_vs_vispec_ref"]
            else:
                rows = [r for r in single if r["sweep"] == "kv" and r["prompt_style"] == "short" and r["benchmark"] == bench and r["kv_policy"] == policy]
                base = [r for r in rows if r["kv_retention"] == 1.0][0]["quality"]
            pts = {r["kv_retention"]: 100 * (r["quality"] / base - 1) for r in rows}
            pts.setdefault(1.0, 0.0)
            xs = sorted(pts)
            ys = [pts[x] for x in xs]
            ax.plot(xs, ys, color=pol_color[policy], linewidth=1.6, zorder=3)
            ax.scatter(xs, ys, s=14, facecolor=pol_color[policy], alpha=0.6, zorder=4, **BORDER)
        ref_h(ax, 0)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1"])
        ax.set_xlim(0.2, 1.05)
        ax.set_ylim(-15, 5)
        ax.set_xlabel("Image entries kept")
        ax.set_title(title, loc="left")
        grid(ax)
    axes[1].set_ylabel("Quality change (%)")
    handles = [
        Line2D([], [], color=VIOLET, linewidth=1.6, label="spatial policy"),
        Line2D([], [], color=AQUA, linewidth=1.6, label="attention policy"),
        Line2D([], [], marker="^", linestyle="", markerfacecolor=INK2, markeredgecolor="black", markeredgewidth=0.5, markersize=4, label="draft pruning"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor="black", markeredgewidth=0.5, markersize=4, label="KV eviction"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor=INK2, markeredgecolor="black", markeredgewidth=0.5, markersize=3.5, label="both"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=INK2, markeredgecolor="black", markeredgewidth=0.5, markersize=4, label="single image (filled)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=1.0, markersize=4, label="manga window (open)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=1.1)
    save(fig, "sparsity.pdf")


# ------------------------------------------------------------------ bytes read by one decoding step
def cost():
    c = D["cost"]
    pl = D["prompt_lengths"]
    n = np.logspace(2, 6, 300)
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    band_v(ax, pl["shortest"], pl["longest"])
    ax.text(np.sqrt(pl["shortest"] * pl["longest"]), 0.006, "prompts of\nthis report", fontsize=6, color=INK2, ha="center", va="bottom")
    for key, label, color in (("3b", "3B", VIS), ("7b", "7B", HIV)):
        w = c[key]["weight_bytes_per_step"] / 2**30
        kv = n * c[key]["kv_bytes_per_token"] / 2**30
        ax.plot(n, np.full_like(n, w), color=color, linewidth=1.4, linestyle="-.", zorder=3)
        ax.plot(n, kv, color=color, linewidth=1.8, zorder=3)
        eq = c[key]["weight_bytes_per_step"] / c[key]["kv_bytes_per_token"]
        ax.scatter([eq], [w], s=22, facecolor=color, zorder=4, **BORDER)
        ax.text(1.2e2, w * 1.2, f"{label} language-model weights", color=INK, fontsize=6.5)
    ax.text(2.2e4, 0.55, "KV cache", color=INK, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(0.004, 60)
    ax.set_xlabel("Tokens in the KV cache")
    ax.set_ylabel("GiB read per decoding step")
    grid(ax, "both")
    save(fig, "cost.pdf")


teaser()
amdahl()
margins()
sparsity()
cost()
print("figures written to", OUT)
