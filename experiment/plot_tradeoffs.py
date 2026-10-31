"""Draw trade-off curves from per-job sweep tables (sweep.json). Runs no models.

Every ratio is read from the table, which computed it inside one (prompt style, Slurm job)
group. This script never recomputes or averages ratios, and it refuses a series that mixes
groups. Output: PNG + SVG per (benchmark, prompt style) and a figures.json manifest.

Axes: x is the ratio column, y is the quality delta, so a reference sits at (1, 0).
Ratio directions (taken from the column names, never inverted):
  *_speedup_vs_*  = reference latency / arm latency  (above 1: arm is faster)
  *_arm_over_*    = arm / reference memory           (above 1: arm uses more memory)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REQUIRED = (
    "prompt_style", "slurm_job_id", "sweep", "method", "role", "arm", "benchmark",
    "kv_retention", "kv_policy", "draft_retention", "draft_policy",
    "latency_speedup_vs_vispec_ref", "live_kv_arm_over_samepolicy_1_0",
    "post_compaction_peak_gib", "speedup_vs_ar", "quality_delta_vs_vispec_ref",
)
# Optional columns. sweep_table does not write them today; a panel reads them if a table has them.
POST_COMPACTION_RATIO = "post_compaction_peak_arm_over_samepolicy_1_0"
BERTSCORE_DELTA = "bertscore_f1_delta_vs_vispec_ref"

QUALITY_AXIS = {
    "textvqa": "TextVQA accuracy delta vs ViSpec ref (VQA points)",
    "gqa": "GQA exact-match delta vs ViSpec ref (points)",
    "mmvet": "MM-Vet judged score delta vs ViSpec ref (points)",
    "coco_caption": "CIDEr-D x100 delta vs ViSpec ref (points)",
    "reverse_summary": "ROUGE-1 F1 delta vs ViSpec ref (points)",
}
BERT_AXIS = "BERTScore F1 delta vs ViSpec ref (points)\n(truncated: ignores later pages)"
# Okabe-Ito colour-blind-safe palette.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")
MARKERS = {"vispec_speculative": "o", "autoregressive": "s"}
NAMES = {"vispec_speculative": "ViSpec", "autoregressive": "AR"}

SPEEDUP_LABEL = "latency speedup = ViSpec-ref latency / arm latency (above 1: faster)"
# (panel id, title, x column, x label)
PANELS = (
    ("latency_vs_vispec", "Latency speedup vs ViSpec ref", "latency_speedup_vs_vispec_ref", SPEEDUP_LABEL),
    ("live_kv", "Live KV vs same-policy 1.0 arm", "live_kv_arm_over_samepolicy_1_0",
     "live KV = arm / same-policy 1.0 arm (above 1: more memory)"),
    ("post_compaction", "Post-compaction peak vs same-policy 1.0 arm", POST_COMPACTION_RATIO,
     "post-compaction peak = arm / same-policy 1.0 arm (above 1: more memory)"),
    ("speedup_vs_ar", "Speedup vs AR", "speedup_vs_ar",
     "speedup = AR-ref latency / arm latency (above 1: faster)"),
)


def group_of(row):
    return (row["prompt_style"], str(row["slurm_job_id"]))


def series_key(row):
    """One series = one (group, method, sweep, policy). The group is always part of the key."""
    return group_of(row) + (row["method"], row["sweep"], row["kv_policy"], row["draft_policy"])


def check_single_group(series_name, rows):
    groups = {group_of(row) for row in rows}
    if len(groups) > 1:
        raise ValueError(
            f"Refusing to combine ratios across groups in series {series_name!r}: "
            f"{sorted(groups)}. Each (prompt style, Slurm job) has its own references."
        )


def build_series(rows, key=series_key):
    """Group rows into series by `key`, refusing any series that spans several groups."""
    series = {}
    for row in rows:
        series.setdefault(key(row), []).append(row)
    for name, members in series.items():
        check_single_group(name, members)
    return series


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tables(paths):
    rows, sources, seen = [], [], {}
    for path in paths:
        path = Path(path)
        table = json.loads(path.read_text())
        missing = [c for c in REQUIRED if table["rows"] and c not in table["rows"][0]]
        if missing:
            raise ValueError(f"{path}: not a sweep_table sweep.json, missing columns {missing}")
        sources.append({"path": str(path), "sha256": sha256(path), "rows": len(table["rows"])})
        for index, row in enumerate(table["rows"]):
            ident = group_of(row) + (row["benchmark"], row["method"], row["sweep"], row["arm"])
            if ident in seen:
                raise ValueError(f"{path}: row {ident} already read from {seen[ident]}")
            seen[ident] = str(path)
            rows.append(dict(row, _source=str(path), _index=index))
    return rows, sources


def budget_label(row):
    kv, draft = row["kv_retention"], row["draft_retention"]
    if kv is not None and draft is not None:
        return f"kv {kv:g}/dr {draft:g}"
    value = kv if kv is not None else draft
    return "" if value is None else f"{value:g}"


def series_label(key, multi_job):
    style, job, method, sweep, kv_policy, draft_policy = key
    policy = "/".join(p for p in (kv_policy, draft_policy) if p)
    label = f"{NAMES[method]} {sweep}" + (f" ({policy})" if policy else "")
    return f"{label} [job {job}]" if multi_job else label


def sort_budget(row):
    for field in ("kv_retention", "draft_retention"):
        if row[field] is not None:
            return row[field]
    return 0.0


def reference_point(row, xcol, ycol):
    """Where a reference row sits. Memory ratios are 1 against the ViSpec reference only."""
    if xcol in ("live_kv_arm_over_samepolicy_1_0", POST_COMPACTION_RATIO):
        # A reference is not an arm of a sweep, so it has no same-policy ratio; the ViSpec
        # reference is the unit by definition and the AR reference has no point here.
        if row["method"] != "vispec_speculative" or xcol == POST_COMPACTION_RATIO:
            return None, None
        return 1.0, row.get(ycol)
    return row.get(xcol), row.get(ycol)


def draw_panel(ax, spec, ycol, ylabel, series, references, jobs, points, fig_id):
    pid, title, xcol, xlabel = spec
    plotted = 0
    multi_job = len(jobs) > 1
    for index, (key, members) in enumerate(sorted(series.items(), key=lambda kv: str(kv[0]))):
        pts = sorted((r for r in members if r.get(xcol) is not None and r.get(ycol) is not None),
                     key=sort_budget)
        if not pts:
            continue
        color = COLORS[index % len(COLORS)]
        xs, ys = [r[xcol] for r in pts], [r[ycol] for r in pts]
        ax.plot(xs, ys, marker=MARKERS[key[2]], color=color, label=series_label(key, multi_job),
                linewidth=1.2, markersize=5)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(budget_label(r), (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color)
            points.append({
                "figure": fig_id, "panel": pid, "series": series_label(key, True),
                "style": key[0], "job": key[1], "method": key[2], "sweep": key[3],
                "kv_policy": key[4], "draft_policy": key[5], "arm": r["arm"],
                "budget_label": budget_label(r), "x_column": xcol, "x": x,
                "y_column": ycol, "y": y, "source": r["_source"], "row_index": r["_index"],
            })
            plotted += 1
    for (job, method), row in sorted(references.items()):
        x, y = reference_point(row, xcol, ycol)
        if x is None or y is None:
            continue
        name = f"{NAMES[method]} reference"
        ax.plot([x], [y], marker="*", markersize=11, linestyle="none",
                color="black" if method == "vispec_speculative" else "#777777",
                label=name + (f" [job {job}]" if multi_job else ""))
        points.append({
            "figure": fig_id, "panel": pid, "series": name, "style": row["prompt_style"],
            "job": job, "method": method, "sweep": "reference", "kv_policy": None,
            "draft_policy": None, "arm": "reference", "budget_label": "ref",
            "x_column": xcol, "x": x, "y_column": ycol, "y": y,
            "source": row["_source"], "row_index": row["_index"],
        })
        plotted += 1
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if plotted == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
        ax.axvline(1, color="#bbbbbb", linewidth=0.8, zorder=0)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.25)
    return plotted


def plot_all(rows, output_dir, points):
    figures = []
    for benchmark, style in sorted({(r["benchmark"], r["prompt_style"]) for r in rows}):
        sub = [r for r in rows if r["benchmark"] == benchmark and r["prompt_style"] == style]
        series = build_series([r for r in sub if r["role"] == "arm"])
        jobs = sorted({str(r["slurm_job_id"]) for r in sub})
        references = {(str(r["slurm_job_id"]), r["method"]): r for r in sub if r["role"] == "reference"}
        quality_label = QUALITY_AXIS.get(benchmark, f"{benchmark} quality delta vs ViSpec ref (points)")
        panels = [(spec, "quality_delta_vs_vispec_ref", quality_label) for spec in PANELS]
        if benchmark == "reverse_summary":
            # W4-R10: BERTScore is truncated and ignores later pages, so it only ever gets its own panel.
            panels.append((("bertscore", "BERTScore F1 (separate panel, truncated)",
                            "latency_speedup_vs_vispec_ref", SPEEDUP_LABEL), BERTSCORE_DELTA, BERT_AXIS))
        nrows = (len(panels) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.6 * nrows), squeeze=False)
        fig_id = f"{benchmark}_{style}"
        counts = {}
        for ax, (spec, ycol, ylabel) in zip(axes.flat, panels):
            counts[spec[0]] = draw_panel(ax, spec, ycol, ylabel, series, references, jobs, points, fig_id)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{benchmark} / {style} prompts / job{'s' if len(jobs) > 1 else ''} {', '.join(jobs)}"
                     " (every series stays inside one job's references)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        files = []
        for ext in ("png", "svg"):
            fig.savefig(output_dir / f"{fig_id}.{ext}", dpi=150)
            files.append(f"{fig_id}.{ext}")
        plt.close(fig)
        figures.append({"id": fig_id, "benchmark": benchmark, "prompt_style": style,
                        "jobs": jobs, "files": files, "points_per_panel": counts})
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweeps", nargs="+", required=True, type=Path, help="sweep.json files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New folder, for example experiment/sweeps/<UTC>_tradeoffs")
    args = parser.parse_args(argv)
    rows, sources = load_tables(args.sweeps)
    if not rows:
        raise ValueError("The sweep tables hold no rows")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    points = []
    figures = plot_all(rows, args.output_dir, points)
    manifest = {"command": " ".join(sys.argv), "sources": sources, "figures": figures, "points": points}
    (args.output_dir / "figures.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(figures)} figures and {len(points)} points to {args.output_dir}")


if __name__ == "__main__":
    main()
