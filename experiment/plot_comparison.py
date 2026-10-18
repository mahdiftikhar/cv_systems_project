"""Plot saved metrics without loading models or running evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cv-systems-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BENCHMARKS = ("textvqa", "gqa", "mmvet")
NAMES = ("TextVQA\n100 questions", "GQA\n100 questions", "MM-Vet\n218 questions")
COLORS = ("#425979", "#009E83")
LABELS = ("00 · Standard", "01 · ViSpec")
CAUTION = (
    "Observed timings · Different physical H100s / CPU bindings · One run per method\n"
    "51/418 token sequences differ · 14 capped responses per method · Quality not validated"
)


def load_report(directory):
    summary = json.loads((directory / "summary.json").read_text())
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines() if line.strip()]
    if summary["evaluation_exit_code"] != 0:
        raise ValueError("Plots require completed evaluations")
    return summary, samples


def agreement(samples):
    maps = [
        {(r["benchmark"], r["question_id"], r["choice"], r["turn"]): r for r in group}
        for group in samples
    ]
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Question/choice/turn coverage differs")
    result = {}
    for benchmark in BENCHMARKS:
        matched = total = 0
        for key, a in maps[0].items():
            if key[0] != benchmark:
                continue
            b = maps[1][key]
            if a["measurement"]["input_ids_sha256"] != b["measurement"]["input_ids_sha256"]:
                raise ValueError("Prompt fingerprints differ")
            ids = []
            for row in (a, b):
                tokens = row["measurement"]["output_token_ids"]
                eos = row["measurement"]["eos_token_id"]
                ids.append(tokens[: tokens.index(eos) + 1] if eos in tokens else tokens)
            total += 1
            matched += ids[0] == ids[1]
        result[benchmark] = {"matched": matched, "total": total}
    return result


def style(ax, title, unit):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xticks(range(3), NAMES, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3e8ee", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ccd4dd")
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_ylim(bottom=0)


def bars(ax, title, unit, values, decimals, ratios=False):
    style(ax, title, unit)
    for method in range(2):
        locations = [x + (-0.19 if method == 0 else 0.19) for x in range(3)]
        rects = ax.bar(locations, values[method], width=0.33, color=COLORS[method])
        ax.bar_label(rects, fmt=f"%.{decimals}f", padding=4, fontsize=9)
    maximum = max(v for method in values for v in method)
    ax.set_ylim(0, maximum * (1.38 if ratios else 1.22))
    if ratios:
        for x, (a, b) in enumerate(zip(*values)):
            ax.text(x, max(a, b) + maximum * 0.16, f"{a / b:.2f}×", ha="center", fontsize=11, fontweight="bold", color=COLORS[1])


def agreement_plot(ax, counts):
    style(ax, "Exact token sequence agreement", "Percent of paired responses")
    matched = [100 * counts[b]["matched"] / counts[b]["total"] for b in BENCHMARKS]
    ax.bar(range(3), matched, width=0.55, color=COLORS[1])
    ax.bar(range(3), [100 - v for v in matched], bottom=matched, width=0.55, color="#dbe1e9")
    for x, b in enumerate(BENCHMARKS):
        c = counts[b]
        ax.text(x, matched[x] / 2, f'{c["matched"]}/{c["total"]}\n{matched[x]:.1f}%', ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 108)
    ax.text(0.5, 0.97, "Includes agreement on capped prefixes", transform=ax.transAxes, ha="center", fontsize=8, color="#576577")


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="00 report directory")
    parser.add_argument("--draft", type=Path, required=True, help="01 report directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [args.original, args.draft]
    loaded = [load_report(path) for path in reports]
    summaries = [item[0] for item in loaded]
    counts = agreement([item[1] for item in loaded])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#263549", "axes.labelcolor": "#42546a", "xtick.color": "#42546a", "ytick.color": "#42546a", "pdf.fonttype": 42})
    metrics = [
        ("latency", "Mean generation latency ↓", "Seconds / question", "wall_time_s", 1, 2, True),
        ("ttft", "Mean time to first token ↓", "Seconds / question", "ttft_s", 1, 3, False),
        ("throughput", "Output throughput ↑", "Returned token IDs / second", "throughput_tokens_per_second", 1, 1, False),
        ("tpot", "Mean time per output token ↓", "Milliseconds / output token", "time_per_output_token_s", 1000, 2, False),
        ("gpu_memory", "Peak allocated GPU memory ↓", "PyTorch allocated GiB", "gpu_memory_by_visible_device", 1, 2, False),
    ]
    values = {}
    for name, title, unit, key, scale, decimals, ratios in metrics:
        values[name] = []
        for summary in summaries:
            series = []
            for benchmark in BENCHMARKS:
                value = summary["benchmarks"][benchmark][key]
                if key == "gpu_memory_by_visible_device":
                    value = value["0"]["peak_allocated_bytes"] / 2**30
                elif isinstance(value, dict):
                    value = value["mean"]
                series.append(value * scale)
            values[name].append(series)

    handles = [Patch(color=color, label=label) for color, label in zip(COLORS, LABELS)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, metric in zip(axes.flat, metrics):
        name, title, unit, _, _, decimals, ratios = metric
        bars(ax, title, unit, values[name], decimals, ratios)
    agreement_plot(axes.flat[-1], counts)
    fig.suptitle("ViSpec reduces generation time; first-token latency stays similar", x=0.05, ha="left", fontsize=19, fontweight="bold", y=0.97)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.925), ncol=2, frameon=False)
    fig.text(0.05, 0.89, "LLaVA-1.6 Vicuna 7B · FP16 · Greedy · 1,024-token cap · H100 PCIe ×1 per job", fontsize=11)
    fig.text(0.05, 0.025, CAUTION, fontsize=10, color="#576577", linespacing=1.5)
    fig.subplots_adjust(top=0.81, bottom=0.17, left=0.065, right=0.98, wspace=0.3, hspace=0.48)
    save(fig, args.output_dir, "overview")

    for name, title, unit, _, _, decimals, ratios in metrics:
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        bars(ax, title, unit, values[name], decimals, ratios)
        fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
        fig.text(0.08, 0.035, CAUTION, fontsize=8, linespacing=1.5)
        fig.subplots_adjust(top=0.82, bottom=0.25, left=0.12, right=0.96)
        save(fig, args.output_dir, name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, benchmark, label in zip(axes, BENCHMARKS, NAMES):
        for method, (_, samples) in enumerate(loaded):
            times = sorted(r["wall_time_s"] for r in samples if r["benchmark"] == benchmark)
            x = [0] + times
            y = [0] + [(i + 1) / len(times) * 100 for i in range(len(times))]
            ax.step(x, y, where="post", color=COLORS[method], linewidth=2, label=LABELS[method])
        ax.set_title(label.replace("\n", " · "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation latency (s)")
        ax.set_ylabel("Questions completed within latency (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 102)
        ax.grid(color="#e3e8ee", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Generation latency across every measured question", fontsize=17, fontweight="bold", y=0.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), frameon=False, ncol=2)
    fig.text(0.06, 0.025, CAUTION + "\nEmpirical cumulative distributions; capped responses remain included.", fontsize=9, linespacing=1.4)
    fig.subplots_adjust(top=0.75, bottom=0.28, left=0.06, right=0.98, wspace=0.3)
    save(fig, args.output_dir, "latency_distributions")
    provenance = {
        "matplotlib_version": matplotlib.__version__,
        "benchmark_order": list(BENCHMARKS),
        "plotted_values": values,
        "token_agreement": counts,
        "sources": [{"path": str(path / filename), "sha256": hashlib.sha256((path / filename).read_bytes()).hexdigest()} for path in reports for filename in ("summary.json", "samples.jsonl")],
        "limitations": CAUTION,
    }
    (args.output_dir / "plotted_values.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved plots and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()
