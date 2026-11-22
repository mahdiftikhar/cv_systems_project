"""Collect every number the report prints, from the saved result files.

Usage: python -I report_scripts/make_data.py
Writes report/generated/numbers.tex (one LaTeX macro per number) and
report_scripts/figure_data.json (the values the figure script plots).

The run folders and analysis tables are git-ignored and live in the main checkout and its
worktrees, so this script runs only on the machine that holds them. Set PROJECT_ROOT to point it
at another checkout. The LLaVA values come from report_scripts/llava_teammates.json, whose
source note says where they were transcribed from.
"""
import glob
import json
import os
import statistics
import struct


HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(HERE), "report")
ROOT = os.environ.get("PROJECT_ROOT", "/home/aeternum/Research2/cv_systems_project")
SWEEPS = os.path.join(ROOT, "experiment", "sweeps")
WT = os.path.join(ROOT, ".claude", "worktrees")
CAP = 1024

W3 = {
    "long_kv": "20261105T192610Z_week3_long_kv_job398883",
    "long_draft": "20261105T192610Z_week3_long_draft_job398884",
    "long_combined": "20261105T192610Z_week3_long_combined_job398885",
    "short_all": "20261105T192610Z_week3_short_all_job398923",
}
W4 = "20261121T212041Z_week4_reverse_full"
Q7 = "20261121T233327Z_qwen7b_methods_full"
R7 = "20261122T112816Z_qwen7b_reverse_full"
DIAG_3B = [
    os.path.join(ROOT, "experiment/03_qwen_vispec/runs/20261019T054357Z_a06dd479abba_long_kDp191/diagnosis/20261019T060057Z/diagnosis.json"),
    os.path.join(ROOT, "experiment/03_qwen_vispec/runs/20261019T063023Z_a06dd479abba_short_ogIM31/diagnosis/20261019T085127Z/diagnosis.json"),
    os.path.join(SWEEPS, "20261121T211835Z_week4_reverse_comparisons/job402643_vispec_vs_ar/diagnosis/diagnosis.json"),
]
DIAG_7B = sorted(glob.glob(os.path.join(WT, "hivis/experiment/10_qwen7b_methods/runs/20261122T020622Z_def2e960f7f8_checks_diagnosis_JaXa78/checks/diagnose_*/diagnosis.json")))
DIAG_7B_PAGES = sorted(glob.glob(os.path.join(WT, "reverse7b/experiment/11_qwen7b_reverse/runs/20261122T104930Z_4d3aaf84b0fe_checks_diagnosis_CW5BgZ/checks/diagnose_*/diagnosis.json")))
FP32 = {
    "3b_single": sorted(glob.glob(os.path.join(SWEEPS, "week3_offset_oracle/*/offset_check.json"))),
    "3b_pages": sorted(glob.glob(os.path.join(WT, "integration/experiment/sweeps/week4_offset_oracle/*/offset_check.json"))),
    "7b_single": sorted(
        glob.glob(os.path.join(WT, "hivis/experiment/10_qwen7b_methods/runs/20261112T113151Z_1a0f282bc280_checks_port_dkKVdV/checks/fp32_*/offset_check.json"))
        + glob.glob(os.path.join(WT, "hivis/experiment/10_qwen7b_methods/runs/20261122T020622Z_def2e960f7f8_checks_diagnosis_JaXa78/checks/oracle_*/offset_check.json"))
    ),
    "7b_pages": sorted(glob.glob(os.path.join(WT, "reverse7b/experiment/11_qwen7b_reverse/runs/20261122T093629Z_4d3aaf84b0fe_checks_port_vS8sY0/checks/fp32_*/offset_check.json"))),
}
PROFILE = os.path.join(ROOT, "experiment/04_qwen_decode_profile/runs/20261019T230855Z_65078a0dbe9b_long_J4OYTT/profile")
MODELS = {"3b": os.path.join(ROOT, "models/Qwen2.5-VL-3B-Instruct"), "7b": os.path.join(ROOT, "models/Qwen2.5-VL-7B-Instruct")}

M = {}  # macro name -> printed text
FIG = {}  # figure data


def put(name, value):
    if not name.isalpha():
        raise ValueError(f"macro name must be letters only: {name}")
    if name in M:
        raise ValueError(f"duplicate macro {name}")
    M[name] = value


def f(x, d=2):
    return f"{x:.{d}f}"


def grp(x):
    """Integer with a thousands separator."""
    return f"{int(round(x)):,}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def load_json(path):
    with open(path, encoding="utf-8") as h:
        return json.load(h)


def sweep_rows(folder):
    return load_json(os.path.join(SWEEPS, folder, "sweep.json"))["rows"]


def samples(run, bench):
    out = {}
    for path in glob.glob(os.path.join(run, "results", f"{bench}_test", "*", "test.jsonl")):
        with open(path, encoding="utf-8") as h:
            for line in h:
                if not line.strip():
                    continue
                r = json.loads(line)
                c = r["choices"][0]
                m = c["measurement"][0]
                out[r["question_id"]] = {
                    "wall": c["wall_time"][0],
                    "ttft": c["ttft"][0],
                    "new": c["new_tokens"][0],
                    "ids": m["output_token_ids"],
                    "prompt": m.get("input_ids_tokens"),
                }
    return out


def metadata(run):
    with open(os.path.join(run, "metadata.txt"), encoding="utf-8") as h:
        return dict(line.rstrip("\n").split("=", 1) for line in h if "=" in line)


def summary(run):
    return load_json(sorted(glob.glob(os.path.join(run, "reports", "*", "summary.json")))[-1])


def decode_ratios(a, d):
    """Decode-only speedup of d over a: all samples, and below the cap in both."""
    qs = sorted(set(a) & set(d))
    dec = lambda x: x["wall"] - x["ttft"]
    allr = statistics.mean(dec(a[q]) for q in qs) / statistics.mean(dec(d[q]) for q in qs)
    below = [q for q in qs if a[q]["new"] < CAP and d[q]["new"] < CAP]
    belowr = statistics.mean(dec(a[q]) for q in below) / statistics.mean(dec(d[q]) for q in below)
    tpot = lambda x: dec(x) / max(x["new"] - 1, 1)
    tok = statistics.mean(tpot(a[q]) for q in qs) / statistics.mean(tpot(d[q]) for q in qs)
    return allr, belowr, len(below), tok


# ---------------------------------------------------------------- LLaVA (teammates' runs)
L = load_json(os.path.join(HERE, "llava_teammates.json"))
for bench, key in (("textvqa", "Textvqa"), ("gqa", "Gqa"), ("mmvet", "Mmvet")):
    v = L["long"][bench]
    put(f"llavaLong{key}Speedup", v["speedup"])
    put(f"llavaLong{key}Tokens", v["mean_output_tokens"])
    put(f"llavaLong{key}Accept", v["accepted_per_step"])
    put(f"llavaLong{key}Same", v["identical"])
    put(f"llavaLong{key}Total", v["total"])
for bench, key in (("textvqa", "Textvqa"), ("gqa", "Gqa")):
    v = L["short"][bench]
    put(f"llavaShort{key}Speedup", v["speedup"])
    put(f"llavaShort{key}Acc", v["accuracy_both_methods"])
    put(f"llavaShort{key}Same", v["identical"])
put("llavaShortFirstShareMin", L["short"]["first_token_share_min_percent"])
put("llavaShortFirstShareMax", L["short"]["first_token_share_max_percent"])
pk = L["pruned_kv"]
for kept, acc in zip(pk["retained_percent"], pk["short_textvqa_accuracy"]):
    word = {"100": "Full", "50": "Half", "25": "Quarter", "10": "Tenth", "0": "None"}[kept]
    put(f"llavaPruned{word}Acc", acc)
put("llavaPrunedMaxGain", pk["max_throughput_gain_percent"])
put("llavaPrefixMin", pk["unpruned_prefix_match_min_percent"])
put("llavaPrefixMax", pk["unpruned_prefix_match_max_percent"])
FIG["llava"] = L

# ---------------------------------------------------------------- 3B single images (Week 3 jobs)
w3 = {k: sweep_rows(v) for k, v in W3.items()}
refs3 = [r for rows in w3.values() for r in rows if r["role"] == "reference"]


def ref_pairs(rows):
    """(job, style, bench) -> {'ar': row, 'vispec': row} for the reference rows."""
    out = {}
    for r in rows:
        if r["role"] != "reference":
            continue
        k = (r["slurm_job_id"], r["prompt_style"], r["benchmark"])
        out.setdefault(k, {})["ar" if r["method"] == "autoregressive" else "vispec"] = r
    return out


pairs3 = ref_pairs(refs3)
amdahl = []
for (job, style, bench), p in sorted(pairs3.items()):
    a, v = p["ar"], p["vispec"]
    amdahl.append({
        "model": "3B", "method": "ViSpec", "setting": f"{style} {bench}", "job": job,
        "ar_ttft_share": a["ttft_mean_s"] / a["latency_mean_s"],
        "speedup": a["latency_mean_s"] / v["latency_mean_s"],
        "ar_ttft": a["ttft_mean_s"], "ar_decode": a["latency_mean_s"] - a["ttft_mean_s"],
        "m_ttft": v["ttft_mean_s"], "m_decode": v["latency_mean_s"] - v["ttft_mean_s"],
    })


def three_b(style, bench, key):
    ps = [p for (j, s, b), p in pairs3.items() if s == style and b == bench]
    sp = [p["ar"]["latency_mean_s"] / p["vispec"]["latency_mean_s"] for p in ps]
    share = [p["vispec"]["ttft_mean_s"] / p["vispec"]["latency_mean_s"] for p in ps]
    arshare = [p["ar"]["ttft_mean_s"] / p["ar"]["latency_mean_s"] for p in ps]
    put(f"tb{key}SpeedMin", f(min(sp)))
    put(f"tb{key}SpeedMax", f(max(sp)))
    put(f"tb{key}Jobs", str(len(ps)))
    put(f"tb{key}Share", pct(statistics.mean(share)))
    put(f"tb{key}ArShare", pct(statistics.mean(arshare)))
    v = ps[0]["vispec"]
    put(f"tb{key}Accept", f(v["acceptance_mean"]))
    put(f"tb{key}Same", str(v["token_matched_vs_ar"]))
    put(f"tb{key}Total", str(v["token_checked_vs_ar"]))
    put(f"tb{key}QualAr", f(ps[0]["ar"]["quality"], 1 if bench == "textvqa" else 2))
    put(f"tb{key}QualSpec", f(v["quality"], 1 if bench == "textvqa" else 2))
    return ps


three_b("long", "textvqa", "LongTextvqa")
three_b("long", "coco_caption", "LongCoco")
three_b("short", "textvqa", "ShortTextvqa")
three_b("short", "coco_caption", "ShortCoco")

# decoding-phase speedup on long single-image answers, below the cap in both
dec_single = []
for (job, style, bench), p in pairs3.items():
    if style != "long":
        continue
    sa, sv = samples(p["ar"]["run_dir"], bench), samples(p["vispec"]["run_dir"], bench)
    dec_single.append(decode_ratios(sa, sv)[1])
put("tbSingleDecodeMin", f(min(dec_single)))
put("tbSingleDecodeMax", f(max(dec_single)))

# prompt lengths on single images (3B AR reference of job 398883 and 398923)
pl = {}
for (job, style, bench), p in pairs3.items():
    s = samples(p["ar"]["run_dir"], bench)
    pl.setdefault(bench, []).extend(x["prompt"] for x in s.values())
put("promptTextvqa", grp(statistics.mean(pl["textvqa"])))
put("promptCoco", grp(statistics.mean(pl["coco_caption"])))

# ---------------------------------------------------------------- 3B manga windows (five jobs)
w4 = sweep_rows(W4)
pairs4 = ref_pairs(w4)
sp4, share4, arshare4, bound4, below4, belown4, all4, tok4 = [], [], [], [], [], [], [], []
ar_len, sp_len, ar_cap, sp_cap = [], [], [], []
prompts = []
for (job, style, bench), p in sorted(pairs4.items()):
    a, v = p["ar"], p["vispec"]
    sp4.append(a["latency_mean_s"] / v["latency_mean_s"])
    share4.append(v["ttft_mean_s"] / v["latency_mean_s"])
    arshare4.append(a["ttft_mean_s"] / a["latency_mean_s"])
    bound4.append(a["latency_mean_s"] / v["ttft_mean_s"])
    sa, sv = samples(a["run_dir"], bench), samples(v["run_dir"], bench)
    allr, belowr, n, tok = decode_ratios(sa, sv)
    all4.append(allr)
    below4.append(belowr)
    belown4.append(n)
    tok4.append(tok)
    ar_len.append(statistics.mean(x["new"] for x in sa.values()))
    sp_len.append(statistics.mean(x["new"] for x in sv.values()))
    ar_cap.append(sum(x["new"] >= CAP for x in sa.values()))
    sp_cap.append(sum(x["new"] >= CAP for x in sv.values()))
    amdahl.append({
        "model": "3B", "method": "ViSpec", "setting": "pages", "job": job,
        "ar_ttft_share": a["ttft_mean_s"] / a["latency_mean_s"], "speedup": sp4[-1],
        "ar_ttft": a["ttft_mean_s"], "ar_decode": a["latency_mean_s"] - a["ttft_mean_s"],
        "m_ttft": v["ttft_mean_s"], "m_decode": v["latency_mean_s"] - v["ttft_mean_s"],
    })
    prompts = [x["prompt"] for x in sa.values()]
put("pagesJobs", str(len(sp4)))
put("tbPagesSpeedMin", f(min(sp4)))
put("tbPagesSpeedMax", f(max(sp4)))
put("tbPagesSpeedMean", f(statistics.mean(sp4)))
lat_ar3 = [p["ar"]["latency_mean_s"] for p in pairs4.values()]
put("tbPagesArSpread", pct(max(lat_ar3) / min(lat_ar3) - 1))
put("tbPagesShareMin", pct(min(share4)))
put("tbPagesShareMax", pct(max(share4)))
put("tbPagesArShareMin", pct(min(arshare4)))
put("tbPagesArShareMax", pct(max(arshare4)))
put("tbPagesBoundMin", f(min(bound4)))
put("tbPagesBoundMax", f(max(bound4)))
put("tbPagesDecodeMin", f(min(below4)))
put("tbPagesDecodeMax", f(max(below4)))
put("tbPagesDecodeN", str(belown4[0]))
put("tbPagesDecodeAllMin", f(min(all4)))
put("tbPagesDecodeAllMax", f(max(all4)))
put("tbPagesTokenMin", f(min(tok4)))
put("tbPagesTokenMax", f(max(tok4)))
put("tbPagesLenAr", f(ar_len[0], 1))
put("tbPagesLenSpec", f(sp_len[0], 1))
put("tbPagesCapAr", str(ar_cap[0]))
put("tbPagesCapSpec", str(sp_cap[0]))
ttfts4 = [p[m]["ttft_mean_s"] for p in pairs4.values() for m in ("ar", "vispec")]
put("tbPagesTtftMin", f(min(ttfts4)))
put("tbPagesTtftMax", f(max(ttfts4)))
v0 = next(iter(pairs4.values()))
put("tbPagesAccept", f(v0["vispec"]["acceptance_mean"]))
put("tbPagesQualAr", f(v0["ar"]["quality"]))
put("tbPagesQualSpec", f(v0["vispec"]["quality"]))
put("tbPagesBertAr", f(v0["ar"]["bertscore_f1"]))
put("tbPagesBertSpec", f(v0["vispec"]["bertscore_f1"]))
put("tbPagesSame", str(v0["vispec"]["token_matched_vs_ar"]))
put("pagesN", str(v0["vispec"]["n"]))
put("pagesPromptMean", grp(statistics.mean(prompts)))
put("pagesPromptMin", grp(min(prompts)))
put("pagesPromptMax", grp(max(prompts)))
img = [r["kv_image_tokens_mean"] for r in w4 if r.get("kv_image_tokens_mean")][0]
put("pagesImageShare", pct(img / statistics.mean(prompts)))
FIG["pages_3b"] = {"speedups": sp4, "ttft_share": share4}

# ---------------------------------------------------------------- 7B single images (experiment 10)
q7 = sweep_rows(Q7)
prim = [r for r in q7 if r["prompt_order"] == "image-last" and r["tree"] in (None, "none", "30-3-8")]
by = {}
for r in prim:
    k = (r["slurm_job_id"], r["prompt_style"], r["benchmark"])
    meth = {"autoregressive": "ar", "vispec_speculative": "vispec", "hivis_speculative": "hivis"}[r["method"]]
    by.setdefault(k, {})[meth] = r
for (job, style, bench), p in sorted(by.items()):
    a = p["ar"]
    for meth, label in (("vispec", "ViSpec"), ("hivis", "HiViS")):
        m = p[meth]
        amdahl.append({
            "model": "7B", "method": label, "setting": f"{style} {bench}", "job": job,
            "ar_ttft_share": a["ttft_mean_s"] / a["latency_mean_s"],
            "speedup": a["latency_mean_s"] / m["latency_mean_s"],
            "ar_ttft": a["ttft_mean_s"], "ar_decode": a["latency_mean_s"] - a["ttft_mean_s"],
            "m_ttft": m["ttft_mean_s"], "m_decode": m["latency_mean_s"] - m["ttft_mean_s"],
        })


def seven_b(style, bench, key):
    ps = [p for (j, s, b), p in by.items() if s == style and b == bench]
    for meth, mk in (("vispec", "Vis"), ("hivis", "Hiv")):
        sp = [p["ar"]["latency_mean_s"] / p[meth]["latency_mean_s"] for p in ps]
        put(f"sb{key}{mk}SpeedMin", f(min(sp)))
        put(f"sb{key}{mk}SpeedMax", f(max(sp)))
        m = ps[0][meth]
        put(f"sb{key}{mk}Accept", f(m["acceptance_mean"]))
        put(f"sb{key}{mk}Same", str(m["token_matched_vs_ar"]))
        put(f"sb{key}{mk}Qual", f(m["quality"], 1 if bench == "textvqa" else 2))
        put(f"sb{key}{mk}Share", pct(statistics.mean(p[meth]["ttft_mean_s"] / p[meth]["latency_mean_s"] for p in ps)))
    hv = [p["vispec"]["latency_mean_s"] / p["hivis"]["latency_mean_s"] for p in ps]
    put(f"sb{key}HovMin", f(min(hv)))
    put(f"sb{key}HovMax", f(max(hv)))
    put(f"sb{key}Jobs", str(len(ps)))
    put(f"sb{key}QualAr", f(ps[0]["ar"]["quality"], 1 if bench == "textvqa" else 2))
    put(f"sb{key}ArShare", pct(statistics.mean(p["ar"]["ttft_mean_s"] / p["ar"]["latency_mean_s"] for p in ps)))
    return ps


seven_b("long", "textvqa", "LongTextvqa")
coco7 = seven_b("long", "coco_caption", "LongCoco")
seven_b("short", "textvqa", "ShortTextvqa")
seven_b("short", "coco_caption", "ShortCoco")
# step time and time per output token on long COCO (job 412894)
p = by[("412894", "long", "coco_caption")]
for meth, mk in (("vispec", "Vis"), ("hivis", "Hiv")):
    s = summary(p[meth]["run_dir"])["benchmarks"]["coco_caption"]
    tpot = s["time_per_output_token_s"]["mean"]
    put(f"sbCoco{mk}Tpot", f(1000 * tpot, 1))
    # time per verification step: decode time over target passes, averaged over samples
    steps = []
    for path in glob.glob(os.path.join(p[meth]["run_dir"], "results", "coco_caption_test", "*", "test.jsonl")):
        with open(path, encoding="utf-8") as h:
            for line in h:
                if line.strip():
                    c = json.loads(line)["choices"][0]
                    steps.append(c["decode_time"][0] / max(c["measurement"][0]["iterations"], 1))
    put(f"sbCoco{mk}Step", f(1000 * statistics.mean(steps), 1))
FIG["teaser_single"] = {
    "short textvqa": {m: (by[("412895", "short", "textvqa")][m]["ttft_mean_s"], by[("412895", "short", "textvqa")][m]["latency_mean_s"]) for m in ("ar", "vispec", "hivis")},
    "long coco_caption": {m: (by[("412894", "long", "coco_caption")][m]["ttft_mean_s"], by[("412894", "long", "coco_caption")][m]["latency_mean_s"]) for m in ("ar", "vispec", "hivis")},
    "long textvqa": {m: (by[("412893", "long", "textvqa")][m]["ttft_mean_s"], by[("412893", "long", "textvqa")][m]["latency_mean_s"]) for m in ("ar", "vispec", "hivis")},
}

# ---------------------------------------------------------------- 7B manga windows (experiment 11)
with open(os.path.join(SWEEPS, R7, "run_dirs.txt"), encoding="utf-8") as h:
    runs7 = [line.strip() for line in h if line.strip()]
jobs7 = {}
for run in runs7:
    meta = metadata(run)
    meth = {"autoregressive": "ar", "vispec_speculative": "vispec", "hivis_speculative": "hivis"}[meta["method"]]
    jobs7.setdefault(meta["slurm_job_id"], {})[meth] = {"run": run, "s": samples(run, "reverse_summary"), "sum": summary(run), "order": meta.get("arm_list")}
mean = statistics.mean
sv7, sh7, hov7, ttft7, dv7, dh7, dvn, dhn = [], [], [], [], [], [], [], []
share_v, share_h, share_a = [], [], []
for job, p in sorted(jobs7.items()):
    a, v, hv = p["ar"]["s"], p["vispec"]["s"], p["hivis"]["s"]
    lat = {m: mean(x["wall"] for x in p[m]["s"].values()) for m in ("ar", "vispec", "hivis")}
    tt = {m: mean(x["ttft"] for x in p[m]["s"].values()) for m in ("ar", "vispec", "hivis")}
    sv7.append(lat["ar"] / lat["vispec"])
    sh7.append(lat["ar"] / lat["hivis"])
    hov7.append(lat["vispec"] / lat["hivis"])
    ttft7.extend(tt.values())
    share_a.append(tt["ar"] / lat["ar"])
    share_v.append(tt["vispec"] / lat["vispec"])
    share_h.append(tt["hivis"] / lat["hivis"])
    _, bv, nv, _ = decode_ratios(a, v)
    _, bh, nh, _ = decode_ratios(a, hv)
    dv7.append(bv)
    dh7.append(bh)
    dvn.append(nv)
    dhn.append(nh)
    for meth, label in (("vispec", "ViSpec"), ("hivis", "HiViS")):
        amdahl.append({
            "model": "7B", "method": label, "setting": "pages", "job": job,
            "ar_ttft_share": tt["ar"] / lat["ar"], "speedup": lat["ar"] / lat[meth],
            "ar_ttft": tt["ar"], "ar_decode": lat["ar"] - tt["ar"],
            "m_ttft": tt[meth], "m_decode": lat[meth] - tt[meth],
        })
    FIG.setdefault("teaser_pages", {})[job] = {m: (tt[m], lat[m]) for m in ("ar", "vispec", "hivis")}
put("sbPagesJobs", str(len(jobs7)))
put("sbPagesVisSpeedMin", f(min(sv7)))
put("sbPagesVisSpeedMax", f(max(sv7)))
put("sbPagesHivSpeedMin", f(min(sh7)))
put("sbPagesHivSpeedMax", f(max(sh7)))
put("sbPagesHovMin", f(min(hov7)))
put("sbPagesHovMax", f(max(hov7)))
put("sbPagesTtftMin", f(min(ttft7)))
put("sbPagesTtftMax", f(max(ttft7)))
put("sbPagesArShareMin", pct(min(share_a)))
put("sbPagesArShareMax", pct(max(share_a)))
put("sbPagesVisShare", pct(mean(share_v)))
put("sbPagesHivShare", pct(mean(share_h)))
put("sbPagesVisDecodeMin", f(min(dv7)))
put("sbPagesVisDecodeMax", f(max(dv7)))
put("sbPagesHivDecodeMin", f(min(dh7)))
put("sbPagesHivDecodeMax", f(max(dh7)))
put("sbPagesVisDecodeN", str(dvn[0]))
put("sbPagesHivDecodeN", str(dhn[0]))
first = jobs7[sorted(jobs7)[0]]
for meth, mk in (("ar", "Ar"), ("vispec", "Vis"), ("hivis", "Hiv")):
    s = first[meth]["s"]
    sc = first[meth]["sum"]["reverse_summary_scores"]
    put(f"sbPages{mk}Len", f(mean(x["new"] for x in s.values()), 1))
    put(f"sbPages{mk}Cap", str(sum(x["new"] >= CAP for x in s.values())))
    put(f"sbPages{mk}Rouge", f(sc["rouge1_f1"]))
    put(f"sbPages{mk}Bert", f(sc["bertscore_f1"]))
    if meth != "ar":
        acc = first[meth]["sum"]["benchmarks"]["reverse_summary"]["accepted_draft_tokens_per_iteration"]["mean"]
        put(f"sbPages{mk}Accept", f(acc))
        put(f"sbPages{mk}Same", str(sum(first["ar"]["s"][q]["ids"] == s[q]["ids"] for q in s)))
# HiViS over ViSpec on decode time, windows below the cap in both
hd = []
for job, p in sorted(jobs7.items()):
    v, hv = p["vispec"]["s"], p["hivis"]["s"]
    _, b, n, _ = decode_ratios(v, hv)
    hd.append((b, n))
put("sbPagesHovDecodeMin", f(min(b for b, _ in hd)))
put("sbPagesHovDecodeMax", f(max(b for b, _ in hd)))
put("sbPagesHovDecodeN", str(hd[0][1]))
lat_ar7 = [mean(x["wall"] for x in p["ar"]["s"].values()) for p in jobs7.values()]
put("sbPagesArSpread", pct(max(lat_ar7) / min(lat_ar7) - 1))

# ---------------------------------------------------------------- exactness
fp = {}
for k, paths in FP32.items():
    q = sum(load_json(x)["summary"]["questions"] for x in paths)
    e = sum(load_json(x)["summary"]["exact"] for x in paths)
    fp[k] = (q, e, len(paths))
for k, mk in (("3b_single", "TbSingle"), ("3b_pages", "TbPages"), ("7b_single", "SbSingle"), ("7b_pages", "SbPages")):
    q, e, n = fp[k]
    put(f"fpq{mk}", str(q))
    put(f"fpe{mk}", str(e))
put("fpqAll", str(sum(v[0] for v in fp.values())))
put("fpeAll", str(sum(v[1] for v in fp.values())))

margins = []
diag_tot = {}
for name, paths in (("3B", DIAG_3B), ("7B", DIAG_7B), ("7Bpages", DIAG_7B_PAGES)):
    cases = near = unexp = 0
    for path in paths:
        d = load_json(path)
        s = d["summary"]["overall"]
        cases += s["cases"]
        near += s["classes"]["near_tie"]
        unexp += s["classes"]["unexplained"] + s["classes"].get("both_wrong", 0)
        for c in d["cases"]:
            margins.append({"group": name, "bf16": c["bf16"]["margin"] if isinstance(c.get("bf16"), dict) else c.get("bf16_margin"),
                            "fp32": c["fp32"]["margin"] if isinstance(c.get("fp32"), dict) else c.get("fp32_margin"),
                            "id": c["question_id"], "index": c["first_mismatch_index"], "ar": c["ar_token_id"], "draft": c.get("vispec_token_id", c.get("draft_token_id"))})
    diag_tot[name] = (cases, near, unexp)
for name, mk in (("3B", "Tb"), ("7B", "Sb"), ("7Bpages", "SbPages")):
    c, n, u = diag_tot[name]
    put(f"diagCases{mk}", grp(c))
    put(f"diagNear{mk}", grp(n))
    put(f"diagOther{mk}", str(u))
put("diagCasesAll", grp(sum(v[0] for v in diag_tot.values())))
put("diagNearAll", grp(sum(v[1] for v in diag_tot.values())))
put("diagOtherAll", str(sum(v[2] for v in diag_tot.values())))
big = [m for m in margins if m["bf16"] is not None and m["bf16"] > 0.5]
put("diagBigBf", f(big[0]["bf16"]))
put("diagBigFp", f(big[0]["fp32"]))
near_max = max(m["bf16"] for m in margins if m["bf16"] is not None and m["bf16"] <= 0.5)
put("diagNearMax", f(near_max, 3))
put("diagNearMedian", f(statistics.median(m["bf16"] for m in margins if m["bf16"] is not None), 3))
FIG["margins"] = margins

# ---------------------------------------------------------------- visual sparsity
arms = [r for rows in w3.values() for r in rows if r["role"] == "arm"]


def same_policy_ref(rows, r):
    for x in rows:
        if (x["role"] == "arm" and x["sweep"] == r["sweep"] and x["benchmark"] == r["benchmark"] and x["prompt_style"] == r["prompt_style"]
                and x["slurm_job_id"] == r["slurm_job_id"] and (x.get("kv_policy") or x.get("draft_policy")) == (r.get("kv_policy") or r.get("draft_policy"))
                and (x.get("kv_retention") or x.get("draft_retention")) == 1.0):
            return x
    return None


allrows3 = [r for rows in w3.values() for r in rows]
below = [r for r in arms if (r.get("kv_retention") or r.get("draft_retention")) < 1.0]
draft = [r for r in below if r["sweep"] == "draft"]
evict = [r for r in below if r["sweep"] in ("kv", "combined")]
lat_draft = [r["latency_speedup_vs_samepolicy_1_0"] for r in draft]
lat_evict = [r["latency_speedup_vs_samepolicy_1_0"] for r in evict]
kv_evict = [r["live_kv_arm_over_samepolicy_1_0"] for r in evict]
kv_draft = [r["live_kv_arm_over_samepolicy_1_0"] for r in draft]
acc_draft = [r["acceptance_mean"] - same_policy_ref(allrows3, r)["acceptance_mean"] for r in draft]
ttft_ev = [same_policy_ref(allrows3, r)["ttft_mean_s"] / r["ttft_mean_s"] for r in evict]
put("spDraftLatMin", f(min(lat_draft), 3))
put("spDraftLatMax", f(max(lat_draft), 3))
put("spDraftKvMin", f(min(kv_draft), 3))
put("spDraftKvMax", f(max(kv_draft), 3))
put("spDraftAccMin", f(min(acc_draft), 3))
put("spDraftAccMax", f(max(acc_draft), 3))
put("spEvictLatMin", f(min(lat_evict), 3))
put("spEvictLatMax", f(max(lat_evict), 3))
put("spEvictKvMin", f(min(kv_evict), 3))
put("spEvictKvMax", f(max(kv_evict), 3))
put("spEvictTtftMin", f(min(ttft_ev), 3))
put("spEvictTtftMax", f(max(ttft_ev), 3))
long_ev = [r for r in evict if r["prompt_style"] == "long"]
put("spLongSameMin", str(min(r["token_matched_vs_vispec"] for r in long_ev)))
put("spLongSameMax", str(max(r["token_matched_vs_vispec"] for r in long_ev)))
# short-answer quality at a quarter of the image entries, KV eviction arms
for bench, bk in (("textvqa", "Textvqa"), ("coco_caption", "Coco")):
    ref = [r for r in allrows3 if r["role"] == "reference" and r["method"] == "vispec_speculative" and r["prompt_style"] == "short" and r["benchmark"] == bench][0]
    put(f"spShort{bk}Ref", f(ref["quality"], 1 if bench == "textvqa" else 2))
    for pol, pk in (("spatial", "Spatial"), ("attention", "Attention")):
        r = [x for x in arms if x["sweep"] == "kv" and x["prompt_style"] == "short" and x["benchmark"] == bench and x["kv_policy"] == pol and x["kv_retention"] == 0.25][0]
        put(f"spShort{bk}{pk}Quarter", f(r["quality"], 1 if bench == "textvqa" else 2))
# acceptance under eviction, long TextVQA
lt = [r for r in evict if r["sweep"] == "kv" and r["prompt_style"] == "long" and r["benchmark"] == "textvqa"]
put("spLongTextvqaAccMin", f(min(r["acceptance_mean"] for r in lt), 3))
put("spLongTextvqaAccMax", f(max(r["acceptance_mean"] for r in lt), 3))
FIG["sparsity_single"] = [
    {k: r.get(k) for k in ("prompt_style", "benchmark", "sweep", "kv_retention", "kv_policy", "draft_retention", "draft_policy",
                           "latency_speedup_vs_samepolicy_1_0", "live_kv_arm_over_samepolicy_1_0", "quality", "quality_delta_vs_vispec_ref", "slurm_job_id")}
    for r in arms
]
# manga windows
ref4 = {r["slurm_job_id"]: r for r in w4 if r["role"] == "reference" and r["method"] == "vispec_speculative"}
pg = [r for r in w4 if r["role"] == "arm" and r["method"] == "vispec_speculative"]
put("spPagesKvFull", f(ref4["398924"]["live_kv_mean_mib"], 1))
quarter = [r for r in pg if r["sweep"] == "kv" and r.get("kv_retention") == 0.25]
for r in quarter:
    pk = "Spatial" if r["kv_policy"] == "spatial" else "Attention"
    put(f"spPagesKvQuarter{pk}", f(r["live_kv_mean_mib"], 1))
    put(f"spPagesRougeQuarter{pk}", f(r["quality"]))
    put(f"spPagesLatQuarter{pk}", f(r["latency_speedup_vs_vispec_ref"], 3))
put("spPagesRougeRef", f(ref4["398924"]["quality"]))
rep = {}
with open(os.path.join(SWEEPS, W4, "repetition.tsv"), encoding="utf-8") as h:
    next(h)
    for line in h:
        run, d4, d4c, tail, capped = line.rstrip("\n").split("\t")
        rep[run] = (float(d4), int(capped))
name = lambda r: os.path.basename(r["run_dir"].rstrip("/"))
put("spPagesDistinctRef", f(rep[name(ref4["398924"])][0]))
put("spPagesCapRef", str(rep[name(ref4["398924"])][1]))
for r in quarter:
    pk = "Spatial" if r["kv_policy"] == "spatial" else "Attention"
    put(f"spPagesDistinctQuarter{pk}", f(rep[name(r)][0]))
    put(f"spPagesCapQuarter{pk}", str(rep[name(r)][1]))
FIG["sparsity_pages"] = [
    {k: r.get(k) for k in ("sweep", "kv_retention", "kv_policy", "draft_retention", "draft_policy", "latency_speedup_vs_vispec_ref",
                           "live_kv_mean_mib", "quality", "quality_delta_vs_vispec_ref", "slurm_job_id")}
    | {"ref_live_kv_mib": ref4[r["slurm_job_id"]]["live_kv_mean_mib"], "distinct4": rep[name(r)][0], "capped": rep[name(r)][1]}
    for r in pg
]

# ---------------------------------------------------------------- cost model
def lm_bytes(model_dir):
    total = embed = 0
    for path in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with open(path, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            header = json.loads(h.read(n))
        for k, v in header.items():
            if k == "__metadata__" or k.startswith("visual."):
                continue
            size = v["data_offsets"][1] - v["data_offsets"][0]
            total += size
            if k.endswith("embed_tokens.weight"):
                embed += size
    return total, embed


cost = {}
for key, d in MODELS.items():
    cfg = load_json(os.path.join(d, "config.json"))
    head = cfg["hidden_size"] // cfg["num_attention_heads"]
    per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * head * 2
    total, embed = lm_bytes(d)
    # A decode step reads the output projection but only one row of the input embedding, so an
    # untied input embedding is left out of the per-step read.
    step = total if cfg.get("tie_word_embeddings") else total - embed
    cost[key] = {"kv_bytes_per_token": per_token, "weight_bytes_per_step": step}
    mk = {"3b": "Tb", "7b": "Sb"}[key]
    put(f"cost{mk}KvPerToken", grp(per_token))
    put(f"cost{mk}Weights", f(step / 2**30))
    put(f"cost{mk}Thousand", f(100 * 1000 * per_token / step, 1))
    put(f"cost{mk}SevenThousand", f(100 * 7000 * per_token / step, 1))
    put(f"cost{mk}Equal", grp(round(step / per_token, -3)))
put("costTbPagesShare", f(100 * ref4["398924"]["live_kv_mean_mib"] * 2**20 / cost["3b"]["weight_bytes_per_step"], 1))
FIG["cost"] = cost
FIG["prompt_lengths"] = {"textvqa": statistics.mean(pl["textvqa"]), "coco_caption": statistics.mean(pl["coco_caption"]), "pages": statistics.mean(prompts),
                       "shortest": min(min(pl["coco_caption"]), min(pl["textvqa"])), "longest": max(prompts)}

# decode profile: estimated unprofiled GPU-busy fraction
busy = []
for path in glob.glob(os.path.join(PROFILE, "*", "profile_summary.json")):
    busy.append(load_json(path)["aggregate"]["profiled_windows"]["decode"]["estimated_unprofiled_gpu_busy_fraction"])
put("profileBusyMin", pct(min(busy)))
put("profileBusyMax", pct(max(busy)))

FIG["amdahl"] = amdahl

# ---------------------------------------------------------------- per-job tables for the appendix
def date_of(run):
    return metadata(run).get("started_utc", "")[:10]


BENCH = {"textvqa": "TextVQA", "coco_caption": "COCO"}
rows_single = []
for (job, style, bench), p in sorted(pairs3.items(), key=lambda kv: (kv[0][1], kv[0][2], date_of(kv[1]["ar"]["run_dir"]), kv[0][0])):
    a, v = p["ar"], p["vispec"]
    rows_single.append(("3B", style, BENCH[bench], date_of(a["run_dir"]), f(a["latency_mean_s"]), f(v["latency_mean_s"]), "n/a",
                        f(a["latency_mean_s"] / v["latency_mean_s"]), "n/a", pct(v["ttft_mean_s"] / v["latency_mean_s"])))
for (job, style, bench), p in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
    a, v, hv = p["ar"], p["vispec"], p["hivis"]
    rows_single.append(("7B", style, BENCH[bench], date_of(a["run_dir"]), f(a["latency_mean_s"]), f(v["latency_mean_s"]), f(hv["latency_mean_s"]),
                        f(a["latency_mean_s"] / v["latency_mean_s"]), f(a["latency_mean_s"] / hv["latency_mean_s"]), pct(v["ttft_mean_s"] / v["latency_mean_s"])))
rows_pages = []
for (job, style, bench), p in sorted(pairs4.items()):
    a, v = p["ar"], p["vispec"]
    rows_pages.append(("3B", date_of(a["run_dir"]), f(a["latency_mean_s"]), f(v["latency_mean_s"]), "n/a", f(a["ttft_mean_s"]),
                       f(a["latency_mean_s"] / v["latency_mean_s"]), "n/a"))
for job, p in sorted(jobs7.items()):
    lat = {m: mean(x["wall"] for x in p[m]["s"].values()) for m in ("ar", "vispec", "hivis")}
    tt = mean(x["ttft"] for x in p["ar"]["s"].values())
    rows_pages.append(("7B", date_of(p["ar"]["run"]), f(lat["ar"]), f(lat["vispec"]), f(lat["hivis"]), f(tt),
                       f(lat["ar"] / lat["vispec"]), f(lat["ar"] / lat["hivis"])))
os.makedirs(os.path.join(REPORT, "generated"), exist_ok=True)
TARGET = {"3B": "\\qwenS", "7B": "\\qwenL"}
with open(os.path.join(REPORT, "generated", "jobs_single.tex"), "w", encoding="utf-8") as h:
    h.write("% Generated by report_scripts/make_data.py. Do not edit.\n")
    last = None
    for r in rows_single:
        if r[:3] != last:
            if last is not None:
                h.write("\\midrule\n")
            h.write(f"\\multicolumn{{7}}{{l}}{{\\textit{{{TARGET[r[0]]}, {r[1]} {r[2]}}}}} \\\\\n")
        last = r[:3]
        h.write(" & ".join(r[3:]) + " \\\\\n")
with open(os.path.join(REPORT, "generated", "jobs_pages.tex"), "w", encoding="utf-8") as h:
    h.write("% Generated by report_scripts/make_data.py. Do not edit.\n")
    last = None
    for r in rows_pages:
        if last is not None and r[0] != last:
            h.write("\\midrule\n")
        last = r[0]
        h.write(" & ".join(r) + " \\\\\n")
put("jobsSingleCount", str(len(rows_single)))
put("jobsPagesCount", str(len(rows_pages)))

# ---------------------------------------------------------------- write
os.makedirs(os.path.join(REPORT, "generated"), exist_ok=True)
with open(os.path.join(REPORT, "generated", "numbers.tex"), "w", encoding="utf-8") as h:
    h.write("% Generated by report_scripts/make_data.py from the saved result files. Do not edit.\n")
    for k in sorted(M):
        v = M[k]
        if v.startswith("-"):
            v = "$-$" + v[1:]
        if "–" in v or "—" in v:
            raise ValueError(f"dash in {k}")
        h.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
with open(os.path.join(HERE, "figure_data.json"), "w", encoding="utf-8") as h:
    json.dump(FIG, h, indent=1)
print(f"{len(M)} macros, {len(amdahl)} speedup points, {len(margins)} diagnosed differences")
