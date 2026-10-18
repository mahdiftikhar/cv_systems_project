# Experiment workflowz

- Use judgment when formatting documentation: write explanations and connected
  reasoning as short paragraphs, use bullets for genuinely parallel items or
  checklists, use numbered lists for steps, and use tables for comparisons.
  Do not turn every sentence into a bullet or pack fragments together with
  semicolons. Explain relevant context and what each metric measures and why.
- Keep reader-facing results focused on the current relevant runs. Preserve raw
  evidence from all launches, but omit earlier-run summaries unless requested.
- Do not create or modify test files in this project. Use lightweight checks or
  direct verification as appropriate instead.
- Maintain experiments under `experiment/NN_descriptive_name/`, starting with
  `experiment/00_baseline_original/`.
- Run compute-heavy experiments on the user's UCF cluster. Local work should focus
  on implementation, documentation, and lightweight checks; do not launch model
  evaluation or download large weights/datasets locally without a request.
- Each experiment needs a runnable shell script and a brief README documenting
  purpose/hypothesis, changes from baseline, models, data/splits/subsets, settings,
  cluster prerequisites, exact run command, metric definitions, results, and limitations.
- Save combined stdout/stderr and the full Git commit at launch inside the
  experiment directory. Use unique `runs/` subdirectories to preserve reruns. Capture
  exact commands, working tree status/diff, environment, GPU, scheduler job ID,
  timestamps, raw outputs, and exit status. Prefer committed code for cluster runs.
- Every experiment must capture device details before evaluation, using
  `experiment/capture_device.sh` (or an equivalent report). Preserve CPU model/core
  count/affinity, host RAM and allocation limits, OS/kernel, GPU model/count/UUID/
  VRAM/compute capability, driver and CUDA/PyTorch versions, GPU clocks/power/MIG
  state, GPU visibility, and scheduler resource allocation. Record unavailable
  probes explicitly. Distinguish host-visible hardware from allocated devices.
  Include a hardware summary and link to the device report with reported results;
  keep separate result rows for different hardware. Launch snapshots do not measure
  peak memory, energy, or utilization during evaluation.
- Keep `experiment/README.md` as the register of all experiments, with an overall
  results table and links to individual READMEs. Update it when adding experiments
  and when verified measurements become available. Mark unmeasured results pending.
- Document per-benchmark results and sample counts before aggregating. State units,
  aggregation rules, and token-count definitions; compare matching settings and
  hardware. Do not invent results or substitute upstream paper results.
- For `00_baseline_original`, use the existing autoregressive evaluators with
  `llava-hf/llava-v1.6-vicuna-7b-hf` (smallest LLaVA-1.6 listed here). Keep Qwen,
  larger models, training, and inference changes outside this experiment unless
  the user explicitly changes its scope. Document inherited evaluation limitations.
- The baseline suite is TextVQA (priority, run first), GQA, and MM-Vet. Keep future
  comparisons on this same suite unless the user changes it; do not restore all
  eight upstream benchmarks by default.
- Keep generated runs out of Git by default and preserve their evidence on project
  storage. The user selected the exact snapshots for jobs 950793 and 950803 for
  versioning so teammates can inspect all answers, reports, and launch evidence.
  Their explicit `.gitignore` exceptions do not authorize tracking other runs.
