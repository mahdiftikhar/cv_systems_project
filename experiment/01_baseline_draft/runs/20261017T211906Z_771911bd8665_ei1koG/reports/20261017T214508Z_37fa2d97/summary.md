# Run metrics

- Run: `/lustre/fs1/home/ma076549/cv_systems_project/experiment/01_baseline_draft/runs/20261017T211906Z_771911bd8665_ei1koG`
- Evaluation exit code: `0`
- Units: seconds, tokens/second, and quality score percent.
- Quality applies to completed questions in the evaluated subset.
- Device report: [device.txt](../../device.txt)

| Benchmark | Questions | Mean latency | Median | p95 | Mean TTFT | TTFT p95 | Tokens/s | Truncated | Quality (%) | Scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| textvqa | 100/100 | 3.3647 | 3.3347 | 6.6669 | 0.2869 | 0.3875 | 127.4066 | 0.1100 | 0.0000 | 100/100 |
| gqa | 100/100 | 0.6400 | 0.5843 | 1.1124 | 0.2622 | 0.3135 | 76.3285 | 0.0000 | 0.0000 | 100/100 |
| mmvet | 218/218 | 1.6507 | 1.2501 | 3.9971 | 0.2776 | 0.4140 | 108.6883 | 0.0138 | Pending | 0/218 |

- Mean/median/p95 TTFT, decode time, TPOT, output lengths, acceptance, and per-device peak memory are in `summary.json`.
- Missing measurements remain null; summary distributions include their sample counts.
- Peak bytes are reported separately for each visible GPU; peaks are not summed across devices.
- Quality scoring status and missing prerequisites are listed below.

- **textvqa:** complete; VQA consensus (0–1)
- **gqa:** complete; exact match, strip/lowercase (0–1)
- **mmvet:** pending; pending: import official MM-Vet judge grades

| Benchmark | Visible GPU | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | --- | --- | --- |
| textvqa | 0 | 21.721 | 24.877 |
| gqa | 0 | 21.672 | 22.893 |
| mmvet | 0 | 21.785 | 31.127 |

- **textvqa hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **textvqa allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[1, 3, 5, 7, 33, 35, 37, 39]`.
- **gqa hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **gqa allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[1, 3, 5, 7, 33, 35, 37, 39]`.
- **mmvet hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **mmvet allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[1, 3, 5, 7, 33, 35, 37, 39]`.

- Generation limits differ between original and draft; inspect stop reasons before interpreting speedups.
- Inherited TextVQA prompts request OCR explanations; strict answer scoring may penalize verbose responses.
- Memory is measured after warmup; initial cache/model allocation peaks are excluded.
