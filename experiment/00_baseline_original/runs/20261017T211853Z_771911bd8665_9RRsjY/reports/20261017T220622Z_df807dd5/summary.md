# Run metrics

- Run: `/lustre/fs1/home/ma076549/cv_systems_project/experiment/00_baseline_original/runs/20261017T211853Z_771911bd8665_9RRsjY`
- Evaluation exit code: `0`
- Units: seconds, tokens/second, and quality score percent.
- Quality applies to completed questions in the evaluated subset.
- Device report: [device.txt](../../device.txt)

| Benchmark | Questions | Mean latency | Median | p95 | Mean TTFT | TTFT p95 | Tokens/s | Truncated | Quality (%) | Scored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| textvqa | 100/100 | 9.6748 | 9.5279 | 22.8054 | 0.2857 | 0.3850 | 44.3731 | 0.1100 | 0.0000 | 100/100 |
| gqa | 100/100 | 1.3804 | 1.2514 | 2.7895 | 0.2729 | 0.3472 | 35.4091 | 0.0000 | 0.0000 | 100/100 |
| mmvet | 218/218 | 4.1088 | 2.9408 | 10.6145 | 0.2766 | 0.4125 | 43.7144 | 0.0138 | Pending | 0/218 |

- Mean/median/p95 TTFT, decode time, TPOT, output lengths, acceptance, and per-device peak memory are in `summary.json`.
- Missing measurements remain null; summary distributions include their sample counts.
- Peak bytes are reported separately for each visible GPU; peaks are not summed across devices.
- Quality scoring status and missing prerequisites are listed below.

- **textvqa:** complete; VQA consensus (0–1)
- **gqa:** complete; exact match, strip/lowercase (0–1)
- **mmvet:** pending; pending: import official MM-Vet judge grades

| Benchmark | Visible GPU | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | --- | --- | --- |
| textvqa | 0 | 20.872 | 24.709 |
| gqa | 0 | 20.828 | 22.725 |
| mmvet | 0 | 20.930 | 30.959 |

- **textvqa hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **textvqa allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[0, 2, 4, 6, 32, 34, 36, 38]`.
- **gqa hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **gqa allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[0, 2, 4, 6, 32, 34, 36, 38]`.
- **mmvet hardware:** 0: NVIDIA H100 PCIe (79.2 GiB); precision ['torch.float16'].
- **mmvet allocation:** `{'CUDA_VISIBLE_DEVICES': '0', 'SLURM_CPUS_PER_TASK': '8', 'SLURM_CPUS_ON_NODE': '8', 'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': None, 'SLURM_GPUS': None, 'SLURM_GPUS_PER_TASK': None, 'SLURM_GPUS_ON_NODE': '1'}`; CPU affinity: `[0, 2, 4, 6, 32, 34, 36, 38]`.

- Generation limits differ between original and draft; inspect stop reasons before interpreting speedups.
- Inherited TextVQA prompts request OCR explanations; strict answer scoring may penalize verbose responses.
- Memory is measured after warmup; initial cache/model allocation peaks are excluded.
