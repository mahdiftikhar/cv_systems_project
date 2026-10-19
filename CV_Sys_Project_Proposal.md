**Speculative Decoding for Vision-Language Models: Accelerated Inference via Dynamic Visual Sparsity**

**Selected Topic:** Topic 14  
**Team Members:** Yusra Ahmed, Mahd Qureshi, Aaditya Baranwal

**1\. Problem Statement and Motivation**

Vision-Language Model (VLM) inference incurs substantial computational and memory costs from multimodal prefill and sequential autoregressive decoding. Long visual prefixes drastically increase prefill computation and KV-cache memory, while subsequent text decoding steps must continuously attend to these cached visual tokens.

Speculative decoding mitigates this by allowing a lightweight draft model to propose candidate tokens for parallel target verification. However, existing frameworks like ViSpec face two primary bottlenecks:

1. **1\. Draft Overhead:** Retaining too much visual information slows down the draft, but compressing it too heavily ruins token acceptance rates.

2. **2\. Target KV-Cache Bloat:** The target model still retains full-resolution visual features during verification, limiting overall throughput.

These bottlenecks are severely exacerbated in **long-context sequential visual tasks**, such as reasoning across multi-panel manga narratives, where the visual KV-cache dominates GPU memory. Our project will investigate whether **dynamic visual pruning and target-side KV-cache eviction** can accelerate VLM speculative decoding without degrading task quality, particularly under the extreme memory demands of dense visual storytelling.

**2\. Related Work and Starting Codebase**

* **Speculative Decoding:** Leviathan et al. (2023), EAGLE-2 (Li et al., 2024).

* **Multimodal Acceleration:** Gagrani et al. (2024), SpecVLM (Huang et al., 2025).

* **Base Architecture:** We will build directly on the official **ViSpec** repository (Kang et al., 2025\) and its released pretrained draft checkpoints.

* **Target Model:** Qwen2.5-VL-3B-Instruct (paired with the matching ViSpec checkpoint).

**3\. Planned Approach**

We will establish standard autoregressive inference and unmodified ViSpec as our core baselines. From there, we will implement and evaluate **Dynamic Visual Sparsity**:

1. **1\. Draft-Side Compression:** Implementing aggressive spatial pooling or attention-guided pruning of visual features to speed up draft proposals.

2. **2\. Target-Side KV Eviction:** Pruning the target’s visual KV cache during verification while preserving textual entries to reduce memory overhead.

**Evaluation Domains:**

* **Standard Benchmarks:** Subsets of COCO Captions and TextVQA to measure base accuracy (CIDEr, VQA Accuracy) and token acceptance (α) against established baselines.

* **Long-Context Stress Test (Manga Narratives):** We will evaluate the pipeline on sequential manga panels. This domain requires the VLM to maintain character consistency, dialogue attribution, and scene progression across a massive visual token context. We will measure whether aggressive pruning breaks narrative coherence or if it unlocks massive throughput gains for long-horizon visual reasoning.

**Efficiency Metrics:** End-to-end latency (ms/token), wall-clock speedup, throughput, peak GPU memory (VRAM), and draft-token acceptance rate. For target-side KV pruning, we will explicitly measure output agreement relative to the uncompressed target.

**4\. Compute Plan**

Experiments will utilize the **UCF Newton cluster** for batched VLM inference and memory profiling. Laptops will be used for script development, UI/demo structuring, and small-scale testing to strictly adhere to GPU resource budgets.

**5\. 5-Week Schedule**

| Week | Primary Tasks & Deliverables |
| :---: | ----- |
| **Week 1** | Codebase setup on Newton cluster; establish Qwen2.5-VL standard autoregressive and vanilla ViSpec baselines. |
| **Week 2** | Implement draft-side spatial pruning and target-side KV-cache eviction logic; verify forward-pass correctness. |
| **Week 3** | Conduct visual-token retention sweeps (e.g., 25%, 50%, 75% budgets); log latency, VRAM, and acceptance rates on COCO/TextVQA. |
| **Week 4** | Run long-context stress tests on multi-panel manga sequences; evaluate narrative coherence vs. memory savings. |
| **Week 5** | Analyze empirical results (speedup vs. quality tradeoff curves); complete visualizations; finalize report and presentation. |

**6\. Division of Labor**

| Member | Primary Responsibilities |
| :---: | ----- |
| **Yusra Ahmed** | Literature review; verification mechanics analysis; token acceptance tracking (α) and output consistency evaluation. |
| **Mahd Qureshi** | Base repository environment setup; draft pruning and KV-cache compression implementation; baseline reproduction. |
| **Aaditya Baranwal** | Latency, throughput, and memory profiling; execution of manga narrative long-context stress tests; result visualization. |

**6\. Generative AI Disclosure**

LLMs to explore related work, and assist with refining this proposal.