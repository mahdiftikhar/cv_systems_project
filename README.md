# ViSpec: vision-language inference experiments

This UCF project compares two ways of generating answers to questions about images.
Both use LLaVA-1.6 Vicuna 7B. We want to reduce the time needed to generate an
answer while preserving its content and correctness.

- **Experiment 00** uses standard autoregressive decoding, which produces one
  token at a time. A token is a unit of model output, often part of a word.
- **Experiment 01** uses ViSpec. A smaller draft model proposes several tokens,
  and the original model checks them together.

We evaluate TextVQA (reading text in images), GQA (visual reasoning) and MM-Vet
(broader, open-ended visual understanding). The current subsets contain 100,
100 and 218 questions per method. These are inference experiments using existing
checkpoints, without training new weights.

## Results

The current greedy runs show faster answer completion with ViSpec and a similar
wait for the first token. Exact output equivalence and preserved answer quality
remain unresolved. Start with the [comparison page](experiment/COMPARISON.md)
for plots, benchmark tables and explanations of the measurements.

The [experiment register](experiment/README.md) identifies the runs and links to
their evidence. The [metric guide](experiment/METRICS.md) explains what each metric
measures, why it matters and how scoring works.

## Run on the UCF cluster

Follow [Getting started on Newton](GETTING_STARTED.md) for the full workflow or
the [cluster setup guide](experiment/NEWTON_SETUP.md) for environment details.
Model downloads, data preparation and evaluation run on the cluster. Reading
saved reports and generating plots can run locally.

1. Prepare Python 3.10+ with CUDA-compatible PyTorch and Transformers 4.51.3.
2. Run `bash prepare.sh` from the repository root on a cluster compute node.
3. Submit the experiments using the commands below.

```bash
bash experiment/submit.sh experiment/00_baseline_original
bash experiment/submit.sh experiment/01_baseline_draft
```

The individual [00 guide](experiment/00_baseline_original/README.md) and
[01 guide](experiment/01_baseline_draft/README.md) document settings, data subsets,
prerequisites and results. Generated answers and launch evidence live in unique
`runs/` directories. The two current run snapshots are versioned for team review.
New runs are ignored by default.

## Models and upstream work

| Role | Checkpoint |
| --- | --- |
| Target model used by both experiments | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| Draft used by experiment 01 | `JLKang/ViSpec-llava-v1.6-vicuna-7b-hf` |

This repository builds on **ViSpec: Accelerating Vision-Language Models with
Vision-Aware Speculative Decoding**, by Jialiang Kang, Han Shu, Wenshuo Li,
Yingjie Zhai and Xinghao Chen. The [paper](https://arxiv.org/abs/2509.15235) and
[checkpoint collection](https://huggingface.co/collections/JLKang/vispec-68cd460c5766dc65e908909f)
describe the upstream method. The project results linked above are our own
measurements on the UCF cluster.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{vispec,
  title={ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding},
  author={Kang, Jialiang and Shu, Han and Li, Wenshuo and Zhai, Yingjie and Chen, Xinghao},
  booktitle={Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). Redistribution
and use should follow this license.

## Acknowledgements

This work is supported by Huawei Noah's Ark Lab. We would like to acknowledge the foundational work of previous projects that inspired our approach, especially [EAGLE](https://github.com/SafeAILab/EAGLE) and [Medusa](https://github.com/FasterDecoding/Medusa). We also thank the anonymous NeurIPS reviewers for their insightful comments and valuable feedback.
