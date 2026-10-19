"""Benchmark samples and Qwen2.5-VL prompt inputs for the unified evaluator.

Samples are image-free records (paths only); images are opened in
``build_inputs`` so that decoding stays outside the evaluator's timed region.
"""

import json
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_NAMES = ("textvqa", "coco_caption")
PROMPT_STYLES = ("long", "short")
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28

SYSTEM_TEXT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

INSTRUCTIONS = {
    ("textvqa", "long"): (
        "Perform an OCR task on the provided image. Please extract the text accurately "
        "and provide a detailed explanation of the process. Ensure the response is "
        "comprehensive and well-structured."
    ),
    ("textvqa", "short"): "Answer the question using a single word or phrase.",
    ("coco_caption", "long"): "Please provide a detailed description of the given image.",
    ("coco_caption", "short"): "Provide a one-sentence caption for the provided image.",
}


@dataclass(frozen=True)
class Sample:
    question_id: str
    image_paths: tuple[str, ...]  # opened by build_inputs, outside timing
    question: str | None  # None for captioning
    references: tuple[str, ...]


def coco_subset(parquet_path: Path, num_samples: int = 100, seed: int = 42):
    """Seeded shuffle of the Karpathy-test parquet, first ``num_samples`` rows."""
    from datasets import Dataset

    ds = Dataset.from_parquet(str(parquet_path))
    return ds.shuffle(seed=seed).select(range(num_samples))


def _textvqa_samples(data_dir: Path, num_samples: int) -> list[Sample]:
    from datasets import Dataset

    root = Path(data_dir) / "textvqa"
    data = json.loads((root / "TextVQA_0.5.1_val.json").read_text())
    assert (
        data["dataset_type"] == "val"
        and data["dataset_name"] == "textvqa"
        and data["dataset_version"] == "0.5.1"
    )
    rows = [
        {
            "question_id": d["question_id"],
            "question": d["question"],
            "image_id": d["image_id"],
            "answers": d["answers"],
        }
        for d in data["data"]
    ]
    # Same permutation as experiment 00 (gen_baseline_answer_textvqa.load_data).
    subset = Dataset.from_list(rows).shuffle(seed=42).select(range(0, num_samples))
    return [
        Sample(
            question_id=str(r["question_id"]),
            image_paths=(str(root / "train_images" / (r["image_id"] + ".jpg")),),
            question=r["question"],
            references=tuple(r["answers"]),
        )
        for r in subset
    ]


def _coco_samples(data_dir: Path, num_samples: int) -> list[Sample]:
    root = Path(data_dir) / "coco"
    subset = coco_subset(root / "karpathy_test.parquet", num_samples)
    return [
        Sample(
            question_id=str(r["cocoid"]),
            image_paths=(str(root / "val2014" / r["filename"]),),
            question=None,
            references=tuple(r["sentences"]),
        )
        for r in subset
    ]


def load_samples(benchmark: str, data_dir: Path, num_samples: int = 100) -> list[Sample]:
    if benchmark == "textvqa":
        return _textvqa_samples(data_dir, num_samples)
    if benchmark == "coco_caption":
        return _coco_samples(data_dir, num_samples)
    raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARK_NAMES}")


def build_messages(benchmark: str, sample: Sample, prompt_style: str) -> list[dict]:
    if benchmark not in BENCHMARK_NAMES:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {prompt_style!r}")
    content = []
    if benchmark == "textvqa":
        content.append({"type": "text", "text": sample.question})
    content.append({"type": "text", "text": INSTRUCTIONS[(benchmark, prompt_style)]})
    content.append({"type": "image"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": content},
    ]


def load_processor(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path,
        use_fast=True,
        min_pixels=QWEN_MIN_PIXELS,
        max_pixels=QWEN_MAX_PIXELS,
    )


def build_inputs(processor, benchmark: str, sample: Sample, prompt_style: str, device: str = "cuda:0"):
    from PIL import Image

    images = []
    for p in sample.image_paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    messages = build_messages(benchmark, sample, prompt_style)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(images=images, text=text, return_tensors="pt").to(device)
