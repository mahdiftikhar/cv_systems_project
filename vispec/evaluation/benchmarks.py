"""Benchmark samples and Qwen2.5-VL prompt inputs for the unified evaluator.

Samples are image-free records (paths only); images are opened in
``build_inputs`` so that decoding stays outside the evaluator's timed region.

``reverse_summary`` (Week 4) reads the Re:Verse+ volume 1 pages through the git-ignored
symlink ``<data_dir>/reverse_plus``. That dataset is read-only: this module only opens its
files for reading and never writes, links or caches anything inside it.
"""

import dataclasses
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_NAMES = ("textvqa", "coco_caption", "reverse_summary")
PROMPT_STYLES = ("long", "short")
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28

# Re:Verse+ volume 1 page windows. Each chapter of n pages is split into ceil(n / 6)
# consecutive windows whose sizes differ by at most 1, the larger windows first.
REVERSE_DIR_NAME = "reverse_plus"
REVERSE_VOLUME = "volume_1"
REVERSE_ASSETS = REPO_DIR / ".preparation/assets_reverse.json"
REVERSE_WINDOW_PAGES = 6
# Rows of the ViSpec draft's RoPE table (max_position_embeddings of the released
# ViSpec-Qwen2.5-VL-3B-Instruct draft). The draft indexes it with the full, uncompressed
# token count (cnets_ours.Model.topK_genrate, len_posi = input_ids.shape[1]), and that index
# keeps growing during generation. So a window's prompt must leave room for the whole
# output: prompt + max_new_tokens + depth + 1 <= rows.
DRAFT_POSITION_LIMIT = 8192
# The budget at the Week 3 settings (1,024 new tokens, depth 3): 8192 - 1024 - 3 - 1.
REVERSE_POSITION_BUDGET = DRAFT_POSITION_LIMIT - 1024 - 3 - 1
# Per-page pixel cap for reverse_summary only, lowered once and uniformly from the Week 1
# 1280*28*28: at that cap the largest 6-page window is 7,433 prompt tokens, over the 7,164
# budget. 1200*28*28 is the largest round cap that fits: the largest window is then 6,967
# tokens, 197 under the budget. TextVQA and COCO keep the Week 1 limits.
REVERSE_MAX_PIXELS = 1200 * 28 * 28
PIXEL_LIMITS = {
    "reverse_summary": {"min_pixels": QWEN_MIN_PIXELS, "max_pixels": REVERSE_MAX_PIXELS},
}
# Only the <D>...</D> (speech) and <T>...</T> (thought) tags are stripped; the inner text stays.
REVERSE_MARKUP = re.compile(r"</?[DT]>")
ANY_MARKUP = re.compile(r"</?[A-Za-z][^<>]*>")

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
    # One style only: the same prompt in every Week 4 arm. Pages are images only (no OCR).
    ("reverse_summary", "long"): "Summarize the story told in these manga pages, in order.",
}


@dataclass(frozen=True)
class Sample:
    question_id: str
    image_paths: tuple[str, ...]  # opened by build_inputs, outside timing; pages in order
    question: str | None  # None for captioning and summaries
    references: tuple[str, ...]
    # reverse_summary only: exact prompt length (text + image tokens) and image tokens per
    # page, computed by load_samples with the real processor. None for the other benchmarks.
    prompt_positions: int | None = None
    image_tokens: tuple[int, ...] | None = None


def check_prompt_style(benchmark: str, prompt_style: str) -> None:
    """Raise ValueError unless ``benchmark`` has an instruction for ``prompt_style``."""
    if benchmark not in BENCHMARK_NAMES:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {prompt_style!r}")
    if (benchmark, prompt_style) not in INSTRUCTIONS:
        styles = [style for name, style in INSTRUCTIONS if name == benchmark]
        raise ValueError(
            f"{benchmark} has no {prompt_style!r} prompt; it supports {styles}"
        )


def reference_sha256(text: str) -> str:
    """Digest of one reference text (UTF-8); run files record it instead of the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def reverse_windows(pages: int, max_pages: int = REVERSE_WINDOW_PAGES) -> list[tuple[int, int]]:
    """1-based inclusive (first, last) pages of ceil(pages / max_pages) consecutive windows.

    Window sizes differ by at most 1, and the larger windows come first.
    """
    if pages < 1:
        raise ValueError(f"a chapter needs at least one page, got {pages}")
    count = math.ceil(pages / max_pages)
    size, larger = divmod(pages, count)
    windows, first = [], 1
    for index in range(count):
        last = first + size + (index < larger) - 1
        windows.append((first, last))
        first = last + 1
    return windows


def reverse_page_text(path: Path) -> str:
    """One page's narration: BOM and <D>/<T> tags removed, inner text kept, stripped.

    Opens the file read-only. Raises ValueError on any other markup, so an unknown tag
    is never scored as prose.
    """
    with open(path, "r", encoding="utf-8-sig") as stream:
        text = stream.read()
    text = REVERSE_MARKUP.sub("", text.replace(chr(0xFEFF), ""))
    leftover = ANY_MARKUP.search(text)
    if leftover:
        raise ValueError(f"Unexpected markup {leftover.group(0)!r} in {path}")
    text = text.strip()
    if not text:
        raise ValueError(f"Empty page text: {path}")
    return text


def reverse_assets(path: Path | None = None) -> dict:
    """The prepared Re:Verse manifest (``prepare.sh --suite reverse``); raises when missing."""
    path = REVERSE_ASSETS if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run bash prepare.sh --suite reverse first"
        )
    assets = json.loads(path.read_text(encoding="utf-8"))
    if assets.get("ready") is not True or not assets.get("chapters"):
        raise ValueError(f"{path} is not a completed reverse preparation; rerun it")
    return assets


def reverse_window_specs(data_dir: Path) -> list[dict]:
    """All volume 1 windows in chapter and page order, with their joined references.

    Checks first, for every chapter in the prepared manifest, that the image pages and the
    page texts match one to one by page number and equal 1..N for the prepared count N.
    It also checks that ``<data_dir>/reverse_plus`` is the frozen copy that was prepared.
    """
    assets = reverse_assets()
    root = Path(data_dir) / REVERSE_DIR_NAME
    volume = root / REVERSE_VOLUME
    freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("sha256") != assets.get("freeze_sha256"):
        raise ValueError(
            f"{root} FREEZE.json digest {freeze.get('sha256')} differs from the prepared "
            f"{assets.get('freeze_sha256')} in {REVERSE_ASSETS}"
        )
    specs = []
    for chapter in sorted(assets["chapters"], key=int):
        pages = int(assets["chapters"][chapter]["pages"])
        expected = set(range(1, pages + 1))
        images = volume / "images" / f"chapter_{chapter}"
        texts = volume / "text" / f"chapter_{chapter}"
        image_pages = {int(p.stem) for p in images.glob("[0-9][0-9][0-9].jpg")}
        text_pages = {int(p.stem) for p in texts.glob("[0-9][0-9][0-9].txt")}
        if image_pages != expected or text_pages != expected:
            raise ValueError(
                f"Chapter {chapter}: images {len(image_pages)} and texts {len(text_pages)} "
                f"do not both match pages 1..{pages} one to one "
                f"(images only {sorted(image_pages - text_pages)}, texts only "
                f"{sorted(text_pages - image_pages)}, missing "
                f"{sorted(expected - (image_pages & text_pages))})"
            )
        for first, last in reverse_windows(pages):
            numbers = range(first, last + 1)
            specs.append(
                {
                    "question_id": f"v1_ch{int(chapter):02d}_p{first:03d}-{last:03d}",
                    "chapter": int(chapter),
                    "pages": [first, last],
                    "image_paths": tuple(str(images / f"{n:03d}.jpg") for n in numbers),
                    "reference": "\n".join(
                        reverse_page_text(texts / f"{n:03d}.txt") for n in numbers
                    ),
                }
            )
    return specs


def reverse_references(data_dir: Path) -> dict[str, str]:
    """Joined reference text per window question ID, read from the dataset (no processor)."""
    return {spec["question_id"]: spec["reference"] for spec in reverse_window_specs(data_dir)}


def prompt_layout(processor, benchmark: str, sample: Sample, prompt_style: str):
    """(positions, image tokens per image) of the exact prompt, built on CPU.

    Positions is the prompt's token count, text plus image tokens: the index the ViSpec
    draft gives the next token. Image tokens come from ``image_grid_thw``; the count is
    checked against the image-pad tokens in ``input_ids``.
    """
    inputs = build_inputs(processor, benchmark, sample, prompt_style, device="cpu")
    merge = processor.image_processor.merge_size
    grid = inputs["image_grid_thw"].tolist()
    image_tokens = tuple(int(t * h * w) // merge**2 for t, h, w in grid)
    ids = inputs["input_ids"][0]
    pads = int((ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    if pads != sum(image_tokens) or len(image_tokens) != len(sample.image_paths):
        raise ValueError(
            f"{sample.question_id}: {pads} image-pad tokens, but image_grid_thw gives "
            f"{image_tokens} for {len(sample.image_paths)} images"
        )
    text_tokens = ids.shape[0] - pads
    return text_tokens + sum(image_tokens), image_tokens


def draft_rope_rows(spec_model_path=None) -> int:
    """``max_position_embeddings`` of the draft config at ``spec_model_path``.

    Falls back to ``DRAFT_POSITION_LIMIT`` when the path is not a local checkpoint folder.
    """
    if spec_model_path:
        config = Path(spec_model_path) / "config.json"
        if config.is_file():
            rows = json.loads(config.read_text(encoding="utf-8")).get("max_position_embeddings")
            if isinstance(rows, int) and rows > 0:
                return rows
    return DRAFT_POSITION_LIMIT


def draft_position_budget(
    max_new_tokens: int = 1024, depth: int = 3, spec_model_path=None
) -> int:
    """Largest prompt whose full generation stays inside the draft's RoPE table.

    prompt + max_new_tokens + depth + 1 <= rope rows; 7,164 at the Week 3 settings.
    """
    return draft_rope_rows(spec_model_path) - int(max_new_tokens) - int(depth) - 1


def _reverse_samples(data_dir: Path, num_samples: int, processor, budget) -> list[Sample]:
    """The first ``num_samples`` windows (all 55 when it is larger), position-checked."""
    if processor is None:
        raise ValueError(
            "reverse_summary needs the processor (load_processor) to check every window "
            "against the draft's position budget"
        )
    budget = REVERSE_POSITION_BUDGET if budget is None else int(budget)
    samples = []
    for spec in reverse_window_specs(data_dir)[:num_samples]:
        sample = Sample(
            question_id=spec["question_id"],
            image_paths=spec["image_paths"],
            question=None,
            references=(spec["reference"],),
        )
        positions, image_tokens = prompt_layout(processor, "reverse_summary", sample, "long")
        if positions > budget:
            raise ValueError(
                f"{sample.question_id}: prompt of {positions} tokens exceeds the position "
                f"budget of {budget} (prompt + max_new_tokens + depth + 1 must fit the "
                "draft's RoPE table); lower the reverse_summary max_pixels uniformly "
                "(never per window)"
            )
        samples.append(
            dataclasses.replace(sample, prompt_positions=positions, image_tokens=image_tokens)
        )
    return samples


def load_samples(
    benchmark: str,
    data_dir: Path,
    num_samples: int = 100,
    processor=None,
    position_budget=None,
) -> list[Sample]:
    """Benchmark samples in evaluation order.

    ``num_samples`` is the subset size for TextVQA and COCO, and a maximum for
    reverse_summary, which has 55 windows (so the default 100 gives all of them).
    reverse_summary also needs ``processor``, to compute every window's exact prompt
    length; it raises if any window exceeds ``position_budget`` (default
    ``REVERSE_POSITION_BUDGET``, see ``draft_position_budget()``). The other benchmarks ignore
    ``processor`` and ``position_budget``.
    """
    if benchmark == "textvqa":
        return _textvqa_samples(data_dir, num_samples)
    if benchmark == "coco_caption":
        return _coco_samples(data_dir, num_samples)
    if benchmark == "reverse_summary":
        return _reverse_samples(data_dir, num_samples, processor, position_budget)
    raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARK_NAMES}")


def manifest_row(benchmark: str, sample: Sample) -> dict:
    """The evaluator's manifest entry for one sample (``measurement.write_manifest`` input).

    reverse_summary records the window's pages, prompt positions and a digest of its
    reference instead of the text, so licensed prose is never copied into run folders.
    """
    if benchmark != "reverse_summary":
        return {
            "qid": sample.question_id,
            "question_id": sample.question_id,
            "answers": list(sample.references),
        }
    return {
        "question_id": sample.question_id,
        "answers": [],
        "manifest_fields": {
            "volume": REVERSE_VOLUME,
            "chapter": int(Path(sample.image_paths[0]).parent.name.split("_")[1]),
            "pages": [int(Path(p).stem) for p in sample.image_paths],
            "prompt_positions": sample.prompt_positions,
            "image_tokens": list(sample.image_tokens),
            "text_tokens": sample.prompt_positions - sum(sample.image_tokens),
            "reference_sha256": [reference_sha256(r) for r in sample.references],
        },
    }


def manifest_header(benchmark: str, samples: list[Sample], budget: int) -> dict | None:
    """Top-level manifest fields: the window settings for reverse_summary, else None."""
    if benchmark != "reverse_summary":
        return None
    positions = [s.prompt_positions for s in samples]
    return {
        "windows": len(samples),
        "window_rule": f"ceil(n/{REVERSE_WINDOW_PAGES}) consecutive windows per chapter, "
        "sizes differ by at most 1, larger first",
        "dataset_freeze_sha256": reverse_assets()["freeze_sha256"],
        "min_pixels": PIXEL_LIMITS[benchmark]["min_pixels"],
        "max_pixels": PIXEL_LIMITS[benchmark]["max_pixels"],
        "position_budget": budget,
        "position_budget_rule": "prompt + max_new_tokens + depth + 1 <= draft RoPE rows",
        "prompt_positions": {
            "min": min(positions),
            "median": statistics.median(positions),
            "max": max(positions),
        },
    }


def stored_references(benchmark: str, sample: Sample) -> list[str]:
    """Reference text the evaluator stores in answer records: none for reverse_summary."""
    return [] if benchmark == "reverse_summary" else list(sample.references)


def build_messages(benchmark: str, sample: Sample, prompt_style: str) -> list[dict]:
    check_prompt_style(benchmark, prompt_style)
    instruction = {"type": "text", "text": INSTRUCTIONS[(benchmark, prompt_style)]}
    if benchmark == "reverse_summary":
        # Multi-image: every page in reading order, then the instruction.
        content = [{"type": "image"} for _ in sample.image_paths] + [instruction]
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
            {"role": "user", "content": content},
        ]
    content = []
    if benchmark == "textvqa":
        content.append({"type": "text", "text": sample.question})
    content.append(instruction)
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
    # Only reverse_summary overrides the processor's Week 1 pixel limits (PIXEL_LIMITS).
    return processor(
        images=images, text=text, return_tensors="pt", **PIXEL_LIMITS.get(benchmark, {})
    ).to(device)
