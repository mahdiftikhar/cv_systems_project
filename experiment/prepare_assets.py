"""Prepare the fixed LLaVA/ViSpec benchmark suite beneath one repository root."""

import argparse
import ast
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

MODELS = {
    "base": ("llava-hf/llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-7b-hf"),
    "draft": (
        "JLKang/ViSpec-llava-v1.6-vicuna-7b-hf",
        "ViSpec-llava-v1.6-vicuna-7b-hf",
    ),
}
SOURCES = {
    "textvqa_json": "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
    "textvqa_images": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "gqa_images": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
    "gqa_questions": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
    "gqa_prompts": "https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view",
    "textvqa_evaluator": "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/llava/eval/m4c_evaluator.py",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configure(root):
    """Set cache paths before importing Hugging Face/datasets or gdown."""
    cache = root / ".cache"
    values = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "HF_ASSETS_CACHE": cache / "huggingface/assets",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "XDG_CACHE_HOME": cache,
        "UV_CACHE_DIR": cache / "uv",
        "TMPDIR": cache / "tmp",
    }
    for key, path in values.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    return values


def validate_download(path, kind):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing/empty file: {path}")
    if kind == "zip":
        with ZipFile(path) as archive:
            if not archive.namelist():
                raise ValueError(f"Empty ZIP archive: {path}")
    elif kind == "json":
        json.loads(path.read_text())
    elif kind == "python":
        tree = ast.parse(path.read_text())
        if not any(
            isinstance(node, ast.ClassDef) and node.name == "EvalAIAnswerProcessor"
            for node in tree.body
        ):
            raise ValueError(f"Not a TextVQA answer processor: {path}")


def quarantine(path):
    if path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = path.with_name(path.name + ".invalid-" + stamp)
        path.replace(destination)
        print(f"Preserved invalid download: {destination}", flush=True)


def reuse(path, kind):
    if not path.exists():
        return False
    try:
        validate_download(path, kind)
    except (ValueError, BadZipFile, SyntaxError, UnicodeError):
        quarantine(path)
        return False
    print(f"Reusing {path}", flush=True)
    return True


def download_http(url, path, kind):
    """Resume .part files with Range/If-Range, restarting if the server ignores Range."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if reuse(path, kind):
        return
    partial = path.with_name(path.name + ".part")
    metadata_path = partial.with_name(partial.name + ".json")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        previous = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        if previous.get("url", url) != url:
            quarantine(partial)
            offset = 0
            previous = {}
        headers = {"User-Agent": "cv-systems-preparation/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            validator = previous.get("etag") or previous.get("last_modified")
            if validator:
                headers["If-Range"] = validator
        print(f"Downloading {url} (resume offset {offset} bytes)", flush=True)
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                append = response.status == 206 and offset > 0
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset:
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                expected = response.headers.get("Content-Length")
                atomic_json(
                    metadata_path,
                    {
                        "url": url,
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    },
                )
                received = 0
                next_progress = 256 * 1024 * 1024
                with partial.open("ab" if append else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        received += len(chunk)
                        if received >= next_progress:
                            print(
                                f"  {path.name}: {(received + (offset if append else 0)) / 2**30:.2f} GiB",
                                flush=True,
                            )
                            next_progress += 256 * 1024 * 1024
                if expected is not None and received != int(expected):
                    raise OSError(f"Incomplete response: {received}/{expected} bytes")
            validate_download(partial, kind)
            partial.replace(path)
            metadata_path.unlink(missing_ok=True)
            return
        except HTTPError as error:
            if error.code == 416 and partial.exists():
                try:
                    validate_download(partial, kind)
                    partial.replace(path)
                    metadata_path.unlink(missing_ok=True)
                    return
                except (ValueError, BadZipFile, SyntaxError, UnicodeError):
                    quarantine(partial)
            if error.code not in (416, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (
            URLError,
            OSError,
            ValueError,
            BadZipFile,
            SyntaxError,
            UnicodeError,
        ) as error:
            if isinstance(error, (ValueError, BadZipFile, SyntaxError, UnicodeError)):
                quarantine(partial)
            if attempt == 2:
                raise
            print(f"Retrying after {error}", flush=True)
        time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}")


def download_prompts(path):
    if reuse(path, "zip"):
        return
    try:
        gdown = importlib.import_module("gdown")
    except ModuleNotFoundError:
        raise RuntimeError(
            "Missing gdown. Install requirements.txt into the project environment "
            "with uv pip install --python .venv/bin/python -r requirements.txt "
            "--extra-index-url https://download.pytorch.org/whl/cu128 "
            "--index-strategy unsafe-best-match on a compute node, then rerun preparation."
        ) from None
    # gdown's cookie file otherwise lives in the user's home directory.
    result = gdown.download(
        SOURCES["gqa_prompts"], str(path), fuzzy=True, resume=True, use_cookies=False
    )
    if not result:
        raise RuntimeError("GQA prompt download failed; rerun preparation to resume")
    validate_download(path, "zip")


def extract_member(archive_path, filename, destination):
    with ZipFile(archive_path) as archive:
        matches = [
            item for item in archive.infolist() if Path(item.filename).name == filename
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {filename} in {archive_path}, found {len(matches)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".extract.part")
        with archive.open(matches[0]) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)


def extract_images(archive_path, destination):
    """Flatten image entries into the exact directory expected by the evaluators."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    seen = set()
    try:
        with ZipFile(archive_path) as archive:
            for item in archive.infolist():
                name = Path(item.filename)
                if item.is_dir() or name.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    continue
                if name.name in seen:
                    raise ValueError(
                        f"Duplicate image basename in archive: {name.name}"
                    )
                seen.add(name.name)
                target = destination / name.name
                if target.is_file() and target.stat().st_size == item.file_size:
                    continue
                temporary = target.with_name(target.name + ".extract.part")
                with archive.open(item) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
                extracted += 1
                if extracted % 10000 == 0:
                    print(
                        f"Extracted {extracted} images into {destination}", flush=True
                    )
    except BadZipFile:
        quarantine(archive_path)
        raise
    if not seen:
        raise ValueError(f"No images in {archive_path}")
    print(
        f"Images ready: {destination} ({len(seen)} files; {extracted} extracted this run)",
        flush=True,
    )


def validate_benchmarks(root):
    bench = root / "bench_data"
    text = json.loads((bench / "textvqa/TextVQA_0.5.1_val.json").read_text())
    if (
        text.get("dataset_type"),
        text.get("dataset_name"),
        text.get("dataset_version"),
    ) != ("val", "textvqa", "0.5.1"):
        raise ValueError("Unexpected TextVQA split/version")
    rows = text["data"]
    if len(rows) < 100:
        raise ValueError("TextVQA needs at least 100 rows")
    for row in rows:
        image = bench / "textvqa/train_images" / (row["image_id"] + ".jpg")
        if not image.is_file() or len(row["answers"]) != 10:
            raise ValueError(
                f"Missing image or references for TextVQA {row['question_id']}"
            )
    labels = json.loads((bench / "gqa/testdev_balanced_questions.json").read_text())
    seen = set()
    with (bench / "gqa/llava_gqa_testdev_balanced.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["question_id"])
            image = bench / "gqa/images" / row["image"]
            if (
                not image.is_file()
                or key not in labels
                or not isinstance(labels[key].get("answer"), str)
                or key in seen
            ):
                raise ValueError(
                    f"Missing GQA image/answer or duplicate question: {key}"
                )
            seen.add(key)
    if len(seen) < 100:
        raise ValueError("GQA needs at least 100 rows")
    validate_download(root / "tools/downloaded/m4c_evaluator.py", "python")
    return {"textvqa_input_rows": len(rows), "gqa_input_rows": len(seen)}


def validate_model(path, require_processor=False):
    json.loads((path / "config.json").read_text())
    if require_processor:
        for filename in ("tokenizer_config.json", "preprocessor_config.json"):
            json.loads((path / filename).read_text())
        if not any(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in ("tokenizer.model", "tokenizer.json")
        ):
            raise ValueError(f"Missing tokenizer vocabulary: {path}")
    indexes = list(path.glob("*.index.json"))
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"Missing model weights: {path}")
    for index in indexes:
        data = json.loads(index.read_text())
        for filename in set(data.get("weight_map", {}).values()):
            if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
                raise ValueError(f"Missing model shard: {path / filename}")
    if any(item.stat().st_size == 0 for item in weights):
        raise ValueError(f"Empty model weights: {path}")


def prepare_models(root, state):
    from huggingface_hub import HfApi, snapshot_download

    for key, (repo_id, directory) in MODELS.items():
        entry = state.setdefault("models", {}).setdefault(key, {"repo_id": repo_id})
        if "revision" not in entry:
            entry["revision"] = HfApi().model_info(repo_id).sha
            atomic_json(root / ".preparation/assets.json", state)
        path = root / "models" / directory
        snapshot_download(repo_id, revision=entry["revision"], local_dir=str(path))
        validate_model(path, require_processor=key == "base")
        entry["path"] = str(path.relative_to(root))
        atomic_json(root / ".preparation/assets.json", state)
        print(f"{key} model ready: {path} @ {entry['revision']}", flush=True)


def prepare_data(root):
    downloads = root / "downloads"
    bench = root / "bench_data"
    download_http(
        SOURCES["textvqa_json"], bench / "textvqa/TextVQA_0.5.1_val.json", "json"
    )
    for key, archive_name, image_dir in [
        ("textvqa_images", "textvqa-images.zip", bench / "textvqa/train_images"),
        ("gqa_images", "gqa-images.zip", bench / "gqa/images"),
    ]:
        archive = downloads / archive_name
        download_http(SOURCES[key], archive, "zip")
        extract_images(archive, image_dir)
    archive = downloads / "gqa-questions.zip"
    download_http(SOURCES["gqa_questions"], archive, "zip")
    extract_member(
        archive,
        "testdev_balanced_questions.json",
        bench / "gqa/testdev_balanced_questions.json",
    )
    prompts = downloads / "llava-eval.zip"
    download_prompts(prompts)
    extract_member(
        prompts,
        "llava_gqa_testdev_balanced.jsonl",
        bench / "gqa/llava_gqa_testdev_balanced.jsonl",
    )
    download_http(
        SOURCES["textvqa_evaluator"],
        root / "tools/downloaded/m4c_evaluator.py",
        "python",
    )
    return validate_benchmarks(root)


def cache_mmvet(check_only=False):
    from datasets import DownloadConfig, load_dataset

    # Default revision matches the unchanged evaluator's Hub ID and offline cache lookup.
    data = load_dataset(
        "whyu/mm-vet",
        split="test",
        download_config=DownloadConfig(local_files_only=check_only),
    )
    if len(data) == 0 or not {"id", "question", "image"}.issubset(data.column_names):
        raise ValueError("Unexpected MM-Vet schema or empty split")
    return {"test_rows": len(data), "fingerprint": data._fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing assets without downloads or installations",
    )
    args = parser.parse_args()
    root = args.repo_dir.resolve()
    configure(root)
    state_path = root / ".preparation/assets.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with (root / ".preparation/prepare.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another preparation process is using this repository"
            ) from error
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"schema_version": 1, "sources": SOURCES}
        )
        state["ready"] = False
        if args.check_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            for key, (_, directory) in MODELS.items():
                validate_model(
                    root / "models" / directory, require_processor=key == "base"
                )
            counts = validate_benchmarks(root)
        else:
            for key in (
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            ):
                os.environ.pop(key, None)
            for module in ("huggingface_hub", "datasets", "gdown"):
                try:
                    importlib.import_module(module)
                except ModuleNotFoundError as error:
                    raise RuntimeError(
                        f"Missing preparation dependency {module}. Install requirements.txt "
                        "with uv pip install --python .venv/bin/python -r requirements.txt "
                        "--extra-index-url https://download.pytorch.org/whl/cu128 "
                        "--index-strategy unsafe-best-match on a compute node before preparation"
                    ) from error
            atomic_json(state_path, state)
            counts = prepare_data(root)
            prepare_models(root, state)
        mmvet = cache_mmvet(check_only=args.check_only)
        state.update(
            ready=True,
            counts=counts,
            mmvet=mmvet,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        scorer = root / "tools/downloaded/m4c_evaluator.py"
        state["textvqa_evaluator_sha256"] = hashlib.sha256(
            scorer.read_bytes()
        ).hexdigest()
        if not args.check_only:
            atomic_json(state_path, state)
        print("Preparation complete. All assets are under:", root, flush=True)
        print("Next: bash experiment/00_baseline_original/run.sh", flush=True)
        print("Then: bash experiment/01_baseline_draft/run.sh", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Preparation failed: {error}\nRerun bash prepare.sh to resume. See the preparation log for details.",
            file=sys.stderr,
        )
        raise
