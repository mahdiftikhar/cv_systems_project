"""Offline scoring adapters. No network requests or judge calls are made here."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

TEXTVQA_SOURCE = (
    "https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/m4c_evaluator.py"
)
COCO_CIDER_METHOD = "cider_d_pycocoevalcap_1.2"
MMVET_SOURCE = "https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.py"

# reverse_summary: ROUGE-1 F1 (the per-window score) and BERTScore F1, both 0-100 means.
REPO_DIR = Path(__file__).resolve().parents[2]
REVERSE_ASSETS = REPO_DIR / ".preparation/assets_reverse.json"
BERT_MODEL_DIR = REPO_DIR / "models/bert-base-uncased"
BERT_NUM_LAYERS = 9  # bert-score's default layer for bert-base-uncased
BERT_BATCH_SIZE = 64
CORPUS_BENCHMARKS = ("coco_caption", "reverse_summary")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Built from the installed distributions (bert_score.__version__ still says 0.3.12 in the
# 0.3.13 release), so a version change also changes the method that compare_metrics checks.
ROUGE_METHOD = f"rouge1_f1_rouge-score_{package_version('rouge-score')}_porter_stemmer"
BERTSCORE_METHOD = (
    f"bertscore_f1_bert-score_{package_version('bert-score')}_bert-base-uncased"
    f"_layer{BERT_NUM_LAYERS}_no_idf_no_rescaling"
)


def summary_pending(reason, scored=0):
    """A reverse_summary corpus result with both scores pending for ``reason``."""
    return {
        "rouge1_f1": None,
        "bertscore_f1": None,
        "scored": scored,
        "rouge_scored": 0,
        "bertscore_scored": 0,
        "method": ROUGE_METHOD,
        "rouge_method": ROUGE_METHOD,
        "bertscore_method": BERTSCORE_METHOD,
        "pending_reason": reason,
        "bertscore_pending_reason": reason,
        "bertscore": None,
        "per_sample": {},
    }


def resolve_bertscore_device(device):
    """``auto`` (or None) is CUDA when a GPU is visible, else CPU; other values pass through."""
    if device in (None, "", "auto"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def evidence(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def load_annotations(path):
    if not path:
        return {}
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        rows = json.loads(path.read_text())
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    if isinstance(rows, dict):
        return {str(key): value for key, value in rows.items()}
    result = {}
    for row in rows:
        key = str(
            row.get("question_id", row.get("questionId", row.get("qid", row.get("id"))))
        )
        if key == "None" or key in result:
            raise ValueError(f"Missing/duplicate annotation ID in {path}: {key}")
        result[key] = row
    return result


def answer_list(value):
    if isinstance(value, dict):
        value = value.get("answers", value.get("answer", value.get("label")))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def textvqa_score(prediction, answers, normalize):
    """Ten-reference, leave-one-out VQA consensus; use upstream normalization."""
    if len(answers) != 10:
        raise ValueError("TextVQA requires exactly ten reference answers per question")
    prediction = normalize(prediction)
    answers = [normalize(answer) for answer in answers]
    matches = answers.count(prediction)
    return (
        sum(min(1.0, (matches - (answer == prediction)) / 3) for answer in answers) / 10
    )


class Scorer:
    def __init__(
        self,
        textvqa_evaluator=None,
        textvqa_annotations=None,
        gqa_annotations=None,
        mmvet_grades=None,
        mmvet_graded_predictions=None,
        bert_model=None,
        bertscore_device=None,
    ):
        # BERTScore loads lazily, only when reverse_summary is scored.
        self.bert_model = Path(bert_model) if bert_model else BERT_MODEL_DIR
        self.bertscore_device = bertscore_device or "auto"
        self.sources = {}
        self.annotations = {}
        for benchmark, path in [
            ("textvqa", textvqa_annotations),
            ("gqa", gqa_annotations),
        ]:
            self.annotations[benchmark] = load_annotations(path)
            if path:
                self.sources[benchmark + "_annotations"] = evidence(path)
        self.normalize = None
        if textvqa_evaluator:
            spec = importlib.util.spec_from_file_location(
                "textvqa_external_evaluator", textvqa_evaluator
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.normalize = module.EvalAIAnswerProcessor()
            self.sources["textvqa_evaluator"] = {
                **evidence(textvqa_evaluator),
                "upstream": TEXTVQA_SOURCE,
            }
        self.grades = {}
        self.graded_predictions = {}
        if mmvet_grades:
            if not mmvet_graded_predictions:
                raise ValueError(
                    "--mmvet-grades requires --mmvet-graded-predictions to verify answer identity"
                )
            self.grades = json.loads(Path(mmvet_grades).read_text())
            self.graded_predictions = json.loads(
                Path(mmvet_graded_predictions).read_text()
            )
            self.sources["mmvet_grades"] = {
                **evidence(mmvet_grades),
                "upstream": MMVET_SOURCE,
            }
            self.sources["mmvet_graded_predictions"] = evidence(
                mmvet_graded_predictions
            )

    def score_corpus(self, benchmark, predictions, references):
        """Corpus-level scores; pending (never raising) if unavailable.

        COCO captions get CIDEr-D. reverse_summary gets ROUGE-1 F1 and BERTScore F1 (see
        ``score_summaries``).
        """
        if benchmark == "reverse_summary":
            return self.score_summaries(predictions, references)

        def pending(reason, scored=0):
            return {
                "score": None,
                "scored": scored,
                "method": COCO_CIDER_METHOD,
                "pending_reason": reason,
                "per_sample": {},
            }

        if benchmark != "coco_caption":
            return pending(f"corpus scoring is not defined for {benchmark}")
        if shutil.which("java") is None:
            return pending(
                "pending: java not found on PATH (PTBTokenizer needs Java for CIDEr)"
            )
        try:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        except ImportError as error:
            return pending(f"pending: pycocoevalcap is missing ({error})")
        ids = [key for key in predictions if references.get(key)]
        if not ids:
            return pending("pending: no predictions with reference captions")
        try:
            tokenizer = PTBTokenizer()
            gts = tokenizer.tokenize(
                {key: [{"caption": ref} for ref in references[key]] for key in ids}
            )
            res = tokenizer.tokenize({key: [{"caption": predictions[key]}] for key in ids})
            score, per_sample = Cider().compute_score(gts, res)
        except Exception as error:  # Java/tokenizer failures must not abort reporting
            return pending(f"pending: CIDEr scoring failed ({type(error).__name__}: {error})")
        return {
            "score": float(score),
            "scored": len(ids),
            "method": COCO_CIDER_METHOD,
            "pending_reason": None,
            "per_sample": {key: float(value) for key, value in zip(ids, per_sample)},
        }

    def score_summaries(self, predictions, references):
        """ROUGE-1 F1 and BERTScore F1 of each summary against its one joined reference.

        ``predictions`` maps question IDs to summaries and ``references`` maps them to a
        one-item list. ``rouge1_f1`` and ``bertscore_f1`` are 0-100 means over the scored
        windows; ``per_sample`` holds 0-1 fractions. Each metric is pending on its own
        (missing package, weights or failure), and nothing here raises.
        """
        ids = [key for key in predictions if references.get(key)]
        if not ids:
            return summary_pending("pending: no predictions with reference text")
        if any(len(references[key]) != 1 for key in ids):
            return summary_pending("pending: reverse_summary needs one joined reference per window")
        candidates = [predictions[key] for key in ids]
        targets = [references[key][0] for key in ids]
        result = summary_pending(None, scored=len(ids))
        per_sample = {key: {"rouge1_f1": None, "bertscore_f1": None} for key in ids}

        try:
            from rouge_score import rouge_scorer

            rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            for key, candidate, target in zip(ids, candidates, targets):
                per_sample[key]["rouge1_f1"] = float(
                    rouge.score(target, candidate)["rouge1"].fmeasure
                )
            result["rouge1_f1"] = 100 * statistics.mean(
                per_sample[key]["rouge1_f1"] for key in ids
            )
            result["rouge_scored"] = len(ids)
        except Exception as error:  # a missing package must not abort reporting
            result["pending_reason"] = (
                f"pending: ROUGE-1 scoring failed ({type(error).__name__}: {error})"
            )

        bert, reason = self.bertscore(candidates, targets)
        result["bertscore_pending_reason"] = reason
        if bert is not None:
            for key, value in zip(ids, bert.pop("f1")):
                per_sample[key]["bertscore_f1"] = value
            result["bertscore_f1"] = 100 * statistics.mean(
                per_sample[key]["bertscore_f1"] for key in ids
            )
            result["bertscore_scored"] = len(ids)
            result["bertscore"] = bert
        result["per_sample"] = per_sample
        return result

    def bertscore(self, candidates, targets):
        """(details with per-pair ``f1`` fractions, None) or (None, pending reason).

        Loads the local bert-base-uncased folder offline, checks its weights against the
        prepared manifest, and scores with bert-score's own truncation to 512 word pieces.
        """
        path = self.bert_model
        weights = path / "model.safetensors"
        try:
            prepared = json.loads(REVERSE_ASSETS.read_text(encoding="utf-8"))["bert"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            return None, (
                f"pending: {REVERSE_ASSETS} has no bert entry ({type(error).__name__}); "
                "run bash prepare.sh --suite reverse"
            )
        if not weights.is_file():
            return None, f"pending: BERT weights not found at {weights}"
        weights_evidence = evidence(weights)
        if weights_evidence["sha256"] != prepared.get("weights_sha256"):
            return None, (
                f"pending: {weights} differs from the prepared bert-base-uncased "
                f"weights ({prepared.get('weights_sha256')})"
            )
        self.sources["bertscore_weights"] = {
            **weights_evidence,
            "revision": prepared.get("revision"),
        }
        # Offline: the model loads from the local folder and must never reach the Hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            device = resolve_bertscore_device(self.bertscore_device)
            from bert_score import score as bert_score
            from bert_score.utils import get_tokenizer

            tokenizer = get_tokenizer(str(path), False)  # what bert-score itself loads
            limit = tokenizer.model_max_length

            def pieces(text):
                return len(tokenizer.tokenize(text)) + 2  # with [CLS] and [SEP]

            started = time.perf_counter()
            _, _, f1 = bert_score(
                candidates,
                targets,
                model_type=str(path),
                num_layers=BERT_NUM_LAYERS,
                idf=False,
                device=device,
                batch_size=BERT_BATCH_SIZE,
                lang=None,
                rescale_with_baseline=False,
                verbose=False,
            )
            seconds = time.perf_counter() - started
        except Exception as error:  # missing package, device or weights: stay pending
            return None, f"pending: BERTScore failed ({type(error).__name__}: {error})"
        reference_pieces = [pieces(text) for text in targets]
        candidate_pieces = [pieces(text) for text in candidates]
        return {
            "f1": [float(value) for value in f1.tolist()],
            "model_path": str(path.resolve()),
            "revision": prepared.get("revision"),
            "weights_sha256": weights_evidence["sha256"],
            "num_layers": BERT_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
            "batch_size": BERT_BATCH_SIZE,
            "device": device,
            "seconds": seconds,
            "max_word_pieces": limit,
            "truncated_references": sum(n > limit for n in reference_pieces),
            "truncated_predictions": sum(n > limit for n in candidate_pieces),
            "reference_word_pieces_max": max(reference_pieces),
            "prediction_word_pieces_max": max(candidate_pieces),
        }, None

    def score(self, benchmark, question_id, prediction, embedded_references):
        if benchmark in CORPUS_BENCHMARKS:
            return None, f"pending: {benchmark} is scored as a corpus (score_corpus)"
        annotation = self.annotations.get(benchmark, {}).get(question_id)
        refs = (
            answer_list(annotation) if annotation is not None else embedded_references
        )
        if benchmark == "textvqa":
            if self.normalize is None:
                return (
                    None,
                    "pending: supply --textvqa-evaluator (EvalAIAnswerProcessor)",
                )
            if not refs:
                return None, "pending: missing reference answers"
            return textvqa_score(
                prediction, refs, self.normalize
            ), "VQA consensus (0–1)"
        if benchmark == "gqa":
            if len(refs) != 1:
                return None, "pending: supply GQA answer annotations"
            return float(
                prediction.strip().lower() == refs[0].strip().lower()
            ), "exact match, strip/lowercase (0–1)"
        grade = self.grades.get(question_id)
        if grade is None:
            return None, "pending: import official MM-Vet judge grades"
        if self.graded_predictions.get(question_id) != prediction:
            raise ValueError(
                f"MM-Vet grades do not match the generated answer: {question_id}"
            )
        scores = grade.get("score", [])
        if not scores or any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            for score in scores
        ):
            raise ValueError(f"Invalid MM-Vet judge scores for {question_id}")
        if len(grade.get("model", [])) != len(scores) or not all(grade["model"]):
            raise ValueError(
                f"MM-Vet grades need a judge model ID for every score: {question_id}"
            )
        return sum(scores) / len(scores), "MM-Vet judge mean (0–1)"

    def protocol(self, benchmark, question_id):
        if benchmark == "textvqa":
            return {
                "rule": "VQA consensus",
                "normalizer_sha256": self.sources.get("textvqa_evaluator", {}).get(
                    "sha256"
                ),
            }
        if benchmark == "gqa":
            return {"rule": "strip/lowercase exact match, v1"}
        if benchmark == "coco_caption":
            return {"rule": "CIDEr-D (corpus IDF)", "method": COCO_CIDER_METHOD}
        if benchmark == "reverse_summary":
            # No device here: CPU and GPU BERTScore reports stay comparable.
            return {
                "rule": "ROUGE-1 F1 per window (the score); BERTScore F1 alongside",
                "rouge_method": ROUGE_METHOD,
                "bertscore_method": BERTSCORE_METHOD,
            }
        grade = self.grades.get(question_id, {})
        return {
            "rule": "MM-Vet judge mean",
            "judges": sorted(set(grade.get("model", []))),
            "judge_runs": len(grade.get("score", [])),
        }
