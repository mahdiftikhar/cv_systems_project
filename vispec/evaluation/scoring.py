"""Offline scoring adapters. No network requests or judge calls are made here."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

TEXTVQA_SOURCE = (
    "https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/m4c_evaluator.py"
)
MMVET_SOURCE = "https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.py"


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
    ):
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

    def score(self, benchmark, question_id, prediction, embedded_references):
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
        grade = self.grades.get(question_id, {})
        return {
            "rule": "MM-Vet judge mean",
            "judges": sorted(set(grade.get("model", []))),
            "judge_runs": len(grade.get("score", [])),
        }
