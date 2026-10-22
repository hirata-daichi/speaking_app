import copy
import json

from feedback import Evaluation, EvaluationOutcome

QUESTION = "What did you do last weekend?"
TEXT = "I visited a park with my friends last weekend. We played soccer and had lunch together."
REPORT = {
    "scores": {"relevance": 5, "grammar": 5, "vocabulary": 4, "naturalness": 5},
    "corrected_answer": TEXT,
    "key_corrections": [],
    "expression_suggestions": [],
    "feedback_ja": {
        "good_points": "質問に明確に答えています。",
        "grammar_points": "明確な文法の誤りはありません。",
        "vocabulary_points": "適切な語彙を使えています。",
        "next_step": "次は感想を一文加えてみましょう。",
    },
    "model_answer": "I went hiking with my family. We enjoyed the sunshine.",
}
REVIEW = {"meaning_preserved": True, "grammar_feedback_supported": True, "feedback_consistent": True, "reason_ja": "元の意味と正しい文法が保たれています。"}


def report_dict():
    return copy.deepcopy(REPORT)


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def outcome(approved=True):
    return EvaluationOutcome(Evaluation.model_validate(report_dict()), approved, [] if approved else ["添削内容の確認が必要です。"])
