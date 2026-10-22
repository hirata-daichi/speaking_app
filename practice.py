"""Per-browser practice state and result presentation."""
from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from feedback import EvaluationOutcome, QuestionResult, ServiceError, validate_choices


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    level: str
    topic: str
    source: str


@dataclass(frozen=True)
class Snapshot:
    revision: str
    question: Question
    transcript: str


@dataclass
class Attempt:
    question: Question
    transcript: str
    outcome: EvaluationOutcome
    timestamp: str = field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%m/%d %H:%M:%S"))


class PracticeSession:
    def __init__(self, revision=None, question=None, transcript="", outcome=None, history=None, previous_questions=None):
        self.revision = revision or uuid4().hex
        self.question: Question | None = question
        self.transcript = transcript
        self.outcome: EvaluationOutcome | None = outcome
        self.history: list[Attempt] = history if history is not None else []
        self.previous_questions = previous_questions if previous_questions is not None else {}
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        with self._lock:
            clone = PracticeSession(
                revision=self.revision, question=self.question, transcript=self.transcript,
                outcome=copy.deepcopy(self.outcome, memo), history=copy.deepcopy(self.history, memo),
                previous_questions=copy.deepcopy(self.previous_questions, memo),
            )
        memo[id(self)] = clone
        return clone

    def _clear_answer(self) -> None:
        self.revision = uuid4().hex
        self.transcript = ""
        self.outcome = None

    def reset_settings(self) -> None:
        with self._lock:
            self._clear_answer()
            self.question = None

    def begin_question(self, level: str, topic: str) -> tuple[str, str]:
        validate_choices(level, topic)
        with self._lock:
            self._clear_answer()
            self.question = None
            return self.revision, self.previous_questions.get((level, topic), "")

    def finish_question(self, revision: str, level: str, topic: str, result: QuestionResult) -> bool:
        with self._lock:
            if self.revision != revision:
                return False
            self.question = Question(uuid4().hex, result.text, level, topic, result.source)
            self.previous_questions[(level, topic)] = result.text
            return True

    def clear_answer(self) -> None:
        with self._lock:
            self._clear_answer()

    def edit_transcript(self, text: str | None) -> None:
        with self._lock:
            self.revision = uuid4().hex
            self.transcript = text or ""
            self.outcome = None

    def snapshot(self, transcript: str | None = None) -> Snapshot:
        with self._lock:
            if self.question is None:
                raise ServiceError("先に「新しい質問」を押してください。")
            if transcript is not None and transcript != self.transcript:
                self.edit_transcript(transcript)
            return Snapshot(self.revision, self.question, self.transcript)

    def is_current(self, snapshot: Snapshot) -> bool:
        with self._lock:
            return self.revision == snapshot.revision and self.question == snapshot.question

    def save_transcript(self, snapshot: Snapshot, text: str) -> bool:
        with self._lock:
            if not self.is_current(snapshot):
                return False
            self.transcript = text
            self.outcome = None
            return True

    def save_evaluation(self, snapshot: Snapshot, outcome: EvaluationOutcome) -> bool:
        with self._lock:
            if not self.is_current(snapshot) or self.transcript != snapshot.transcript:
                return False
            self.outcome = outcome
            self.history.append(Attempt(snapshot.question, snapshot.transcript, outcome))
            self.history = self.history[-50:]
            return True

    def replace_sample(self, snapshot: Snapshot, original: EvaluationOutcome, answer: str) -> bool:
        with self._lock:
            if not self.is_current(snapshot) or self.outcome is not original:
                return False
            original.report.model_answer = answer
            original.notices = [n for n in original.notices if not n.startswith("模範回答を取得できませんでした")]
            return True

    def history_rows(self) -> list[list[str]]:
        with self._lock:
            return [[
                item.timestamp, item.question.text, item.question.level,
                f"{item.outcome.report.scores.total} / 20" if item.outcome.quality_checked else "確認待ち",
                item.transcript,
            ] for item in reversed(self.history)]

    def clear_history(self) -> None:
        with self._lock:
            self.history.clear()

    def comparison(self) -> str:
        with self._lock:
            if not self.question:
                return "同じ質問に再挑戦すると、前回の回答と比較できます。"
            attempts = [item for item in self.history if item.question.id == self.question.id]
            if not attempts:
                return "この質問の回答はまだありません。"
            last = attempts[-1]
            previous = attempts[-2] if len(attempts) > 1 else None
            if not previous:
                return f"**前回の回答**\n\n{escape_markdown(last.transcript)}\n\n「同じ質問に再挑戦」で回答し直せます。"
            def score(item):
                return f"{item.outcome.report.scores.total} / 20" if item.outcome.quality_checked else "確認待ち"
            return (
                f"**前回の回答（{score(previous)}）**\n\n{escape_markdown(previous.transcript)}\n\n"
                f"**今回の回答（{score(last)}）**\n\n{escape_markdown(last.transcript)}\n\n"
                "点数はAIによる目安です。回答内容や具体的な修正点も比較してください。"
            )


def escape_markdown(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+!|<>])", r"\\\1", value)


def correction_table(items) -> str:
    if not items:
        return "提案はありません。"
    rows = ["| 元の表現 | 修正・提案例 | 理由 |", "| --- | --- | --- |"]
    for item in items:
        cells = [escape_markdown(value).replace("\n", " ") for value in (item.original, item.corrected, item.reason_ja)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def format_result(outcome: EvaluationOutcome) -> str:
    result = outcome.report
    notices = "\n\n".join(escape_markdown(text) for text in outcome.notices)
    sample = escape_markdown(result.model_answer) if result.model_answer else "未取得です。「模範回答だけ再取得」を利用できます。"
    if not outcome.quality_checked:
        return f"{notices}\n\n### 元の回答\n\n{escape_markdown(result.corrected_answer)}\n\n### 模範回答\n\n{sample}"
    scores = result.scores
    feedback = result.feedback_ja
    return f"""{notices}

### スコア：{scores.total} / 20

| 項目 | 点数 |
| --- | --- |
| 質問との関連性 | {scores.relevance} / 5 |
| 文法 | {scores.grammar} / 5 |
| 語彙 | {scores.vocabulary} / 5 |
| 自然さ | {scores.naturalness} / 5 |

### 添削後の回答

{escape_markdown(result.corrected_answer)}

### 文法の修正

{correction_table(result.key_corrections)}

### 表現の提案（任意）

{correction_table(result.expression_suggestions)}

### フィードバック

**良い点：** {escape_markdown(feedback.good_points)}

**文法：** {escape_markdown(feedback.grammar_points)}

**語彙・表現：** {escape_markdown(feedback.vocabulary_points)}

**次の目標：** {escape_markdown(feedback.next_step)}

### 模範回答

{sample}
""".strip()
