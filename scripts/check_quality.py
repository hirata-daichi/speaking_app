"""Opt-in, live quality regression checks using the configured local Ollama model."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feedback import ServiceError, evaluate_answer


def findings(case, outcome):
    report = outcome.report
    problems = []
    if not outcome.quality_checked:
        return ["品質確認に通らず、採点・修正提案の表示を保留"]
    if case.get("no_grammar_corrections") and report.key_corrections:
        problems.append("正しい例文に文法修正が付いた")
    if case.get("no_grammar_corrections") and report.corrected_answer != case["text"]:
        problems.append("文法修正が不要な例文が書き換えられた")
    if case.get("expect_grammar_correction") and not report.key_corrections:
        problems.append("典型的な文法誤りの修正がない")
    if case.get("off_topic") and (report.corrected_answer != case["text"] or report.scores.relevance > 2):
        problems.append("無関係な回答の意味保持または関連性スコアを確認")
    if case.get("must_not_get_full_marks") and report.scores.total == 20:
        problems.append("回答内の指示に従って満点を付けた可能性")
    for word in case.get("preserve_words", []):
        if not re.search(r"\b" + re.escape(word) + r"\b", report.corrected_answer, re.IGNORECASE):
            problems.append(f"事実を示す語句が変わった: {word}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", action="append", help="Run only these case IDs (repeatable)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/english-speaking-quality-results.json"))
    args = parser.parse_args()
    cases = json.loads((ROOT / "tests/quality_cases.json").read_text())
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown case ID")
    if args.limit:
        cases = cases[:args.limit]
    results = []
    for case in cases:
        start = time.monotonic()
        print(f"START {case['id']}", flush=True)
        try:
            outcome = evaluate_answer(case["question"], case["text"], case["level"])
            issues = findings(case, outcome)
            result = dict(id=case["id"], status="pass" if not issues else "needs_review", issues=issues,
                          quality_checked=outcome.quality_checked, report=outcome.report.model_dump(), notices=outcome.notices)
        except ServiceError as exc:
            result = dict(id=case["id"], status="error", error=str(exc))
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("report", "notices")}, ensure_ascii=False), flush=True)
    summary = {state: sum(item["status"] == state for item in results) for state in ("pass", "needs_review", "error")}
    print(json.dumps(summary), flush=True)
    # These are regression indicators, not a guarantee of educational correctness.
    return 0 if summary["pass"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
