import copy
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

import feedback as services
from practice import format_result
from tests.common import QUESTION, TEXT, REVIEW, encoded, report_dict


class OutputValidationTests(unittest.TestCase):
    def test_valid_scores_have_computed_total(self):
        self.assertEqual(services.Evaluation.model_validate(report_dict()).scores.total, 19)

    def test_wrong_score_container_rejected(self):
        for value in (None, [], "bad"):
            with self.subTest(value=value):
                candidate = report_dict()
                candidate["scores"] = value
                with self.assertRaises(ValidationError):
                    services.Evaluation.model_validate_json(encoded(candidate))

    def test_fractional_boolean_string_out_of_range_scores_rejected(self):
        for value in (True, 4.9, 4.0, "4", 0, 6, 999, None):
            with self.subTest(value=value):
                candidate = report_dict()
                candidate["scores"]["grammar"] = value
                with self.assertRaises(ValidationError):
                    services.Evaluation.model_validate_json(encoded(candidate))

    def test_missing_required_fields_and_unknown_fields_rejected(self):
        for field in ("scores", "feedback_ja", "key_corrections", "expression_suggestions", "corrected_answer"):
            candidate = report_dict()
            del candidate[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                services.Evaluation.model_validate(candidate)
        candidate = report_dict()
        candidate["overall_score"] = 3
        with self.assertRaises(ValidationError):
            services.Evaluation.model_validate(candidate)

    def test_empty_or_wrong_corrections_rejected(self):
        for correction in ("wrong", {"original": "", "corrected": "", "reason_ja": ""}, {"original": TEXT}):
            candidate = report_dict()
            candidate["key_corrections"] = [correction]
            with self.subTest(correction=correction), self.assertRaises(ValidationError):
                services.Evaluation.model_validate(candidate)

    def test_quotation_must_exist_in_learner_text(self):
        candidate = report_dict()
        candidate["key_corrections"] = [{"original": "I went swimming", "corrected": "I swam", "reason_ja": "動詞を修正しました。"}]
        with patch.object(services, "call_ollama", return_value=encoded(candidate)) as call:
            with self.assertRaises(services.ModelOutputError):
                services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertEqual(call.call_count, 2)

    def test_invalid_json_retried_but_never_rendered_as_model_answer(self):
        with patch.object(services, "call_ollama", return_value="not JSON") as call:
            with self.assertRaises(services.ModelOutputError):
                services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertEqual(call.call_count, 2)

    def test_retry_can_recover_from_invalid_output(self):
        with patch.object(services, "call_ollama", side_effect=["{}", encoded(report_dict()), encoded(REVIEW)]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertTrue(result.quality_checked)
        self.assertEqual(result.report.scores.total, 19)

    def test_no_op_corrections_removed_instead_of_shown(self):
        candidate = report_dict()
        candidate["key_corrections"] = [{"original": "We played soccer", "corrected": "We played soccer", "reason_ja": "変更はありません。"}]
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(REVIEW)]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertTrue(result.quality_checked)
        self.assertEqual(result.report.key_corrections, [])


class EvaluationTests(unittest.TestCase):
    def test_normal_evaluation_and_schema_sent(self):
        with patch.object(services, "call_ollama", side_effect=[encoded(report_dict()), encoded(REVIEW)]) as call:
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertEqual(result.report.corrected_answer, TEXT)
        self.assertTrue(result.quality_checked)
        self.assertEqual(call.call_args_list[0].kwargs["schema"]["type"], "object")
        prompt_data = json.loads(call.call_args_list[0].args[0])
        self.assertEqual(prompt_data["learner_text"], TEXT)
        self.assertIn("Shared-subject", call.call_args_list[0].kwargs["system"])

    def test_off_topic_answer_preserves_original_facts(self):
        original = "My favorite food is pizza because it is delicious."
        candidate = report_dict()
        candidate["scores"]["relevance"] = 1
        candidate["corrected_answer"] = "I stayed at home and watched TV."
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(REVIEW)]):
            result = services.evaluate_answer(QUESTION, original, "Beginner")
        self.assertEqual(result.report.corrected_answer, original)
        self.assertEqual(result.report.key_corrections, [])
        self.assertNotIn("watched TV", format_result(result))

    def test_failed_quality_check_withholds_unreliable_feedback_and_scores(self):
        review = {**REVIEW, "grammar_feedback_supported": False}
        candidate = report_dict()
        candidate["feedback_ja"]["grammar_points"] = "UNSUPPORTED_GRAMMAR_ADVICE"
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(review), encoded(candidate), encoded(review)]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        rendered = format_result(result)
        self.assertFalse(result.quality_checked)
        self.assertNotIn("UNSUPPORTED_GRAMMAR_ADVICE", rendered)
        self.assertNotIn("19 / 20", rendered)
        self.assertIn(TEXT, rendered)

    def test_quality_rejection_regenerates_once_and_can_recover(self):
        bad_review = {**REVIEW, "feedback_consistent": False}
        with patch.object(services, "call_ollama", side_effect=[encoded(report_dict()), encoded(bad_review), encoded(report_dict()), encoded(REVIEW)]) as call:
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertTrue(result.quality_checked)
        self.assertEqual(call.call_count, 4)
        self.assertIn("review_notes_to_address", json.loads(call.call_args_list[2].args[0]))

    def test_quality_service_failure_preserves_report_and_retries_possible(self):
        with patch.object(services, "call_ollama", side_effect=[encoded(report_dict()), services.ServiceError("確認処理の接続失敗")]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertFalse(result.quality_checked)
        self.assertEqual(result.report.scores.total, 19)
        self.assertTrue(result.notices)

    def test_sample_retry_failure_preserves_grading(self):
        candidate = report_dict()
        candidate["model_answer"] = "日本語の回答です。"
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(REVIEW), services.ServiceError("timeout")]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertTrue(result.quality_checked)
        self.assertEqual(result.report.scores.total, 19)
        self.assertEqual(result.report.corrected_answer, TEXT)
        self.assertEqual(result.report.model_answer, "")
        self.assertIn("19 / 20", format_result(result))
        self.assertNotIn('"scores"', format_result(result))

    def test_sample_retry_rechecks_language(self):
        candidate = report_dict()
        candidate["model_answer"] = "日本語です。"
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(REVIEW), encoded({"answer": "再び日本語です。"}), encoded({"answer": "まだ日本語です。"})]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertEqual(result.report.model_answer, "")

    def test_missing_sample_can_be_generated_independently(self):
        candidate = report_dict()
        del candidate["model_answer"]
        answer = "I went shopping. I bought some fruit."
        with patch.object(services, "call_ollama", side_effect=[encoded(candidate), encoded(REVIEW), encoded({"answer": answer})]):
            result = services.evaluate_answer(QUESTION, TEXT, "Beginner")
        self.assertEqual(result.report.model_answer, answer)

    def test_empty_and_oversized_input_rejected_before_service(self):
        for text in ("", " " * 3, "a" * 6001):
            with self.subTest(length=len(text)), patch.object(services, "call_ollama") as call:
                with self.assertRaises(services.ServiceError):
                    services.evaluate_answer(QUESTION, text, "Beginner")
                call.assert_not_called()


class QuestionTests(unittest.TestCase):
    def test_fixed_questions_do_not_repeat_immediately(self):
        previous = ""
        for _ in range(20):
            current = services.create_question("Beginner", "Food", False, previous)
            self.assertNotEqual(current.text, previous)
            previous = current.text

    def test_structured_question_not_first_line_of_unstructured_preamble(self):
        with patch.object(services, "call_ollama", return_value=encoded({"question": "What food do you like?"})):
            result = services.create_question("Beginner", "Food", True)
        self.assertEqual(result.text, "What food do you like?")
        self.assertEqual(result.source, "llm")

    def test_generation_failure_has_separate_safe_notice(self):
        with patch.object(services, "call_ollama", side_effect=services.ServiceError("secret internal path")):
            result = services.create_question("Beginner", "Food", True)
        self.assertIn(result.text, services.QUESTION_BANK["Beginner"]["Food"])
        self.assertNotIn("secret", result.text + result.notice)
        self.assertTrue(result.notice)

    def test_empty_preamble_non_english_and_duplicate_questions_fall_back(self):
        previous = "What food do you like?"
        for raw in ("", encoded({"question": "Sure, here is a question:"}), encoded({"question": "日本語で好きな食べ物は何ですか？"}), encoded({"question": previous})):
            with self.subTest(raw=raw), patch.object(services, "call_ollama", return_value=raw):
                result = services.create_question("Beginner", "Food", True, previous)
            self.assertEqual(result.source, "fixed")
            self.assertNotEqual(result.text, previous)

    def test_invalid_choice_rejected(self):
        for level, topic in ((None, "Food"), ("Beginner", None), ("Injected level", "Food")):
            with self.subTest(level=level, topic=topic), self.assertRaises(services.ServiceError):
                services.create_question(level, topic, True)


class AudioAndTransportTests(unittest.TestCase):
    def test_transcription_joins_segments_and_uses_english_vad(self):
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(text=" Hello. "), SimpleNamespace(text=" It was fun. ")], None)
        audio = Path(__file__).resolve().parents[1] / "samples/weekend.wav"
        with patch.object(services, "get_whisper_model", return_value=model):
            self.assertEqual(services.transcribe_audio(str(audio)), "Hello. It was fun.")
        self.assertEqual(model.transcribe.call_args.kwargs, {"language": "en", "vad_filter": True})

    def test_empty_audio_and_unreadable_audio_have_safe_errors(self):
        with self.assertRaises(services.ServiceError):
            services.transcribe_audio(None)
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"not an audio file")
            audio.flush()
            with self.assertRaisesRegex(services.ServiceError, "音声を読み取れません"):
                services.transcribe_audio(audio.name)

    def test_long_audio_rejected_before_loading_whisper(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with wave.open(audio.name, "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(1000)
                target.writeframes(b"\x00\x00" * 121000)
            with patch.object(services, "get_whisper_model") as model:
                with self.assertRaisesRegex(services.ServiceError, "2分以内"):
                    services.transcribe_audio(audio.name)
                model.assert_not_called()

    def test_empty_transcription_has_retry_instruction(self):
        model = Mock()
        model.transcribe.return_value = ([], None)
        audio = Path(__file__).resolve().parents[1] / "samples/weekend.wav"
        with patch.object(services, "get_whisper_model", return_value=model), self.assertRaisesRegex(services.ServiceError, "文字起こしできません"):
            services.transcribe_audio(str(audio))

    def test_http_request_uses_schema_and_timeout(self):
        response = Mock()
        response.json.return_value = {"response": '{"question":"What is your favorite food?"}'}
        with patch.object(services.requests, "post", return_value=response) as post:
            services.call_ollama("input", schema={"type": "object"}, system="instructions")
        self.assertEqual(post.call_args.kwargs["json"]["format"], {"type": "object"})
        self.assertEqual(post.call_args.kwargs["timeout"], (5, services.LLM_TIMEOUT_SECONDS))

    def test_connection_and_timeout_messages_do_not_expose_internal_exception(self):
        for error in (services.requests.ConnectionError("secret internal path"), services.requests.Timeout("secret internal path")):
            with self.subTest(error=type(error).__name__), patch.object(services.requests, "post", side_effect=error):
                with self.assertRaises(services.ServiceError) as context:
                    services.call_ollama("input", schema={}, system="instructions")
                self.assertNotIn("secret", str(context.exception))

    def test_backend_status_handles_missing_model_and_offline(self):
        response = Mock()
        response.json.return_value = {"models": []}
        with patch.object(services.requests, "get", return_value=response):
            self.assertIn("準備", services.backend_status())
        with patch.object(services.requests, "get", side_effect=services.requests.ConnectionError):
            self.assertIn("接続できません", services.backend_status())


if __name__ == "__main__":
    unittest.main()
