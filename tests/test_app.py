import os
import unittest
from unittest.mock import patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import app
from feedback import QuestionResult, ServiceError
from practice import PracticeSession
from tests.common import QUESTION, TEXT, outcome


def callback(name):
    return next(item.fn for item in app.demo.fns.values() if item.fn and item.fn.__name__ == name)


def session_with_question():
    session = PracticeSession()
    revision, _ = session.begin_question("Beginner", "Daily Life")
    session.finish_question(revision, "Beginner", "Daily Life", QuestionResult(QUESTION))
    return session


class InterfaceFlowTests(unittest.TestCase):
    def test_fixed_question_flow_and_clear_outputs(self):
        session = PracticeSession()
        frames = list(callback("question_action")("Beginner", "Food", False, session))
        self.assertEqual(len(frames), 2)
        first_values = list(frames[0].values())
        self.assertIn("", first_values)
        self.assertIsNotNone(session.question)
        self.assertEqual(session.question.topic, "Food")

    def test_transcription_then_edited_text_is_used_for_grading_without_retranscription(self):
        session = session_with_question()
        with patch.object(app.services, "transcribe_audio", return_value="I go park yesterday.") as speech:
            list(callback("transcription_action")("sample.wav", session))
            callback("text_action")(TEXT, session)
            with patch.object(app.services, "evaluate_answer", return_value=outcome()) as grade:
                list(callback("evaluation_action")(TEXT, session))
        speech.assert_called_once()
        grade.assert_called_once_with(QUESTION, TEXT, "Beginner")
        self.assertEqual(session.transcript, TEXT)
        self.assertEqual(len(session.history), 1)

    def test_grading_failure_preserves_transcript_and_allows_retry(self):
        session = session_with_question()
        session.edit_transcript(TEXT)
        with patch.object(app.services, "evaluate_answer", side_effect=ServiceError("接続を確認してください。")):
            frames = list(callback("evaluation_action")(TEXT, session))
        self.assertEqual(session.transcript, TEXT)
        self.assertEqual(session.history, [])
        self.assertIn("接続を確認してください。", list(frames[-1].values()))
        with patch.object(app.services, "evaluate_answer", return_value=outcome()):
            list(callback("evaluation_action")(TEXT, session))
        self.assertEqual(len(session.history), 1)

    def test_stale_evaluation_does_not_render_or_save_result(self):
        session = session_with_question()
        def delayed_result(*args):
            session.reset_settings()
            return outcome()
        with patch.object(app.services, "evaluate_answer", side_effect=delayed_result):
            frames = list(callback("evaluation_action")(TEXT, session))
        self.assertEqual(session.history, [])
        self.assertIsNone(session.outcome)
        self.assertTrue(all(value == {"__type__": "update"} for value in frames[-1].values()))

    def test_stale_question_generation_does_not_override_settings(self):
        session = PracticeSession()
        def delayed_question(*args):
            session.reset_settings()
            return QuestionResult("What food do you like?")
        with patch.object(app.services, "create_question", side_effect=delayed_question):
            list(callback("question_action")("Beginner", "Food", True, session))
        self.assertIsNone(session.question)

    def test_sample_flow_and_retry(self):
        session = PracticeSession()
        values = callback("example_action")(session)
        self.assertTrue(app.SAMPLE_AUDIO.is_file())
        self.assertEqual(session.question.text, QUESTION)
        self.assertTrue(any(isinstance(value, dict) and value.get("value") == str(app.SAMPLE_AUDIO) for value in values.values()))
        session.save_evaluation(session.snapshot(TEXT), outcome())
        callback("retry_action")(session)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.transcript, "")
        self.assertEqual(session.question.text, QUESTION)

    def test_retry_and_settings_clear_actual_media_and_text_outputs(self):
        for action in ("retry_action", "settings_action"):
            with self.subTest(action=action):
                session = session_with_question()
                session.edit_transcript(TEXT)
                values = callback(action)(session)
                media = next(value for component, value in values.items() if component.get_block_name() == "audio")
                text = next(value for component, value in values.items() if component.label == "文字起こし（認識内容を確認・修正できます）")
                self.assertIsNone(media["value"])
                self.assertEqual(text["value"], "")
                self.assertEqual(media["interactive"], action == "retry_action")

    def test_new_audio_clears_displayed_old_transcript(self):
        session = session_with_question()
        session.edit_transcript(TEXT)
        values = callback("audio_action")(session)
        text = next(value for component, value in values.items() if component.label == "文字起こし（認識内容を確認・修正できます）")
        self.assertEqual(text["value"], "")
        self.assertEqual(session.transcript, "")

    def test_sample_answer_only_retry_does_not_repeat_grading(self):
        session = session_with_question()
        result = outcome()
        result.report.model_answer = ""
        session.save_evaluation(session.snapshot(TEXT), result)
        with patch.object(app.services, "generate_english_model_answer", return_value="I went hiking. It was fun.") as generate, patch.object(app.services, "evaluate_answer") as evaluate:
            list(callback("sample_answer_action")(session))
        generate.assert_called_once_with(QUESTION, "Beginner")
        evaluate.assert_not_called()
        self.assertEqual(session.outcome.report.scores.total, 19)
        self.assertEqual(len(session.history), 1)

    def test_sample_failure_preserves_grading_and_transcript(self):
        session = session_with_question()
        result = outcome()
        result.report.model_answer = ""
        session.save_evaluation(session.snapshot(TEXT), result)
        with patch.object(app.services, "generate_english_model_answer", side_effect=ServiceError("模範回答の取得失敗")):
            list(callback("sample_answer_action")(session))
        self.assertIs(session.outcome, result)
        self.assertEqual(session.transcript, TEXT)

    def test_untouched_form_has_safe_error(self):
        frames = list(callback("evaluation_action")(None, PracticeSession()))
        self.assertIn("先に", str(list(frames[0].values())))

    def test_http_page_and_config_work_with_new_dependencies(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import gradio as gr
        web = gr.mount_gradio_app(FastAPI(), app.demo, path="/", max_file_size="20mb", run_history=False)
        with TestClient(web) as client:
            self.assertEqual(client.get("/").status_code, 200)
            response = client.get("/config")
            self.assertEqual(response.status_code, 200)
            self.assertIn("components", response.json())

    def test_invalid_port_gets_actionable_error(self):
        for value in ("invalid", "0", "65536"):
            with self.subTest(value=value), patch.dict(os.environ, {"GRADIO_SERVER_PORT": value}), self.assertRaisesRegex(ValueError, "1〜65535"):
                app.get_server_port()


if __name__ == "__main__":
    unittest.main()
