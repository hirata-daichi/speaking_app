import copy
import threading
import unittest

from feedback import QuestionResult, ServiceError
from practice import PracticeSession, format_result
from tests.common import QUESTION, TEXT, outcome


def new_session():
    session = PracticeSession()
    revision, _ = session.begin_question("Beginner", "Daily Life")
    session.finish_question(revision, "Beginner", "Daily Life", QuestionResult(QUESTION))
    return session


class PracticeStateTests(unittest.TestCase):
    def test_question_keeps_original_level_and_topic(self):
        session = new_session()
        self.assertEqual(session.snapshot().question.level, "Beginner")
        self.assertEqual(session.snapshot().question.topic, "Daily Life")

    def test_new_question_clears_answer_and_result(self):
        session = new_session()
        snapshot = session.snapshot(TEXT)
        session.save_evaluation(snapshot, outcome())
        revision, previous = session.begin_question("Beginner", "Daily Life")
        self.assertEqual(previous, QUESTION)
        self.assertIsNone(session.question)
        self.assertEqual(session.transcript, "")
        self.assertIsNone(session.outcome)
        self.assertEqual(len(session.history), 1)
        self.assertTrue(session.finish_question(revision, "Beginner", "Daily Life", QuestionResult("What is your favorite part of the day?")))

    def test_settings_change_invalidates_question_and_old_result(self):
        session = new_session()
        snapshot = session.snapshot(TEXT)
        session.reset_settings()
        self.assertFalse(session.save_evaluation(snapshot, outcome()))
        self.assertEqual(session.history_rows(), [])
        with self.assertRaises(ServiceError):
            session.snapshot()

    def test_old_generation_cannot_replace_new_question(self):
        session = PracticeSession()
        old_revision, _ = session.begin_question("Beginner", "Food")
        new_revision, _ = session.begin_question("Advanced", "Travel")
        self.assertTrue(session.finish_question(new_revision, "Advanced", "Travel", QuestionResult("How can tourism help local communities?")))
        self.assertFalse(session.finish_question(old_revision, "Beginner", "Food", QuestionResult("What food do you like?")))
        self.assertEqual(session.question.level, "Advanced")

    def test_audio_change_invalidates_transcription_in_flight(self):
        session = new_session()
        snapshot = session.snapshot()
        session.clear_answer()
        self.assertFalse(session.save_transcript(snapshot, "old audio text"))
        self.assertEqual(session.transcript, "")

    def test_edit_during_evaluation_keeps_new_text_and_discards_old_result(self):
        session = new_session()
        snapshot = session.snapshot(TEXT)
        session.edit_transcript("I went to a different park.")
        self.assertFalse(session.save_evaluation(snapshot, outcome()))
        self.assertEqual(session.transcript, "I went to a different park.")

    def test_retry_preserves_question_and_previous_attempt(self):
        session = new_session()
        session.save_evaluation(session.snapshot(TEXT), outcome())
        question = session.question
        session.clear_answer()
        self.assertEqual(session.question, question)
        self.assertEqual(session.transcript, "")
        self.assertIn(TEXT, session.comparison())
        self.assertEqual(len(session.history), 1)

    def test_two_attempts_have_comparison_and_bounded_history(self):
        session = new_session()
        session.save_evaluation(session.snapshot(TEXT), outcome())
        session.clear_answer()
        session.save_evaluation(session.snapshot("I also enjoyed the weather."), outcome())
        self.assertIn("前回の回答", session.comparison())
        self.assertIn("今回の回答", session.comparison())
        self.assertIn("I also enjoyed the weather.", session.comparison())
        for i in range(55):
            session.save_evaluation(session.snapshot(f"Answer {i}"), outcome())
        self.assertEqual(len(session.history_rows()), 50)

    def test_sample_retry_preserves_score_and_history(self):
        session = new_session()
        result = outcome()
        result.report.model_answer = ""
        result.notices = ["模範回答を取得できませんでした。採点結果は保持されています。"]
        snapshot = session.snapshot(TEXT)
        session.save_evaluation(snapshot, result)
        self.assertTrue(session.replace_sample(snapshot, result, "I went hiking. It was fun."))
        self.assertEqual(result.report.scores.total, 19)
        self.assertEqual(result.notices, [])
        self.assertEqual(len(session.history), 1)
        session.clear_answer()
        self.assertFalse(session.replace_sample(snapshot, result, "Stale answer."))

    def test_sessions_and_deep_copies_do_not_share_history_or_question_state(self):
        template = PracticeSession()
        first = copy.deepcopy(template)
        second = copy.deepcopy(template)
        revision, _ = first.begin_question("Beginner", "Food")
        first.finish_question(revision, "Beginner", "Food", QuestionResult("What food do you like?"))
        first.save_evaluation(first.snapshot(TEXT), outcome())
        self.assertIsNone(second.question)
        self.assertEqual(second.history, [])
        self.assertEqual(template.history, [])

    def test_actual_thread_completion_after_reset_is_discarded(self):
        session = new_session()
        snapshot = session.snapshot(TEXT)
        started = threading.Event()
        resume = threading.Event()
        accepted = []
        def delayed_completion():
            started.set()
            resume.wait(2)
            accepted.append(session.save_evaluation(snapshot, outcome()))
        worker = threading.Thread(target=delayed_completion)
        worker.start()
        self.assertTrue(started.wait(2))
        session.reset_settings()
        resume.set()
        worker.join(2)
        self.assertEqual(accepted, [False])
        self.assertIsNone(session.outcome)

    def test_history_erasure_keeps_current_question(self):
        session = new_session()
        session.save_evaluation(session.snapshot(TEXT), outcome())
        session.clear_history()
        self.assertEqual(session.history_rows(), [])
        self.assertEqual(session.question.text, QUESTION)


class PresentationTests(unittest.TestCase):
    def test_style_suggestions_have_distinct_optional_section(self):
        rendered = format_result(outcome())
        self.assertIn("文法の修正", rendered)
        self.assertIn("表現の提案（任意）", rendered)

    def test_learner_markdown_cannot_create_remote_image_or_html(self):
        result = outcome()
        result.report.corrected_answer = '![image](https://example.com/pixel) <script>alert(1)</script>'
        rendered = format_result(result)
        self.assertNotIn('![image](', rendered)
        self.assertNotIn('<script>', rendered)


if __name__ == "__main__":
    unittest.main()
