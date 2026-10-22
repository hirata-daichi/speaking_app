"""Gradio interface for English speaking practice."""
from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import gradio as gr

import feedback as services
from practice import PracticeSession, format_result
from question_bank import QUESTION_BANK

logger = logging.getLogger(__name__)
SAMPLE_AUDIO = Path(__file__).parent / "samples" / "weekend.wav"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="English Speaking Feedback", delete_cache=(3600, 86400)) as demo:
        session = gr.State(PracticeSession())
        gr.Markdown("""
# English Speaking Feedback

質問に英語で答え、文字起こしを確認してからフィードバックを受けましょう。
発音・イントネーション・音響的な流暢さは評価しません。点数と添削はAIによる目安です。
""")
        with gr.Accordion("初めて使う方へ", open=True):
            gr.Markdown("""
1. レベルとトピックを選び、「新しい質問」を押します。
2. 2分以内の英語回答を録音するか、音声ファイルを選びます。
3. 「文字起こし」を押し、聞き取り間違いだけを修正します。
4. 「この文章を採点」を押します。改善後は「同じ質問に再挑戦」を使えます。

録音を試す前に「サンプルで練習」も利用できます。
履歴はこのブラウザのセッション内に最大50件保持し、永続保存しません。
一時音声は定期削除します（作成から24時間経過後、1時間ごとに確認）。
""")
        with gr.Row():
            level = gr.Dropdown(list(QUESTION_BANK), value="Beginner", label="レベル", allow_custom_value=False)
            topic = gr.Dropdown(list(QUESTION_BANK["Beginner"]), value="Daily Life", label="トピック", allow_custom_value=False)
            use_llm = gr.Checkbox(False, label="AIで質問を生成する")
        with gr.Row():
            new_question = gr.Button("新しい質問", variant="primary")
            sample = gr.Button("サンプルで練習")
        question = gr.Textbox(label="質問", lines=2, interactive=False)
        question_details = gr.Markdown("レベルとトピックを選び、新しい質問を作成してください。")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="あなたの回答（2分・20MB以内）", interactive=False)
        transcribe = gr.Button("文字起こし", interactive=False)
        transcript = gr.Textbox(label="文字起こし（認識内容を確認・修正できます）", lines=4, max_length=services.MAX_TRANSCRIPT_LENGTH, interactive=False)
        with gr.Row():
            evaluate = gr.Button("この文章を採点", variant="primary", interactive=False)
            retry = gr.Button("同じ質問に再挑戦", interactive=False)
            retry_sample = gr.Button("模範回答だけ再取得", interactive=False)
        status = gr.Markdown("まず「新しい質問」または「サンプルで練習」を押してください。")
        feedback = gr.Markdown()
        with gr.Accordion("同じ質問への回答を比較", open=False):
            comparison = gr.Markdown("同じ質問に再挑戦すると、前回の回答と比較できます。")
        with gr.Accordion("このセッションの学習履歴（最大50件）", open=False):
            history = gr.Dataframe(headers=["時刻（日本時間）", "質問", "レベル", "点数", "回答"], datatype=["str"] * 5, interactive=False, type="array")
            clear_history = gr.Button("履歴を消去")
        with gr.Row():
            connection = gr.Markdown("採点にはOllamaとローカルモデルが必要です。")
            check_connection = gr.Button("接続を確認")

        changing = [question, question_details, audio, transcript, feedback, status, new_question, sample, transcribe, evaluate, retry, retry_sample, comparison, history, level, topic]

        def unchanged():
            return {status: gr.skip()}

        def reset_answer(current: PracticeSession):
            has_question = current.question is not None
            return {audio: gr.update(value=None, interactive=has_question), transcript: gr.update(value="", interactive=has_question), feedback: "", retry_sample: gr.update(interactive=False), comparison: current.comparison()}

        def busy(message):
            return {status: message, new_question: gr.update(interactive=False), sample: gr.update(interactive=False), transcribe: gr.update(interactive=False), evaluate: gr.update(interactive=False), retry: gr.update(interactive=False), retry_sample: gr.update(interactive=False)}

        def ready(current: PracticeSession, message: str):
            has_question = current.question is not None
            return {status: message, new_question: gr.update(interactive=True), sample: gr.update(interactive=True), audio: gr.update(interactive=has_question), transcript: gr.update(interactive=has_question), transcribe: gr.update(interactive=has_question), evaluate: gr.update(interactive=has_question), retry: gr.update(interactive=has_question), retry_sample: gr.update(interactive=bool(current.outcome and not current.outcome.report.model_answer))}

        def question_action(selected_level, selected_topic, use_ai, current):
            try:
                revision, previous = current.begin_question(selected_level, selected_topic)
            except services.ServiceError as exc:
                yield ready(current, str(exc))
                return
            yield {**reset_answer(current), **busy("質問を準備しています…"), question: "", question_details: "", audio: gr.update(value=None, interactive=False), transcript: gr.update(value="", interactive=False)}
            result = services.create_question(selected_level, selected_topic, use_ai, previous)
            if not current.finish_question(revision, selected_level, selected_topic, result):
                yield unchanged()
                return
            yield {**ready(current, result.notice or "録音するか、音声ファイルを選んでください。"), question: result.text, question_details: f"出題条件：{selected_level} / {selected_topic}", comparison: current.comparison()}

        def settings_action(current):
            current.reset_settings()
            return {**ready(current, "条件を変更しました。「新しい質問」を押してください。"), **reset_answer(current), question: "", question_details: "新しい条件で質問を作成してください。"}

        def audio_action(current):
            current.clear_answer()
            return {**ready(current, "「文字起こし」を押してください。"), transcript: gr.update(value="", interactive=current.question is not None), feedback: ""}

        def text_action(text, current):
            current.edit_transcript(text)
            return {feedback: "", **ready(current, "認識内容を確認して「この文章を採点」を押してください。")}

        def transcription_action(path, current):
            try:
                snapshot = current.snapshot()
            except services.ServiceError as exc:
                yield ready(current, str(exc))
                return
            yield {**busy("音声を文字起こししています。初回はモデルの準備に時間がかかります…"), feedback: ""}
            try:
                text = services.transcribe_audio(path)
            except services.ServiceError as exc:
                yield ready(current, str(exc)) if current.is_current(snapshot) else unchanged()
                return
            if not current.save_transcript(snapshot, text):
                yield unchanged()
                return
            yield {**ready(current, "聞き取り間違いがあれば修正し、「この文章を採点」を押してください。"), transcript: text}

        def evaluation_action(text, current):
            try:
                snapshot = current.snapshot(text or "")
                if not snapshot.transcript.strip():
                    raise services.ServiceError("先に文字起こしを行うか、英語の回答を入力してください。")
            except services.ServiceError as exc:
                yield ready(current, str(exc))
                return
            yield {**busy("英文を採点し、修正提案の整合性を確認しています…"), feedback: ""}
            try:
                result = services.evaluate_answer(snapshot.question.text, snapshot.transcript, snapshot.question.level)
            except services.ServiceError as exc:
                yield ready(current, str(exc)) if current.is_current(snapshot) else unchanged()
                return
            if not current.save_evaluation(snapshot, result):
                yield unchanged()
                return
            yield {**ready(current, "採点が完了しました。" if result.quality_checked else "添削内容の確認が必要です。回答を保持したまま再評価できます。"), feedback: format_result(result), history: current.history_rows(), comparison: current.comparison()}

        def sample_answer_action(current):
            try:
                snapshot = current.snapshot()
                original = current.outcome
                if original is None:
                    raise services.ServiceError("先に回答を採点してください。")
            except services.ServiceError as exc:
                yield ready(current, str(exc))
                return
            yield busy("模範回答だけを再取得しています…")
            try:
                answer = services.generate_english_model_answer(snapshot.question.text, snapshot.question.level)
            except services.ServiceError as exc:
                yield ready(current, str(exc)) if current.is_current(snapshot) else unchanged()
                return
            if current.replace_sample(snapshot, original, answer):
                yield {**ready(current, "模範回答を更新しました。"), feedback: format_result(original)}
            else:
                yield unchanged()

        def retry_action(current):
            current.clear_answer()
            return {**ready(current, "同じ質問にもう一度答えてみましょう。前回の回答は比較欄に残っています。"), **reset_answer(current)}

        def example_action(current):
            revision, _ = current.begin_question("Beginner", "Daily Life")
            result = services.QuestionResult("What did you do last weekend?")
            current.finish_question(revision, "Beginner", "Daily Life", result)
            return {**ready(current, "サンプル音声を読み込みました。「文字起こし」を押してください。"), **reset_answer(current), question: result.text, question_details: "出題条件：Beginner / Daily Life（サンプル）", audio: gr.update(value=str(SAMPLE_AUDIO), interactive=True), level: "Beginner", topic: "Daily Life"}

        common = dict(outputs=changing, concurrency_id="local_models", concurrency_limit=1, api_visibility="private")
        transcription_event = transcribe.click(transcription_action, inputs=[audio, session], **common)
        evaluation_event = evaluate.click(evaluation_action, inputs=[transcript, session], **common)
        sample_answer_event = retry_sample.click(sample_answer_action, inputs=session, **common)
        model_events = [transcription_event, evaluation_event, sample_answer_event]
        question_event = new_question.click(question_action, inputs=[level, topic, use_llm, session], cancels=model_events, **common)
        all_events = model_events + [question_event]
        level.input(settings_action, inputs=session, outputs=changing, queue=False, cancels=all_events, api_visibility="private")
        topic.input(settings_action, inputs=session, outputs=changing, queue=False, cancels=all_events, api_visibility="private")
        audio.input(audio_action, inputs=session, outputs=changing, queue=False, cancels=model_events, api_visibility="private")
        transcript.input(text_action, inputs=[transcript, session], outputs=changing, queue=False, cancels=[evaluation_event, sample_answer_event], api_visibility="private")
        retry.click(retry_action, inputs=session, outputs=changing, queue=False, cancels=all_events, api_visibility="private")
        sample.click(example_action, inputs=session, outputs=changing, queue=False, cancels=all_events, api_visibility="private")

        def erase_history(current):
            current.clear_history()
            return [], current.comparison()
        clear_history.click(erase_history, inputs=session, outputs=[history, comparison], queue=False, api_visibility="private")
        check_connection.click(services.backend_status, outputs=connection, api_visibility="private")
    demo.queue(max_size=16, default_concurrency_limit=1)
    return demo


def get_server_port() -> int:
    configured = os.getenv("GRADIO_SERVER_PORT")
    if configured:
        try:
            port = int(configured)
        except ValueError as exc:
            raise ValueError("GRADIO_SERVER_PORT は1〜65535の整数で指定してください。") from exc
        if not 1 <= port <= 65535:
            raise ValueError("GRADIO_SERVER_PORT は1〜65535の整数で指定してください。")
        return port
    for port in range(8960, 9060):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("空きポートがありません。GRADIO_SERVER_PORTを指定してください。")


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_port=get_server_port(), max_file_size="20mb", footer_links=[], run_history=False)
