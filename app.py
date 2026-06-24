from __future__ import annotations

import json
import os
import random
import socket
from typing import Any, Optional

import gradio as gr
import requests
from faster_whisper import WhisperModel


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
GRADIO_SERVER_PORT = os.getenv("GRADIO_SERVER_PORT")


QUESTION_BANK = {
    "Beginner": {
        "Daily Life": [
            "What do you usually do after school or work?",
            "What did you do last weekend?",
            "What is your favorite part of the day?",
        ],
        "Food": [
            "What is your favorite food and why?",
            "Do you like cooking?",
            "What food do you often eat at home?",
        ],
        "Travel": [
            "Where do you want to go in the future?",
            "Do you like traveling?",
            "Describe a place you visited before.",
        ],
        "School": [
            "What subject do you like and why?",
            "Do you prefer studying alone or with friends?",
            "What do you usually do at school?",
        ],
        "Technology": [
            "What app do you use every day?",
            "Do you like using computers?",
            "How do you use your smartphone?",
        ],
    },
    "Intermediate": {
        "Daily Life": [
            "Describe a habit that helps you in your daily life.",
            "How do you usually spend your free time?",
            "What is something you want to improve in your daily routine?",
        ],
        "Food": [
            "Why do you think food culture is important?",
            "Describe a memorable meal you have had.",
            "Do you think eating at home is better than eating out?",
        ],
        "Travel": [
            "Describe a memorable trip you have taken.",
            "What are the benefits of traveling to other countries?",
            "Would you rather travel alone or with other people?",
        ],
        "School": [
            "What makes a class interesting for you?",
            "How can students learn English more effectively?",
            "Do you think homework is useful?",
        ],
        "Technology": [
            "How has technology changed the way people communicate?",
            "What are the advantages and disadvantages of smartphones?",
            "Do you think AI will change education?",
        ],
    },
    "Advanced": {
        "Daily Life": [
            "How can people maintain a healthy work-life balance?",
            "What social changes have affected people's daily lives recently?",
            "Why do some people find it difficult to manage their time effectively?",
        ],
        "Food": [
            "How does globalization influence local food culture?",
            "Should governments encourage healthier eating habits?",
            "What role does food play in cultural identity?",
        ],
        "Travel": [
            "How can tourism benefit and harm local communities?",
            "Should people travel more to understand different cultures?",
            "What responsibilities do travelers have when visiting other countries?",
        ],
        "School": [
            "What skills should schools focus on in the future?",
            "How should education change in response to AI?",
            "Is standardized testing a fair way to evaluate students?",
        ],
        "Technology": [
            "What ethical issues should people consider when using AI?",
            "How might technology change the future of work?",
            "Should social media platforms be more strictly regulated?",
        ],
    },
}


_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def get_fixed_question(level: str, topic: str) -> str:
    return random.choice(QUESTION_BANK[level][topic])


def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
        timeout=LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def generate_question_with_llm(level: str, topic: str) -> str:
    prompt = f"""
Create one English speaking practice question.

Requirements:
- Learner level: {level}
- Topic: {topic}
- Output only one question.
- The question must be natural, clear, and suitable for English learners.
- Do not include explanations, numbering, or quotation marks.
""".strip()
    question = call_ollama(prompt, temperature=0.4)
    return question.strip().splitlines()[0].strip()


def create_question(level, topic, use_llm):
    if not use_llm:
        return get_fixed_question(level, topic)
    try:
        return generate_question_with_llm(level, topic)
    except Exception as exc:
        fallback = get_fixed_question(level, topic)
        return f"{fallback}\n\n[LLM question generation failed: {exc}]"


def transcribe_audio(audio_path: str | None) -> str:
    if not audio_path:
        raise ValueError("音声を録音またはアップロードしてください。")

    model = get_whisper_model()
    segments, _info = model.transcribe(audio_path, language="en", vad_filter=True)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()

    if not transcript:
        raise ValueError("音声を文字起こしできませんでした。もう一度はっきり話して録音してください。")

    return transcript


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def evaluate_answer(question: str, transcript: str, level: str) -> dict[str, Any]:
    prompt = f"""
You are an English speaking tutor for Japanese learners.

Evaluate the learner's answer based only on the transcribed English text.
Do not evaluate pronunciation, intonation, or fluency from audio.

Question:
{question}

Learner level:
{level}

Learner's transcribed answer:
{transcript}

Return only valid JSON with this exact structure:
{{
  "overall_score": 0,
  "scores": {{
    "relevance": 0,
    "grammar": 0,
    "vocabulary": 0,
    "naturalness": 0
  }},
  "corrected_answer": "",
  "more_natural_answer": "",
  "feedback_ja": "",
  "model_answer": ""
}}

Scoring rules:
- overall_score must be an integer from 0 to 100.
- each score in scores must be an integer from 1 to 5.
- feedback_ja must be concise Japanese feedback.
- corrected_answer should preserve the learner's intended meaning.
- model_answer should match the learner level.
""".strip()
    raw = call_ollama(prompt, temperature=0.1)
    try:
        return extract_json(raw)
    except Exception as exc:
        return {
            "overall_score": "形式エラー",
            "scores": {
                "relevance": "-",
                "grammar": "-",
                "vocabulary": "-",
                "naturalness": "-",
            },
            "corrected_answer": "",
            "more_natural_answer": "",
            "feedback_ja": f"LLMの出力をJSONとして解釈できませんでした: {exc}",
            "model_answer": raw,
        }


def format_result(result: dict[str, Any]) -> str:
    scores = result.get("scores", {})
    return f"""
## Score

**Overall:** {result.get("overall_score", "-")} / 100

| Criterion | Score |
| --- | --- |
| Relevance | {scores.get("relevance", "-")} / 5 |
| Grammar | {scores.get("grammar", "-")} / 5 |
| Vocabulary | {scores.get("vocabulary", "-")} / 5 |
| Naturalness | {scores.get("naturalness", "-")} / 5 |

## Corrected Answer

{result.get("corrected_answer", "")}

## More Natural Expression

{result.get("more_natural_answer", "")}

## Feedback

{result.get("feedback_ja", "")}

## Model Answer

{result.get("model_answer", "")}
""".strip()


def get_server_port() -> int:
    if GRADIO_SERVER_PORT:
        return int(GRADIO_SERVER_PORT)

    for port in range(8960, 9060):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    raise RuntimeError("空いているGradio用ポートが見つかりませんでした。GRADIO_SERVER_PORTを指定してください。")


def run_evaluation(audio_path, question, level):
    if not question.strip():
        return "", "先に質問を生成してください。"

    try:
        transcript = transcribe_audio(audio_path)
        result = evaluate_answer(question, transcript, level)
        return transcript, format_result(result)
    except requests.RequestException as exc:
        return "", f"Ollamaへの接続に失敗しました: {exc}"
    except Exception as exc:
        return "", str(exc)


with gr.Blocks(title="English Speaking Feedback") as demo:
    gr.Markdown(
        """
# English Speaking Feedback

音声認識で得られた英文テキストを評価します。発音、イントネーション、流暢さの音響的評価は対象外です。
"""
    )

    with gr.Row():
        level = gr.Dropdown(
            choices=list(QUESTION_BANK.keys()),
            value="Beginner",
            label="Level",
        )
        topic = gr.Dropdown(
            choices=list(next(iter(QUESTION_BANK.values())).keys()),
            value="Daily Life",
            label="Topic",
        )
        use_llm_question = gr.Checkbox(
            value=False,
            label="Use LLM question generation",
        )

    question = gr.Textbox(label="Question", interactive=False, lines=3)
    new_question_button = gr.Button("New Question")

    audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Your Answer",
    )
    evaluate_button = gr.Button("Evaluate")

    transcript = gr.Textbox(label="Transcript", lines=4)
    feedback = gr.Markdown(label="Feedback")

    new_question_button.click(
        create_question,
        inputs=[level, topic, use_llm_question],
        outputs=question,
    )
    evaluate_button.click(
        run_evaluation,
        inputs=[audio, question, level],
        outputs=[transcript, feedback],
    )


if __name__ == "__main__":
    demo.launch(server_port=get_server_port())
