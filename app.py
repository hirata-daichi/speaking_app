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
        f"{OLLAMA_BASE_URL}/api/generate",   # http://localhost:11434/api/generate このエンドポイント（自分のPC上で起動しているOllama）にリクエストを送信
        json={
            "model": OLLAMA_MODEL,  # 使用するモデルを指定（例: qwen2.5:7b-instruct）
            "prompt": prompt,   # 生成したいテキストの内容を指定
            "stream": False,   # ストリーミングは生成結果をリアルタイムで受け取る場合に使用
            "options": {
                "temperature": temperature,   # 生成の多様性を制御するパラメータ。0に近いほど決定的な出力、1に近いほどランダムな出力
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
  "key_corrections": [
    {{
      "original": "",
      "corrected": "",
      "reason_ja": ""
    }}
  ],
  "feedback_ja": {{
    "good_points": "",
    "grammar_points": "",
    "vocabulary_points": "",
    "next_step": ""
  }},
  "model_answer": ""
}}

Scoring rules:
- each score in scores must be an integer from 1 to 5.
- overall_score must be the sum of relevance, grammar, vocabulary, and naturalness, so it must be an integer from 4 to 20.
- Use this scale for each score:
  - 5 = very good, no major issue.
  - 4 = good, only minor issues.
  - 3 = understandable, but with several issues.
  - 2 = difficult to understand, with many issues.
  - 1 = mostly unclear or does not answer the question.
- relevance means how well the answer responds to the question.
- grammar means grammatical accuracy.
- vocabulary means appropriate word choice and phrase use.
- naturalness means how natural the English sounds in context.

Output rules:
- corrected_answer must preserve the learner's original meaning.
- corrected_answer should fix grammar, word choice, and unnatural expressions.
- Do not add new ideas to corrected_answer.
- key_corrections should include 1 to 3 important corrections.
- Each key_corrections item must compare the learner's original wording with the corrected wording.
- key_corrections.reason_ja must explain the correction in Japanese.
- If there are no important corrections, return an empty array.
- feedback_ja must be written in Japanese.
- feedback_ja.good_points should mention what the learner did well.
- feedback_ja.grammar_points should explain one or two grammar issues with examples.
- feedback_ja.vocabulary_points should suggest better words or phrases if useful.
- feedback_ja.next_step should give one concrete improvement goal for the next answer.
- model_answer must be a different sample answer to the same question.
- model_answer should be appropriate for the learner level.
- model_answer should be 2 to 4 sentences.
- model_answer must be written in English.
- Do not make model_answer too advanced for the learner level.
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
            "key_corrections": [],
            "feedback_ja": {
                "good_points": "",
                "grammar_points": f"LLMの出力をJSONとして解釈できませんでした: {exc}",
                "vocabulary_points": "",
                "next_step": "もう一度評価を実行してください。",
            },
            "model_answer": raw,
        }


def format_key_corrections(corrections: Any) -> str:
    if not isinstance(corrections, list) or not corrections:
        return "大きな修正点はありません。"

    def table_cell(value: Any) -> str:
        return str(value or "").replace("\n", " ").replace("|", "\\|").strip()

    rows = [
        "| Original | Corrected | Reason |",
        "| --- | --- | --- |",
    ]
    for item in corrections:
        if not isinstance(item, dict):
            continue
        original = table_cell(item.get("original", ""))
        corrected = table_cell(item.get("corrected", ""))
        reason = table_cell(item.get("reason_ja", ""))
        rows.append(f"| {original} | {corrected} | {reason} |")

    if len(rows) == 2:
        return "大きな修正点はありません。"
    return "\n".join(rows)


def format_feedback(feedback: Any) -> str:
    if isinstance(feedback, dict):
        return f"""
**良い点:** {feedback.get("good_points", "")}

**文法:** {feedback.get("grammar_points", "")}

**語彙・表現:** {feedback.get("vocabulary_points", "")}

**次に意識すること:** {feedback.get("next_step", "")}
""".strip()

    return str(feedback or "")


def parse_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None

    if 1 <= score <= 5:
        return score
    return None


def calculate_overall_score(scores: Any) -> int | str:
    if not isinstance(scores, dict):
        return "-"

    score_values = [
        parse_score(scores.get("relevance")),
        parse_score(scores.get("grammar")),
        parse_score(scores.get("vocabulary")),
        parse_score(scores.get("naturalness")),
    ]
    if any(score is None for score in score_values):
        return "-"

    return sum(score for score in score_values if score is not None)


def format_result(result: dict[str, Any]) -> str:
    scores = result.get("scores", {})
    overall_score = calculate_overall_score(scores)
    key_corrections = format_key_corrections(result.get("key_corrections", []))
    feedback = format_feedback(result.get("feedback_ja", ""))
    return f"""
## Score

**Overall:** {overall_score} / 20

| Criterion | Score |
| --- | --- |
| Relevance | {scores.get("relevance", "-")} / 5 |
| Grammar | {scores.get("grammar", "-")} / 5 |
| Vocabulary | {scores.get("vocabulary", "-")} / 5 |
| Naturalness | {scores.get("naturalness", "-")} / 5 |

## Corrected Answer

{result.get("corrected_answer", "")}

## Key Corrections

{key_corrections}

## Feedback

{feedback}

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
    demo.launch(server_port=get_server_port(), show_api=False)
