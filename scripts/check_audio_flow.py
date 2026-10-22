"""Opt-in real audio/LLM integration check through Gradio's processing pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gradio.state_holder import SessionState

from app import build_demo
from feedback import ServiceError, transcribe_audio


async def invoke(demo, state, name, values):
    index = next(i for i, item in demo.fns.items() if item.fn and item.fn.__name__ == name)
    response = await demo.process_api(index, values, state=state)
    frames = [response["data"]]
    while response["is_generating"]:
        response = await demo.process_api(index, [], state=state, iterator=response["iterator"])
        frames.append(response["data"])
    return frames


async def main():
    demo = build_demo()
    state = SessionState(demo)
    state_id = next(i for i, block in demo.blocks.items() if isinstance(block, gr.State))
    current = state[state_id]
    summary = {}
    with TestClient(gr.mount_gradio_app(FastAPI(), demo, path="/", max_file_size="20mb", run_history=False)) as client:
        summary["page_status"] = client.get("/").status_code
        summary["config_status"] = client.get("/config").status_code
        await invoke(demo, state, "example_action", [None])
        with (ROOT / "samples/weekend.wav").open("rb") as source:
            upload = client.post("/gradio_api/upload", files={"files": ("weekend.wav", source, "audio/wav")})
        summary["upload_status"] = upload.status_code
        upload.raise_for_status()
        payload = {"path": upload.json()[0], "orig_name": "weekend.wav", "meta": {"_type": "gradio.FileData"}}
        await invoke(demo, state, "audio_action", [None])
        started = time.monotonic()
        await invoke(demo, state, "transcription_action", [payload, None])
        summary["transcription_seconds"] = round(time.monotonic() - started, 2)
        summary["transcript"] = current.transcript
        assert current.transcript.strip()
        # Correct the recognizer's punctuation before evaluating, as a learner can.
        reviewed = "I visited a park with my friends last weekend. We played soccer and had lunch together."
        await invoke(demo, state, "text_action", [reviewed, None])
        started = time.monotonic()
        await invoke(demo, state, "evaluation_action", [reviewed, None])
        summary["evaluation_seconds"] = round(time.monotonic() - started, 2)
        assert current.outcome is not None
        summary["quality_checked"] = current.outcome.quality_checked
        summary["score"] = current.outcome.report.scores.total if current.outcome.quality_checked else None
        assert len(current.history) == 1
        question_id = current.question.id
        frames = await invoke(demo, state, "retry_action", [None])
        assert current.question.id == question_id and current.transcript == "" and len(current.history) == 1
        summary["retry_keeps_question_and_history"] = True
        retry_index = next(i for i, fn in demo.fns.items() if fn.fn and fn.fn.__name__ == "retry_action")
        outputs = demo.fns[retry_index].outputs
        audio_index = next(i for i, component in enumerate(outputs) if isinstance(component, gr.Audio))
        text_index = next(i for i, component in enumerate(outputs) if isinstance(component, gr.Textbox) and component.label.startswith("文字起こし"))
        assert frames[0][audio_index].get("value") is None
        assert frames[0][text_index]["value"] == ""
        summary["retry_clears_rendered_audio_and_text"] = True
        await invoke(demo, state, "settings_action", [None])
        assert current.question is None and current.transcript == ""
        summary["settings_reset"] = True
        oversized = client.post("/gradio_api/upload", files={"files": ("too-large.wav", b"x" * (21 * 1024 * 1024), "audio/wav")})
        summary["oversized_upload_status"] = oversized.status_code
        assert oversized.status_code == 413
    with tempfile.NamedTemporaryFile(suffix=".wav") as silence:
        with wave.open(silence.name, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(b"\x00\x00" * 16000 * 3)
        try:
            transcribe_audio(silence.name)
        except ServiceError as exc:
            assert "文字起こしできません" in str(exc)
            summary["silence_rejected"] = True
        else:
            raise AssertionError("Silence was accepted")
    path = Path("/tmp/english-speaking-audio-flow-results.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
