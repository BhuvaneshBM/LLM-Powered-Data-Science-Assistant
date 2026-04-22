from __future__ import annotations

import logging
import os
from uuid import uuid4

import gradio as gr
from dotenv import load_dotenv

import config
from models.agent import ensure_toolkit, reload_toolkit, run_automl_agent
from utils.dataset_manager import load_dataframe_from_upload

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def _extract_assistant_text(result: dict) -> str:
    responses: list[str] = []
    for msg in result.get("messages", []):
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", None)
        if msg_type == "ai" and content:
            if isinstance(content, list):
                parts: list[str] = []
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        text = str(chunk.get("text", "")).strip()
                        if text:
                            parts.append(text)
                if parts:
                    responses.append("\n".join(parts))
            else:
                text = str(content).strip()
                if text:
                    responses.append(text)
    if responses:
        return "\n\n".join(responses)
    return "I could not generate a response for that request."


def _ask_agent(message: str, thread_id: str) -> str:
    if not message.strip():
        return "Please enter a question."
    try:
        result = run_automl_agent(message, thread_id=thread_id)
        return _extract_assistant_text(result)
    except Exception as exc:
        logger.error("Error in Gradio interface: %s", exc)
        return f"Error: {exc}"


def _new_thread_id() -> str:
    return f"gradio-{uuid4()}"


def _smart_examples(columns: list[str]) -> list[str]:
    cols_lower = {c.lower(): c for c in columns}
    numeric_hints = ["price", "salary", "revenue", "age", "score", "value",
                     "amount", "cost", "sales", "income", "fare", "rate"]
    categorical_hints = ["type", "category", "class", "status", "gender",
                         "species", "label", "grade", "region", "country"]
    regression_target = next(
        (cols_lower[h] for h in numeric_hints if h in cols_lower), columns[0]
    )
    classification_target = next(
        (cols_lower[h] for h in categorical_hints if h in cols_lower), columns[-1]
    )
    return [
        "What is the structure of this dataset?",
        "Check missing values across all columns.",
        f"Get column stats for {regression_target}.",
        f"Evaluate baseline regression models for {regression_target}.",
        f"Run baseline classification models for {classification_target}.",
        "Detect and handle outliers in all numeric columns.",
        "Apply PCA with 2 components and visualize.",
    ]


# ============================================================================
# UPLOAD HANDLER
# ============================================================================

def handle_upload(file_obj):
    if file_obj is None:
        return "No file uploaded.", _new_thread_id()
    try:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        df, status = load_dataframe_from_upload(file_path)
        reload_toolkit(df)
        return status, _new_thread_id()
    except ValueError as exc:
        return f"Upload failed: {exc}", _new_thread_id()
    except Exception as exc:
        logger.error("Unexpected upload error: %s", exc)
        return f"Unexpected error: {exc}", _new_thread_id()


# ============================================================================
# GRADIO UI
# ============================================================================

def build_demo() -> gr.Blocks:
    default_examples = [
        "What is the structure of this dataset?",
        "Check missing values across all columns.",
        "Evaluate baseline regression models for PRICE.",
        "Run baseline classification models for BEDS.",
        "Detect outliers in PRICE using IQR.",
        "Apply PCA with 2 components and visualize.",
    ]

    with gr.Blocks(title="LLM-Powered Data Science Assistant") as demo:

        gr.Markdown("# LLM-Powered Data Science Assistant")
        gr.Markdown(
            "Upload **any CSV** or use the built-in NY Housing dataset. "
            "Ask questions in plain English — the assistant profiles data, "
            "runs preprocessing, evaluates models, and saves outputs."
        )

        with gr.Row():
            with gr.Column(scale=2):
                upload = gr.File(
                    label="Upload your own CSV dataset (optional)",
                    file_types=[".csv", ".tsv", ".txt"],
                    type="filepath",
                )
            with gr.Column(scale=3):
                upload_status = gr.Markdown(
                    value="Using default dataset: **NY House Dataset**. "
                          "Upload a CSV above to switch datasets."
                )

        gr.Markdown("---")

        chatbot = gr.Chatbot(label="Assistant", height=420)

        with gr.Row():
            user_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Evaluate baseline regression models for PRICE.",
                scale=5,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear chat")
            reset_btn = gr.Button("New session")

        gr.Markdown("**Example prompts** (click to fill):")
        with gr.Row():
            btn0 = gr.Button(default_examples[0], size="sm")
            btn1 = gr.Button(default_examples[1], size="sm")
            btn2 = gr.Button(default_examples[2], size="sm")
        with gr.Row():
            btn3 = gr.Button(default_examples[3], size="sm")
            btn4 = gr.Button(default_examples[4], size="sm")
            btn5 = gr.Button(default_examples[5], size="sm")

        thread_id = gr.State(_new_thread_id())

        # ── Event handlers ──────────────────────────────────────────────────

        def _chat_fn(message: str, history: list, current_thread_id: str):
            if not message.strip():
                return history, ""
            answer = _ask_agent(message, current_thread_id)
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history, ""

        upload.upload(fn=handle_upload, inputs=[upload], outputs=[upload_status, thread_id])

        send_btn.click(fn=_chat_fn, inputs=[user_input, chatbot, thread_id], outputs=[chatbot, user_input])
        user_input.submit(fn=_chat_fn, inputs=[user_input, chatbot, thread_id], outputs=[chatbot, user_input])

        btn0.click(fn=lambda: default_examples[0], outputs=[user_input])
        btn1.click(fn=lambda: default_examples[1], outputs=[user_input])
        btn2.click(fn=lambda: default_examples[2], outputs=[user_input])
        btn3.click(fn=lambda: default_examples[3], outputs=[user_input])
        btn4.click(fn=lambda: default_examples[4], outputs=[user_input])
        btn5.click(fn=lambda: default_examples[5], outputs=[user_input])

        clear_btn.click(fn=lambda: [], outputs=[chatbot])
        reset_btn.click(fn=lambda: _new_thread_id(), outputs=[thread_id])
        reset_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main() -> None:
    load_dotenv()
    config.ensure_output_directories()

    try:
        ensure_toolkit()
        logger.info("Default dataset loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Default dataset not found (%s). Upload a CSV to begin.", exc)

    demo = build_demo()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]),
    )


if __name__ == "__main__":
    main()
