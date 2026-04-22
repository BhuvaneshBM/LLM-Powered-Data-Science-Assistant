from __future__ import annotations

import logging
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
    """Extract assistant text robustly from LangGraph result messages."""
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
    """Call the AutoML agent and return a user-friendly response string."""
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
    """Generate context-aware example prompts based on column names."""
    cols_lower = {c.lower(): c for c in columns}

    # Try to find numeric columns that look like good targets
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
    """Process an uploaded CSV file and reload the agent toolkit.

    Returns:
        (status_markdown, updated_examples_list, new_thread_id)
    """
    if file_obj is None:
        return "No file uploaded.", gr.update(), _new_thread_id()

    try:
        df, status = load_dataframe_from_upload(file_obj.name)
        reload_toolkit(df)
        examples = _smart_examples(df.columns.tolist())
        thread_id = _new_thread_id()
        return status, gr.update(value=examples), thread_id
    except ValueError as exc:
        return f"Upload failed: {exc}", gr.update(), _new_thread_id()
    except Exception as exc:
        logger.error("Unexpected upload error: %s", exc)
        return f"Unexpected error: {exc}", gr.update(), _new_thread_id()


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

    with gr.Blocks(title="LLM-Powered Data Science Assistant", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# LLM-Powered Data Science Assistant")
        gr.Markdown(
            "Upload **any CSV** or use the built-in NY Housing dataset. "
            "Ask questions in plain English — the assistant profiles data, "
            "runs preprocessing, evaluates models, and saves outputs."
        )

        # ── Dataset panel ──────────────────────────────────────────────────
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

        # ── Chat panel ─────────────────────────────────────────────────────
        chatbot = gr.Chatbot(label="Assistant", height=420, type="messages")

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

        example_list = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[[e] for e in default_examples],
            label="Example prompts (click to use)",
            type="index",
        )

        thread_id = gr.State(_new_thread_id())

        # ── Event handlers ─────────────────────────────────────────────────

        def _chat_fn(message: str, history: list, current_thread_id: str):
            if not message.strip():
                return history, ""
            answer = _ask_agent(message, current_thread_id)
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history, ""

        def _on_example_click(evt: gr.SelectData, current_examples):
            """Fill textbox when user clicks an example."""
            idx = evt.index
            if isinstance(current_examples, list) and idx < len(current_examples):
                row = current_examples[idx]
                return row[0] if isinstance(row, list) else row
            return ""

        def _reset_session():
            return _new_thread_id()

        # Upload triggers toolkit reload and example refresh
        upload.upload(
            fn=handle_upload,
            inputs=[upload],
            outputs=[upload_status, example_list, thread_id],
        )

        send_btn.click(
            fn=_chat_fn,
            inputs=[user_input, chatbot, thread_id],
            outputs=[chatbot, user_input],
        )
        user_input.submit(
            fn=_chat_fn,
            inputs=[user_input, chatbot, thread_id],
            outputs=[chatbot, user_input],
        )
        example_list.click(
            fn=_on_example_click,
            inputs=[example_list],
            outputs=[user_input],
        )
        clear_btn.click(fn=lambda: [], outputs=[chatbot])
        reset_btn.click(fn=_reset_session, outputs=[thread_id])
        reset_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main() -> None:
    load_dotenv()
    config.ensure_output_directories()

    # Pre-load default dataset so the first message is fast
    try:
        ensure_toolkit()
        logger.info("Default dataset loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Default dataset not found (%s). Upload a CSV to begin.", exc)

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
