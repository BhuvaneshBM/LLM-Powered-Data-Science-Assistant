from __future__ import annotations

import logging

from dotenv import load_dotenv

import config
from models.agent import ensure_toolkit, run_automl_agent


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def chat_interface() -> None:
    """Command-line chat interface for interacting with the AutoML assistant."""
    thread_id = "local-chat"

    print("Welcome to the AutoML Assistant")
    print("You can ask questions about the NY House dataset.")
    print("Type 'exit' to end the conversation.")
    print("\nExample questions:")
    print("1. What is the overall structure of the dataset?")
    print("2. Are there any missing values?")
    print("3. Evaluate regression models for PRICE.")

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() == "exit":
            print("\nThank you for using the AutoML Assistant. Goodbye!")
            break
        if not user_input:
            print("Please enter a valid question.")
            continue

        try:
            result = run_automl_agent(user_input, thread_id)
            print("\nAssistant:", end=" ")
            for msg in result["messages"]:
                content = getattr(msg, "content", None)
                if content:
                    print(content)
        except Exception as exc:
            logger.error("Error in chat interface: %s", exc)
            print(f"\nAn error occurred: {exc}")


def main() -> None:
    load_dotenv()
    config.ensure_output_directories()
    ensure_toolkit()
    chat_interface()


if __name__ == "__main__":
    main()
