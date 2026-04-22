# Project Overview: LLM-Powered Data Science Assistant

## 1. What this project is
This is a local AI-powered data science assistant for the New York housing dataset.

It takes natural-language questions and turns them into data analysis actions through tools such as profiling, preprocessing, modeling, dimensionality reduction, and output saving.

In short:
- You ask a question about the dataset.
- The agent calls the right tool(s).
- The project returns a data-backed answer or saved artifact.

## 2. Current architecture
The project uses a LangGraph tool-calling workflow with LangChain and provider-based LLM routing.

Current flow:
1. Load the dataset from `DATASET_PATH`, then `data/NY-House-Dataset.csv`, then KaggleHub fallback.
2. Build the reusable `DataScienceToolkit` from the loaded dataframe.
3. Expose toolkit methods as LangGraph tools.
4. Route the model through either Groq or Gemini based on `LLM_PROVIDER`.
5. Run an assistant node plus tools node with memory.
6. Present the agent through CLI, notebook helpers, or Gradio UI.

## 3. Repository contents
Top-level items currently present:
- `.env`
- `.git/`
- `.gitignore`
- `.python-version`
- `.venv/`
- `config.py`
- `data/`
- `gradio_app.py`
- `main.ipynb`
- `main.py`
- `models/`
- `outputs/`
- `PROJECT_OVERVIEW.md`
- `pyproject.toml`
- `README.md`
- `utils/`
- `uv.lock`

## 4. Important project files

### `main.py`
CLI entrypoint for interactive terminal use.

Responsibilities:
- Loads environment variables.
- Ensures output directories exist.
- Initializes the dataset toolkit.
- Starts a simple input/output chat loop.

### `gradio_app.py`
Web UI entrypoint.

Responsibilities:
- Loads environment variables.
- Ensures output directories exist.
- Initializes the dataset toolkit.
- Builds a Gradio Blocks interface with chatbot, textbox, send button, clear button, reset button, and example prompts.
- Calls `run_automl_agent()` for each user message.

### `models/agent.py`
Agent orchestration layer.

Responsibilities:
- Creates tool wrappers around toolkit methods.
- Builds the LangGraph with assistant and tools nodes.
- Supports provider selection through `LLM_PROVIDER`.
- Supports Gemini via `ChatGoogleGenerativeAI`.
- Supports Groq via `ChatGroq`.
- Adds strict JSON sanitization to tool outputs so `NaN`/`Infinity` do not break provider payloads.
- Injects dataset context so the model knows the dataset is already loaded.

### `utils/dataset_manager.py`
Dataset resolution and loading logic.

Resolution order:
1. `DATASET_PATH`
2. `data/NY-House-Dataset.csv`
3. KaggleHub download fallback

If `DATASET_PATH` is invalid, the code now falls back instead of crashing.

### `utils/data_tools.py`
Reusable notebook-derived toolkit class.

Implemented capability groups:
- Profiling: dataset info, missing values, column stats, auto type detection
- Cleaning/transforms: missing value handling, categorical encoding, scaling
- Feature engineering: feature selection, polynomial feature generation
- Analysis: outlier detection, distribution analysis, dataset operations helper
- Modeling: baseline classification and regression cross-validation
- Dimensionality reduction: PCA, t-SNE, UMAP
- Export: save dataframe to CSV

### `utils/outputs_manager.py`
Output artifact pipeline.

Responsibilities:
- Saves plots to `outputs/plots/`
- Saves reports to `outputs/reports/`
- Saves logs/session summaries to `outputs/logs/`
- Uses timestamped filenames by default

### `config.py`
Central configuration for model behavior, tool behavior, logging, and outputs.

Important settings:
- `DEFAULT_PROVIDER`
- `DEFAULT_MODEL`
- `DEFAULT_GROQ_MODEL`
- `TEMPERATURE`
- `MAX_TOKENS`
- `TOOLS_CONFIG`
- `OUTPUT_DIR` and subdirectories
- `SYSTEM_PROMPT`

### `main.ipynb`
Notebook-based testing and exploration workflow.

Notebook summary:
- Cells 1 to 4 executed successfully.
- Current cell at lines 75 to 79 previously executed with errors.
- Notebook already contains `ROOT`, `df`, and `toolkit` variables in kernel state.

### `ai_driven_ml_and_datascience_assistant.ipynb`
Original prototype notebook retained for reference.

### `README.md`
User-facing setup/run guide.

## 5. Current runtime behavior
The project currently supports both providers:
- Groq
- Gemini

Provider selection is controlled by `LLM_PROVIDER`, so the active backend depends on the current environment configuration rather than a hardcoded app path.

Current behavior also includes:
- Dataset context priming so the model does not keep asking the user to upload a dataset
- Strict JSON tool serialization
- Output saving for plots and reports
- Dataset loading fallback when `DATASET_PATH` is invalid

## 6. What has been built so far
Completed work summary:

1. Modularized the notebook into Python modules.
2. Built a tool-calling LangGraph agent.
3. Added CLI and Gradio entrypoints.
4. Added provider switching for Groq and Gemini.
5. Added strict JSON sanitization for tool outputs.
6. Added dataset path fallback logic.
7. Added an outputs pipeline for plots, reports, and logs.
8. Updated notebook and docs to use the modular project structure.

## 7. How to run the project

### Terminal / CLI
```powershell
uv run python main.py
```

### Gradio UI
```powershell
uv run python gradio_app.py
```

### Notebook
Open `main.ipynb` and run the cells top to bottom.

## 8. Environment variables
Required depending on provider:
- `LLM_PROVIDER=groq` or `LLM_PROVIDER=gemini`
- `GROQ_API_KEY` if using Groq
- `GOOGLE_API_KEY` if using Gemini

Useful optional variables:
- `GROQ_MODEL`
- `GROQ_MAX_TOKENS`
- `GEMINI_MODEL`
- `DATASET_PATH`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_ENDPOINT`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_PROJECT`

## 9. Example prompts
- What is the shape of the dataset?
- Check missing values across all columns.
- Show dataset info.
- Get column stats for PRICE.
- Detect outliers in PRICE using IQR.
- Run baseline regression models for PRICE.
- Apply PCA with 2 components.
- Save the current dataframe to CSV.

## 10. Current status and known limitations
Status: Functional and runnable.

Known limitations:
- Tool-calling quality still depends on the chosen provider/model.
- Some prompts can trigger provider quota or function-call limits on free tiers.
- Some toolkit operations mutate the in-memory dataframe by design.
- There is no formal automated test suite yet.

## 11. Things to improve next
1. Add unit tests for toolkit methods and agent wrappers.
2. Add a small evaluation script for prompt and tool-call reliability.
3. Add a stable non-tool fallback mode for extremely limited model tiers.
4. Add a saved run history or session summary file under `outputs/logs/`.
5. Clean up the original prototype notebook further if you want a single source of truth.
