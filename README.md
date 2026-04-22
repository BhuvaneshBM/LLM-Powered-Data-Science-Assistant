# LLM-Powered Data Science Assistant

An agentic data science assistant built with LangGraph, LangChain tools, and Gradio.
It can profile a dataset, clean and transform data, run baseline ML models, perform dimensionality reduction, and save outputs.

The app supports two interaction modes:
- CLI chat via main.py
- Web UI via gradio_app.py with optional CSV upload and dataset hot-swap

## Features

- LangGraph tool-calling agent with memory checkpointing
- Provider routing with environment-based selection:
	- Gemini (ChatGoogleGenerativeAI)
	- Groq (ChatGroq)
- Dataset-aware system grounding (agent knows active dataset shape and columns)
- CSV/TSV/TXT upload in Gradio and runtime toolkit reload
- Data science tools for:
	- profiling and schema checks
	- missing values and outlier handling
	- encoding/scaling/feature engineering
	- baseline classification and regression
	- PCA/t-SNE/UMAP
	- exporting CSVs and plots

## Project layout

- main.py: CLI entrypoint
- gradio_app.py: Gradio interface and upload flow
- models/agent.py: LangGraph graph, provider selection, tool registry
- utils/data_tools.py: DataScienceToolkit implementation
- utils/dataset_manager.py: dataset resolution and upload parsing
- config.py: default provider/model settings, output paths, prompt/config values
- data/: optional local dataset storage
- outputs/: generated plots, reports, models, logs

## Requirements

- Python 3.12+
- One LLM provider credential:
	- GOOGLE_API_KEY for Gemini
	- GROQ_API_KEY for Groq

## Setup

Recommended (uv):

```powershell
uv venv --clear
uv sync
```

Alternative (pip):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Environment variables

Create a .env file in the project root.

Core settings:

```env
# Provider selection: gemini or groq
LLM_PROVIDER=gemini

# Gemini
GOOGLE_API_KEY=your_google_key
GEMINI_MODEL=gemini-2.5-flash-lite

# Groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
GROQ_MAX_TOKENS=1024

# Optional tracing
LANGCHAIN_API_KEY=optional_langsmith_key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AutoML-Hackathon

# Optional: force a local dataset path
# DATASET_PATH=C:/path/to/your/data.csv

# Optional Gradio port override
# GRADIO_SERVER_PORT=7860
```

## Dataset loading behavior

The default dataset is resolved in this order:

1. DATASET_PATH if set and valid
2. data/NY-House-Dataset.csv
3. KaggleHub fallback (new-york-housing-market)

In Gradio, uploading a CSV/TSV/TXT replaces the active toolkit at runtime using reload_toolkit, so new prompts operate on the uploaded file immediately.

## Run the app

CLI:

```powershell
uv run python main.py
```

Gradio UI:

```powershell
uv run python gradio_app.py
```


## Example prompts

- What is the structure of this dataset?
- Check missing values across all columns.
- Get column stats for PRICE.
- Evaluate baseline regression models for PRICE.
- Run baseline classification models for BEDS.
- Detect and handle outliers in all numeric columns.
- Apply PCA with 2 components and visualize.

## Outputs

Generated artifacts are saved under outputs/:

- outputs/plots
- outputs/reports
- outputs/models
- outputs/logs

Directories are auto-created at startup by config.ensure_output_directories().


## Notes

- .env is gitignored. Keep keys private.
- uv.lock is tracked for reproducible environments.
- main.ipynb can be used for iterative local experimentation.
