# LLM-Powered Data Science Assistant

This project provides a local AI assistant that can inspect the New York housing dataset and execute a broad set of data-science tools, including preprocessing, statistics, outlier handling, dimensionality reduction, and baseline ML model evaluation.

## What is included

- `main.py`: runnable CLI entrypoint
- `models/agent.py`: LangGraph + Gemini tool-calling agent
- `utils/dataset_manager.py`: dataset path resolution and loading
- `utils/data_tools.py`: full notebook-derived data-science toolkit
- `main.ipynb`: notebook workflow for local testing
- `data/`: local dataset location (`NY-House-Dataset.csv`)

## Toolkit capabilities

- Data profiling: schema, dtypes, per-column stats, missing values, data type detection
- Data cleaning/transformation: missing value handling, categorical encoding, scaling
- Feature engineering: variance/correlation feature selection, polynomial features
- Analysis: outlier detection, distribution analysis, dataset operations helper
- Modeling: baseline classification and regression cross-validation
- Dimensionality reduction: PCA, t-SNE, UMAP
- Export: save current transformed dataframe to CSV

## Prerequisites

- Python 3.12+
- A Gemini API key

## Setup

1. Create and activate a virtual environment (if not already active).
2. Install dependencies:

```powershell
pip install -e .
```

3. Configure environment variables (recommended via `.env`):

```env
GOOGLE_API_KEY=your_key_here
LANGCHAIN_API_KEY=optional_langsmith_key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AutoML-Hackathon
```

4. Dataset handling:
- If `data/NY-House-Dataset.csv` exists, it is used directly.
- Otherwise, the app downloads the dataset using KaggleHub.

## Run from terminal

```powershell
python main.py
```

## Run Gradio UI

```powershell
python gradio_app.py
```

Or with uv:

```powershell
uv run python gradio_app.py
```

The app launches at `http://127.0.0.1:7860` by default.

Example prompts:
- `What columns are present in this dataset?`
- `Check missing values for all columns.`
- `Evaluate regression baselines for PRICE.`
- `Run classification baselines for BEDS.`
- `Apply minmax scaling on numeric features.`
- `Do PCA with 2 components.`

## Run in notebook

Open `main.ipynb` and run cells top-to-bottom.

- Cell 1: environment + path setup
- Cell 2: dataset load + toolkit init
- Cell 3: quick dataset metadata
- Cell 4: helper function to query the assistant
- Cell 5: sample assistant questions

## Troubleshooting

- `GOOGLE_API_KEY is required`: set your API key in `.env` or shell environment.
- Dataset download issues: place `NY-House-Dataset.csv` manually in `data/`.
