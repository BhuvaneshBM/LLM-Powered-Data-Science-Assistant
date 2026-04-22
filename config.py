"""Configuration file for LLM-Powered Data Science Assistant.

Defines default model selection, tool behaviors, and output settings.
"""

from pathlib import Path
from typing import Any, Dict

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_PROVIDER = "groq"
"""Default LLM provider: groq or gemini."""

DEFAULT_MODEL = "gemini-1.5-flash"
"""Default Gemini model to use for the agent."""

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
"""Default Groq model to use for the agent."""

TEMPERATURE = 0.7
"""Temperature for model generation (0-1, higher = more creative)."""

MAX_TOKENS = 4096
"""Maximum tokens in model responses."""

# ============================================================================
# TOOL BEHAVIOR CONFIGURATION
# ============================================================================

TOOLS_CONFIG: Dict[str, Dict[str, Any]] = {
    "data_profiling": {
        "enabled": True,
        "auto_detect_types": True,
        "missing_value_threshold": 0.5,  # Flag if > 50% missing
    },
    "preprocessing": {
        "enabled": True,
        "handle_missing": "mean",  # mean, median, drop, forward_fill
        "encoding_method": "label",  # label or onehot
        "scaling_method": "standard",  # standard or minmax
    },
    "feature_engineering": {
        "enabled": True,
        "polynomial_degree": 2,
        "variance_threshold": 0.01,
    },
    "modeling": {
        "enabled": True,
        "cv_folds": 5,
        "test_size": 0.2,
        "random_state": 42,
    },
    "dimensionality_reduction": {
        "enabled": True,
        "methods": ["pca", "tsne", "umap"],
        "n_components": 2,
    },
    "outlier_detection": {
        "enabled": True,
        "method": "iqr",  # iqr or zscore
        "iqr_multiplier": 1.5,
        "zscore_threshold": 3,
    },
}
"""Configuration for each tool's behavior and parameters."""

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output directory paths (relative to project root)
OUTPUT_DIR = Path("outputs")
"""Root output directory."""

PLOTS_DIR = OUTPUT_DIR / "plots"
"""Directory for saved visualizations."""

REPORTS_DIR = OUTPUT_DIR / "reports"
"""Directory for saved data science reports."""

MODELS_DIR = OUTPUT_DIR / "models"
"""Directory for saved trained models."""

LOGS_DIR = OUTPUT_DIR / "logs"
"""Directory for operation logs and summaries."""

# Output format settings
OUTPUT_FORMATS = {
    "plots": {
        "format": "png",  # png, jpg, pdf, svg
        "dpi": 300,
        "transparent": False,
        "bbox_inches": "tight",
    },
    "reports": {
        "format": "csv",  # csv, excel, json, html
    },
}
"""Settings for how outputs are formatted and saved."""

AUTO_SAVE_PLOTS = True
"""Whether to automatically save plots to plots/ directory."""

AUTO_SAVE_REPORTS = True
"""Whether to automatically save data summaries to reports/ directory."""

TIMESTAMP_OUTPUTS = True
"""Whether to include timestamps in output filenames."""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
"""Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""Format for log messages."""

ENABLE_FILE_LOGGING = True
"""Whether to save logs to file in outputs/logs/."""

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

MEMORY_TYPE = "dict"  # dict or sql
"""Type of memory to use for conversation persistence."""

SYSTEM_PROMPT = """You are an expert AI/ML data science assistant specializing in exploratory data analysis, 
feature engineering, and model building. You can:

1. Profile and understand datasets thoroughly
2. Detect and handle data quality issues (missing values, outliers)
3. Engineer meaningful features for improved model performance
4. Train and evaluate multiple machine learning models
5. Perform dimensionality reduction for visualization
6. Generate insightful reports and visualizations

When working with data:
- Always profile the dataset first to understand its structure
- Consider domain knowledge in your analysis
- Provide multiple approaches when appropriate
- Explain your reasoning and recommendations clearly
- Save important findings and visualizations to disk

Always aim for actionable insights and reproducible results."""

TOOL_TIMEOUT = 300
"""Timeout in seconds for tool execution."""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_output_directories() -> None:
    """Create all output directories if they don't exist."""
    for directory in [PLOTS_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_tool_config(tool_name: str) -> Dict[str, Any]:
    """Get configuration for a specific tool.

    Args:
        tool_name: Name of the tool (e.g., 'preprocessing', 'modeling')

    Returns:
        Dictionary of tool configuration, or empty dict if not found.
    """
    return TOOLS_CONFIG.get(tool_name, {})


def get_output_path(file_type: str, file_name: str) -> Path:
    """Get the full output path for a file.

    Args:
        file_type: Type of file ('plot', 'report', 'model', 'log')
        file_name: Name of the file

    Returns:
        Full Path object for the output file.
    """
    dir_map = {
        "plot": PLOTS_DIR,
        "report": REPORTS_DIR,
        "model": MODELS_DIR,
        "log": LOGS_DIR,
    }
    target_dir = dir_map.get(file_type, OUTPUT_DIR)
    return target_dir / file_name
