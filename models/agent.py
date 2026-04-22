from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from utils.data_tools import DataScienceToolkit
from utils.dataset_manager import load_dataset

logger = logging.getLogger(__name__)

_toolkit: Optional[DataScienceToolkit] = None
_dataset_context_cache: Optional[str] = None

# Module-level singleton graph + saver (rebuilt only when dataset changes)
_graph = None
_graph_saver: Optional[MemorySaver] = None


# ============================================================================
# TOOLKIT MANAGEMENT
# ============================================================================

def ensure_toolkit() -> DataScienceToolkit:
    """Return the current toolkit, loading the default dataset if needed."""
    global _toolkit
    if _toolkit is None:
        _toolkit = DataScienceToolkit(load_dataset())
    return _toolkit


def reload_toolkit(df: pd.DataFrame) -> str:
    """Replace the active toolkit with a new dataframe.

    Called when the user uploads a CSV. Also resets the graph so the new
    dataset context is injected into the next conversation.

    Returns a status string describing the loaded dataset.
    """
    global _toolkit, _dataset_context_cache, _graph, _graph_saver

    _toolkit = DataScienceToolkit(df)
    _dataset_context_cache = None  # force re-computation of context message
    _graph = None                  # force graph rebuild with fresh MemorySaver
    _graph_saver = None

    cols = df.columns.tolist()
    preview = ", ".join(str(c) for c in cols[:12])
    if len(cols) > 12:
        preview += f" (and {len(cols) - 12} more)"
    logger.info("Toolkit reloaded — shape %s", df.shape)
    return f"Dataset ready: {df.shape[0]:,} rows x {df.shape[1]} columns. Columns: {preview}."


def _get_dataset_context() -> str:
    """Return a compact, cached dataset context string for model grounding."""
    global _dataset_context_cache
    if _dataset_context_cache is not None:
        return _dataset_context_cache

    toolkit = ensure_toolkit()
    df = toolkit.df
    shape = list(df.shape)
    columns = [str(c) for c in df.columns.tolist()]
    preview_columns = columns[:12]
    more = "" if len(columns) <= 12 else f" (and {len(columns) - 12} more columns)"

    _dataset_context_cache = (
        "The dataset is already loaded and available via tools. "
        f"Shape: {shape[0]} rows x {shape[1]} columns. "
        f"Columns: {', '.join(preview_columns)}{more}. "
        "Do not ask the user to upload/provide a dataset unless a tool reports a loading error."
    )
    return _dataset_context_cache


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def _to_strict_json(value: Any) -> str:
    """Serialize tool output to strict JSON (no NaN/Infinity values)."""

    def _sanitize(obj: Any) -> Any:
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                obj = obj.item()
            except Exception:
                pass

        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if obj is None or isinstance(obj, (str, int, bool)):
            return obj

        try:
            if obj != obj:
                return None
        except Exception:
            pass

        return str(obj)

    return json.dumps(_sanitize(value), indent=2, ensure_ascii=True, allow_nan=False)


# ============================================================================
# TOOLS
# ============================================================================

@tool
def check_missing_values(data: str = "all") -> str:
    """Check missing values for one column or all columns."""
    result = ensure_toolkit().check_missing_values(data=data)
    return _to_strict_json(result)


@tool
def get_dataset_info() -> str:
    """Get dataset shape, columns, dtypes, and summary description."""
    result = ensure_toolkit().get_dataset_info()
    return _to_strict_json(result)


@tool
def get_column_stats(column_name: str) -> str:
    """Get stats for a specific column."""
    result = ensure_toolkit().get_column_stats(column_name=column_name)
    return _to_strict_json(result)


@tool
def auto_detect_data_types() -> str:
    """Categorize columns as numeric/categorical/datetime/text/binary/other."""
    result = ensure_toolkit().auto_detect_data_types()
    return _to_strict_json(result)


@tool
def detect_and_handle_outliers(column_name: str, method: str = "zscore", threshold: float = 3.0) -> str:
    """Detect outliers in a numeric column using zscore or iqr."""
    result = ensure_toolkit().detect_and_handle_outliers(
        column_name=column_name, method=method, threshold=threshold
    )
    return _to_strict_json(result)


@tool
def analyze_column_distribution(column_name: str, plot: bool = False) -> str:
    """Analyze distribution stats and optionally plot."""
    result = ensure_toolkit().analyze_column_distribution(column_name=column_name, plot=plot)
    return _to_strict_json(result)


@tool
def train_and_evaluate_classification_models(target_column: str, cv_folds: int = 5) -> str:
    """Run baseline classification models and return CV accuracy."""
    result = ensure_toolkit().train_and_evaluate_classification_models(
        target_column=target_column, cv_folds=cv_folds
    )
    return _to_strict_json(result)


@tool
def train_and_evaluate_regression_models(target_column: str, cv_folds: int = 5) -> str:
    """Run baseline regression models and return CV MSE."""
    result = ensure_toolkit().train_and_evaluate_regression_models(
        target_column=target_column, cv_folds=cv_folds
    )
    return _to_strict_json(result)


@tool
def handle_missing_values(strategy: str = "median", columns_csv: str = "") -> str:
    """Handle missing values. columns_csv accepts comma-separated columns."""
    columns = [c.strip() for c in columns_csv.split(",") if c.strip()] or None
    result = ensure_toolkit().handle_missing_values(strategy=strategy, columns=columns)
    return _to_strict_json(result)


@tool
def encode_categorical_variables(encoding_type: str = "onehot", columns_csv: str = "") -> str:
    """Encode categorical variables using onehot or label. columns_csv accepts comma-separated columns."""
    columns = [c.strip() for c in columns_csv.split(",") if c.strip()] or None
    result = ensure_toolkit().encode_categorical_variables(
        encoding_type=encoding_type, columns=columns
    )
    return _to_strict_json(result)


@tool
def scale_features(scaling_method: str = "standard", columns_csv: str = "") -> str:
    """Scale numeric features using standard or minmax."""
    columns = [c.strip() for c in columns_csv.split(",") if c.strip()] or None
    result = ensure_toolkit().scale_features(scaling_method=scaling_method, columns=columns)
    return _to_strict_json(result)


@tool
def feature_selection(method: str = "variance", threshold: float = 0.0) -> str:
    """Run variance or correlation based feature selection."""
    result = ensure_toolkit().feature_selection(method=method, threshold=threshold)
    return _to_strict_json(result)


@tool
def create_polynomial_features(degree: int = 2, interaction_only: bool = False, columns_csv: str = "") -> str:
    """Create polynomial features. columns_csv accepts comma-separated columns."""
    columns = [c.strip() for c in columns_csv.split(",") if c.strip()] or None
    result = ensure_toolkit().create_polynomial_features(
        degree=degree, interaction_only=interaction_only, columns=columns
    )
    return _to_strict_json(result)


@tool
def operations_on_dataset(
    task: str,
    column_1: str = "",
    column_2: str = "",
    operation: str = "",
    group_by: str = "",
    filter_column: str = "",
    filter_value: str = "",
) -> str:
    """General utility for calculate/filter/describe/group/visualize operations."""
    result = ensure_toolkit().operations_on_dataset(
        task=task,
        column_1=column_1 or None,
        column_2=column_2 or None,
        operation=operation or None,
        group_by=group_by or None,
        filter_column=filter_column or None,
        filter_value=filter_value or None,
    )
    return _to_strict_json(result)


@tool
def dimensionality_reduction(method: str = "pca", n_components: int = 2, visualize: bool = True) -> str:
    """Run PCA/TSNE/UMAP dimensionality reduction."""
    result = ensure_toolkit().dimensionality_reduction(
        method=method, n_components=n_components, visualize=visualize
    )
    return _to_strict_json(result)


@tool
def save_dataframe_to_csv(file_name: str = "output.csv") -> str:
    """Save current working dataframe to csv."""
    result = ensure_toolkit().save_dataframe_to_csv(file_name=file_name)
    return _to_strict_json(result)


@tool
def save_plot(plot_name: str = "plot") -> str:
    """Save the current matplotlib plot to outputs/plots directory."""
    result = ensure_toolkit().save_plot(plot_name=plot_name, close_figure=True)
    return _to_strict_json(result)


def _tool_registry() -> List[Any]:
    return [
        check_missing_values,
        get_dataset_info,
        get_column_stats,
        auto_detect_data_types,
        detect_and_handle_outliers,
        analyze_column_distribution,
        train_and_evaluate_classification_models,
        train_and_evaluate_regression_models,
        handle_missing_values,
        encode_categorical_variables,
        scale_features,
        feature_selection,
        create_polynomial_features,
        operations_on_dataset,
        dimensionality_reduction,
        save_dataframe_to_csv,
        save_plot,
    ]


# ============================================================================
# PROVIDER / LLM
# ============================================================================

def _resolve_provider() -> str:
    """Resolve active provider from env/config with safe fallbacks."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in {"groq", "gemini"}:
        return provider

    default_provider = str(getattr(config, "DEFAULT_PROVIDER", "gemini")).strip().lower()
    if default_provider in {"groq", "gemini"}:
        return default_provider

    return "gemini"


# ============================================================================
# GRAPH (singleton)
# ============================================================================

def _build_graph_internal(model_name: str = None):
    """Build the LangGraph agent. Called once per process (or after dataset reload)."""
    tools = _tool_registry()
    provider = _resolve_provider()
    groq_llm: Optional[ChatGroq] = None

    def _make_gemini_llm_with_tools(selected_model: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_TOKENS,
        )
        return llm.bind_tools(tools)

    def _make_groq_llm(selected_model: str) -> ChatGroq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        groq_max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "4096"))
        return ChatGroq(
            model=selected_model,
            api_key=api_key,
            temperature=config.TEMPERATURE,
            max_tokens=min(config.MAX_TOKENS, groq_max_tokens),
        )

    if provider == "groq":
        selected_model = model_name or os.getenv("GROQ_MODEL", config.DEFAULT_GROQ_MODEL)
        groq_llm = _make_groq_llm(selected_model)
        llm_with_tools = groq_llm.bind_tools(tools)
    else:
        selected_model = model_name or os.getenv("GEMINI_MODEL", config.DEFAULT_MODEL)
        llm_with_tools = _make_gemini_llm_with_tools(selected_model)

    logger.info("Using provider '%s' with model '%s'.", provider, selected_model)

    sys_msg = SystemMessage(content=config.SYSTEM_PROMPT)

    def assistant(state: MessagesState):
        nonlocal llm_with_tools
        dataset_msg = SystemMessage(content=_get_dataset_context())
        messages_to_model = [sys_msg, dataset_msg] + state["messages"]
        try:
            response = llm_with_tools.invoke(messages_to_model)
            return {"messages": [response]}
        except Exception as exc:
            if provider == "groq":
                error_text = str(exc).lower()
                if "tool_use_failed" in error_text or "failed to call a function" in error_text:
                    logger.warning("Groq tool call failed; falling back to plain response.")
                    response = groq_llm.invoke(messages_to_model)
                    return {"messages": [response]}
                raise

            if provider != "gemini":
                raise

            fallback_model = config.DEFAULT_MODEL
            error_text = str(exc).lower()
            is_model_not_found = "404" in error_text and "model" in error_text

            if selected_model != fallback_model and is_model_not_found:
                logger.warning(
                    "Model '%s' unavailable; retrying with fallback '%s'.",
                    selected_model,
                    fallback_model,
                )
                llm_with_tools = _make_gemini_llm_with_tools(fallback_model)
                response = llm_with_tools.invoke([sys_msg] + state["messages"])
                return {"messages": [response]}

            raise

    saver = MemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=saver), saver


def _get_graph():
    """Return the singleton graph, building it if needed."""
    global _graph, _graph_saver
    if _graph is None:
        _graph, _graph_saver = _build_graph_internal()
    return _graph


# ============================================================================
# PUBLIC API
# ============================================================================

def run_automl_agent(input_message: str, thread_id: str = "1") -> Dict[str, Any]:
    graph = _get_graph()
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    messages = [HumanMessage(content=input_message)]
    response = graph.invoke({"messages": messages}, cfg)
    return response

def build_graph(model_name: str = None):
    """Build and return a fresh LangGraph agent (no singleton caching).

    Use this in notebooks or scripts where you want an isolated graph instance.
    For production use (CLI / Gradio), prefer run_automl_agent() which reuses
    the singleton graph via _get_graph().
    """
    graph, _ = _build_graph_internal(model_name=model_name)
    return graph
