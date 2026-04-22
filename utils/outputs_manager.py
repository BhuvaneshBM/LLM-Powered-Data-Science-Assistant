"""Output management utilities for saving plots and reports.

Provides functionality for saving visualizations, data summaries, and reports
to organized output directories with automatic naming and timestamp support.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

import config

logger = logging.getLogger(__name__)


class OutputsManager:
    """Manager for saving plots, reports, and model artifacts."""

    def __init__(self, use_timestamps: bool = None):
        """Initialize the OutputsManager.

        Args:
            use_timestamps: Whether to add timestamps to filenames.
                           If None, uses config.TIMESTAMP_OUTPUTS.
        """
        self.use_timestamps = (
            use_timestamps if use_timestamps is not None else config.TIMESTAMP_OUTPUTS
        )
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.ensure_output_directories()

    def _get_timestamp(self) -> str:
        """Get a timestamp string for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _build_filename(self, base_name: str, extension: str, use_ts: bool = True) -> str:
        """Build a filename with optional timestamp.

        Args:
            base_name: Base name without extension
            extension: File extension (without dot)
            use_ts: Whether to add timestamp prefix

        Returns:
            Formatted filename with extension.
        """
        if use_ts and self.use_timestamps:
            return f"{self._get_timestamp()}_{base_name}.{extension}"
        return f"{base_name}.{extension}"

    def save_plot(
        self,
        figure: plt.Figure = None,
        plot_name: str = "plot",
        close_figure: bool = True,
        **kwargs,
    ) -> Path:
        """Save a matplotlib figure to the plots directory.

        Args:
            figure: matplotlib Figure object. If None, uses current figure.
            plot_name: Name for the saved plot (without extension).
            close_figure: Whether to close the figure after saving.
            **kwargs: Additional arguments for savefig (e.g., dpi, format).

        Returns:
            Path to the saved plot file.

        Example:
            >>> import matplotlib.pyplot as plt
            >>> plt.scatter([1, 2, 3], [1, 2, 3])
            >>> manager.save_plot(plot_name="scatter_plot")
        """
        if figure is None:
            figure = plt.gcf()

        # Merge kwargs with config defaults
        save_kwargs = {
            "dpi": config.OUTPUT_FORMATS["plots"]["dpi"],
            "transparent": config.OUTPUT_FORMATS["plots"]["transparent"],
            "bbox_inches": config.OUTPUT_FORMATS["plots"]["bbox_inches"],
        }
        save_kwargs.update(kwargs)

        format_ext = config.OUTPUT_FORMATS["plots"]["format"]
        filename = self._build_filename(plot_name, format_ext)
        filepath = config.PLOTS_DIR / filename

        try:
            figure.savefig(filepath, **save_kwargs)
            logger.info(f"Plot saved: {filepath}")

            if close_figure:
                plt.close(figure)

            return filepath
        except Exception as e:
            logger.error(f"Failed to save plot {plot_name}: {e}")
            raise

    def save_dataframe_report(
        self,
        df: pd.DataFrame,
        report_name: str = "report",
        format_type: str = None,
    ) -> Path:
        """Save a DataFrame as a report file.

        Args:
            df: pandas DataFrame to save.
            report_name: Name for the report (without extension).
            format_type: Format for saving ('csv', 'excel', 'json', 'html').
                        If None, uses config default.

        Returns:
            Path to the saved report file.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> manager.save_dataframe_report(df, "my_data")
        """
        if format_type is None:
            format_type = config.OUTPUT_FORMATS["reports"]["format"]

        filename = self._build_filename(report_name, format_type)
        filepath = config.REPORTS_DIR / filename

        try:
            if format_type == "csv":
                df.to_csv(filepath, index=False)
            elif format_type == "excel":
                df.to_excel(filepath, index=False)
            elif format_type == "json":
                df.to_json(filepath, orient="records", indent=2)
            elif format_type == "html":
                df.to_html(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"Report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save report {report_name}: {e}")
            raise

    def save_text_report(
        self,
        content: str,
        report_name: str = "report",
    ) -> Path:
        """Save text content as a report file.

        Args:
            content: Text content to save.
            report_name: Name for the report (without extension).

        Returns:
            Path to the saved report file.

        Example:
            >>> content = "Analysis Summary\\n\\nKey Findings:\\n- Finding 1\\n- Finding 2"
            >>> manager.save_text_report(content, "analysis_summary")
        """
        filename = self._build_filename(report_name, "txt")
        filepath = config.REPORTS_DIR / filename

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Text report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save text report {report_name}: {e}")
            raise

    def save_json_report(
        self,
        data: Dict[str, Any],
        report_name: str = "report",
    ) -> Path:
        """Save dictionary data as a JSON report.

        Args:
            data: Dictionary to save as JSON.
            report_name: Name for the report (without extension).

        Returns:
            Path to the saved report file.

        Example:
            >>> data = {"accuracy": 0.95, "precision": 0.92}
            >>> manager.save_json_report(data, "model_metrics")
        """
        import json

        filename = self._build_filename(report_name, "json")
        filepath = config.REPORTS_DIR / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save JSON report {report_name}: {e}")
            raise

    def save_session_summary(self, summary: str) -> Path:
        """Save a session summary/log.

        Args:
            summary: Summary text to save.

        Returns:
            Path to the saved log file.
        """
        filename = f"session_{self.session_timestamp}_summary.txt"
        filepath = config.LOGS_DIR / filename

        try:
            filepath.write_text(summary, encoding="utf-8")
            logger.info(f"Session summary saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")
            raise

    def get_output_directory(self, output_type: str = "plots") -> Path:
        """Get the output directory for a specific type.

        Args:
            output_type: Type of output ('plots', 'reports', 'models', 'logs').

        Returns:
            Path to the output directory.
        """
        dir_map = {
            "plots": config.PLOTS_DIR,
            "reports": config.REPORTS_DIR,
            "models": config.MODELS_DIR,
            "logs": config.LOGS_DIR,
        }
        return dir_map.get(output_type, config.OUTPUT_DIR)

    def list_outputs(self, output_type: str = "plots") -> list[Path]:
        """List all output files of a specific type.

        Args:
            output_type: Type of output ('plots', 'reports', 'models', 'logs').

        Returns:
            List of Path objects for files in that directory.
        """
        directory = self.get_output_directory(output_type)
        if not directory.exists():
            return []
        return sorted(directory.iterdir())

    def get_session_plots(self) -> list[Path]:
        """Get all plots from the current session.

        Returns:
            List of Path objects for plots from this session.
        """
        return [p for p in self.list_outputs("plots") if self.session_timestamp in p.name]

    def get_session_reports(self) -> list[Path]:
        """Get all reports from the current session.

        Returns:
            List of Path objects for reports from this session.
        """
        return [p for p in self.list_outputs("reports") if self.session_timestamp in p.name]


# Global instance
_manager: Optional[OutputsManager] = None


def get_outputs_manager() -> OutputsManager:
    """Get or create the global OutputsManager instance."""
    global _manager
    if _manager is None:
        _manager = OutputsManager()
    return _manager


def reset_outputs_manager() -> None:
    """Reset the global OutputsManager instance."""
    global _manager
    _manager = None
