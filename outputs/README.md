# Outputs Directory

This directory contains all generated outputs from the LLM-Powered Data Science Assistant.

## Structure

- **plots/** - Saved visualization files (PNG, PDF, etc.)
- **reports/** - Data analysis reports and summaries (CSV, JSON, Excel, HTML)
- **models/** - Trained machine learning models and artifacts (pickle, joblib, etc.)
- **logs/** - Session logs and execution summaries

## Usage

### Saving Plots

Plots are automatically saved to `outputs/plots/` when you use visualization functions. 
Filenames are automatically timestamped for organization.

Example from the agent:
```python
# Create a visualization
plt.scatter(x, y)
plt.title("My Analysis")

# Save it (this will save to outputs/plots/)
save_plot(plot_name="my_scatter_plot")
```

### Saving Reports

Data analysis reports are saved to `outputs/reports/` in various formats (CSV, JSON, HTML, Excel).

Example:
```python
# Save a summary as CSV
save_dataframe_to_csv(file_name="analysis_results")

# Save metrics as JSON
save_json_report(data, report_name="model_metrics")
```

### Configuration

Output behavior is controlled by `config.py` in the project root:

- `AUTO_SAVE_PLOTS` - Automatically save all plots
- `AUTO_SAVE_REPORTS` - Automatically save all reports
- `TIMESTAMP_OUTPUTS` - Add timestamps to filenames
- `OUTPUT_FORMATS` - Configure output file formats and options

## File Naming

By default, all files include a timestamp for easy tracking:
- Format: `YYYYMMDD_HHMMSS_filename.ext`
- Example: `20260422_143022_scatter_analysis.png`

To disable timestamps, set `TIMESTAMP_OUTPUTS = False` in `config.py`.

## Size Management

Generated files can grow over time. Periodically clean up old outputs:

```bash
# Keep only recent outputs
rm outputs/plots/202603*  # Remove March outputs
```

Or from Python:
```python
from utils.outputs_manager import get_outputs_manager
manager = get_outputs_manager()

# List all plots
plots = manager.list_outputs('plots')

# Get session-specific outputs
session_plots = manager.get_session_plots()
```

## Notes

- All output directories are created automatically on first use
- `.gitignore` is configured to exclude this directory from version control
- Output paths are logged for easy reference
