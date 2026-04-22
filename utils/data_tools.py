from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.outputs_manager import get_outputs_manager

logger = logging.getLogger(__name__)


class DataScienceToolkit:
    """Toolkit that contains the full set of notebook-style data science utilities."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def auto_detect_data_types(self) -> Dict[str, List[str]]:
        types: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "binary": [],
            "other": [],
        }
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if self.df[col].nunique(dropna=True) <= 2:
                    types["binary"].append(col)
                else:
                    types["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                types["datetime"].append(col)
            elif pd.api.types.is_string_dtype(dtype):
                if self.df[col].nunique(dropna=True) <= 50:
                    types["categorical"].append(col)
                else:
                    types["text"].append(col)
            else:
                types["other"].append(col)
        return types

    def check_missing_values(self, data: str = "all") -> Dict[str, int]:
        if data == "all":
            return {k: int(v) for k, v in self.df.isna().sum().to_dict().items()}
        if data not in self.df.columns:
            return {"error": 1}
        return {data: int(self.df[data].isna().sum())}

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "shape": list(self.df.shape),
            "columns": self.df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "description": self.df.describe(include="all").to_dict(),
        }

    def get_column_stats(self, column_name: str) -> Dict[str, Any]:
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found in dataset"}

        col = self.df[column_name]
        values: Dict[str, Any] = {
            "dtype": str(col.dtype),
            "unique_values": int(col.nunique(dropna=True)),
            "missing_values": int(col.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(col):
            values.update(
                {
                    "mean": float(col.mean()),
                    "median": float(col.median()),
                    "std": float(col.std()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                }
            )
        return values

    def handle_missing_values(self, strategy: str = "median", columns: Optional[List[str]] = None) -> Dict[str, Any]:
        columns = columns or self.df.columns.tolist()
        for column in columns:
            if column not in self.df.columns:
                continue
            if self.df[column].isna().sum() == 0:
                continue

            if strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[column]):
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self.df[column]):
                self.df[column] = self.df[column].fillna(self.df[column].median())
            elif strategy == "mode":
                mode_values = self.df[column].mode(dropna=True)
                if not mode_values.empty:
                    self.df[column] = self.df[column].fillna(mode_values.iloc[0])
            elif strategy == "drop":
                self.df = self.df.dropna(subset=[column])
        return {"message": "Missing values handled successfully."}

    def encode_categorical_variables(
        self, encoding_type: str = "onehot", columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if columns is None:
            columns = self.auto_detect_data_types().get("categorical", [])

        encoded_columns: List[str] = []
        for column in columns:
            if column not in self.df.columns:
                continue
            if encoding_type == "onehot":
                encoded = pd.get_dummies(self.df[column], prefix=column)
                self.df = pd.concat([self.df.drop(columns=[column]), encoded], axis=1)
                encoded_columns.append(column)
            elif encoding_type == "label":
                encoder = LabelEncoder()
                self.df[column] = encoder.fit_transform(self.df[column].astype(str))
                encoded_columns.append(column)
        return {"encoded_columns": encoded_columns, "encoding_type": encoding_type}

    def scale_features(self, scaling_method: str = "standard", columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns is None:
            columns = self.auto_detect_data_types().get("numeric", [])
        if not columns:
            return {"message": "No numeric columns to scale."}

        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            return {"error": f"Unknown scaling method: {scaling_method}"}

        valid_columns = [c for c in columns if c in self.df.columns]
        if not valid_columns:
            return {"message": "No valid columns found to scale."}

        self.df[valid_columns] = scaler.fit_transform(self.df[valid_columns])
        return {"scaled_columns": valid_columns, "scaling_method": scaling_method}

    def feature_selection(self, method: str = "variance", threshold: float = 0.0) -> Dict[str, Any]:
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "No numeric columns available for feature selection."}

        if method == "variance":
            selector = VarianceThreshold(threshold)
            selector.fit(numeric_df)
            kept = numeric_df.columns[selector.get_support(indices=True)].tolist()
            self.df = self.df[kept].copy()
            return {"kept_features": kept, "method": method}

        if method == "correlation":
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            self.df = self.df.drop(columns=to_drop)
            return {"dropped_features": to_drop, "method": method}

        return {"error": f"Unknown feature selection method: {method}"}

    def create_polynomial_features(
        self, degree: int = 2, interaction_only: bool = False, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if columns is None:
            columns = self.auto_detect_data_types().get("numeric", [])

        valid_columns = [c for c in columns if c in self.df.columns]
        if not valid_columns:
            return {"error": "No numeric columns available for polynomial features."}

        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        transformed = poly.fit_transform(self.df[valid_columns])
        feature_names = poly.get_feature_names_out(valid_columns)
        poly_df = pd.DataFrame(transformed, columns=feature_names, index=self.df.index)

        self.df = self.df.drop(columns=valid_columns)
        self.df = pd.concat([self.df, poly_df], axis=1)
        return {"generated_feature_count": len(feature_names)}

    def detect_and_handle_outliers(
        self, column_name: str, method: str = "zscore", threshold: float = 3.0
    ) -> Dict[str, Any]:
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found in dataset"}
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            return {"error": "Column must be numeric"}

        col = self.df[column_name].dropna()
        if col.empty:
            return {"error": "Column has no values"}

        if method == "zscore":
            z_scores = stats.zscore(col, nan_policy="omit")
            outlier_mask = np.abs(z_scores) > threshold
        elif method == "iqr":
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (col < (q1 - threshold * iqr)) | (col > (q3 + threshold * iqr))
        else:
            return {"error": "Method must be 'zscore' or 'iqr'"}

        outliers_count = int(np.sum(outlier_mask))
        return {
            "total_outliers": outliers_count,
            "outlier_percentage": float((outliers_count / len(col)) * 100),
            "original_stats": {"mean": float(col.mean()), "std": float(col.std())},
        }

    def analyze_column_distribution(self, column_name: str, plot: bool = False) -> Dict[str, Any]:
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found in dataset"}

        result: Dict[str, Any] = {
            "skewness": float(self.df[column_name].skew()) if pd.api.types.is_numeric_dtype(self.df[column_name]) else None,
            "kurtosis": float(self.df[column_name].kurtosis()) if pd.api.types.is_numeric_dtype(self.df[column_name]) else None,
            "normality_test": None,
        }

        if pd.api.types.is_numeric_dtype(self.df[column_name]) and len(self.df[column_name].dropna()) >= 3:
            _, p_value = stats.shapiro(self.df[column_name].dropna())
            result["normality_test"] = {
                "test": "shapiro",
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05),
            }

        if plot:
            plt.figure(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(self.df[column_name]):
                sns.histplot(data=self.df, x=column_name, kde=True, bins=30, color="skyblue", edgecolor="white")
            else:
                sns.countplot(data=self.df, x=column_name)
            plt.title(f"Distribution of {column_name}")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return result

    def _prepare_model_input(self, target_column: str) -> Dict[str, Any]:
        if target_column not in self.df.columns:
            return {"error": f"Target column '{target_column}' not found in dataset"}

        model_df = self.df.dropna().copy()
        if model_df.empty:
            return {"error": "Dataset has no rows after dropping missing values."}

        y = model_df[target_column]
        X = model_df.drop(columns=[target_column])
        X = pd.get_dummies(X, drop_first=True)

        if X.empty:
            return {"error": "No features available after preprocessing."}

        return {"X": X, "y": y}

    def train_and_evaluate_classification_models(
        self, target_column: str, cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        prepared = self._prepare_model_input(target_column)
        if "error" in prepared:
            return {"error": {"message": prepared["error"]}}

        X = prepared["X"]
        y = prepared["y"]

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results: Dict[str, Dict[str, float]] = {}
        for model_name, model in models.items():
            pipeline = Pipeline([("model", model)])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            results[model_name] = {
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
            }
        return results

    def train_and_evaluate_regression_models(
        self, target_column: str, cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        prepared = self._prepare_model_input(target_column)
        if "error" in prepared:
            return {"error": {"message": prepared["error"]}}

        X = prepared["X"]
        y = prepared["y"]

        if not pd.api.types.is_numeric_dtype(y):
            return {"error": {"message": "Target column must be numeric for regression."}}

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Support Vector Regressor": SVR(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results: Dict[str, Dict[str, float]] = {}
        for model_name, model in models.items():
            pipeline = Pipeline([("model", model)])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
            mse_scores = -scores
            results[model_name] = {
                "mse_mean": float(mse_scores.mean()),
                "mse_std": float(mse_scores.std()),
            }
        return results

    def operations_on_dataset(
        self,
        task: str,
        column_1: Optional[str] = None,
        column_2: Optional[str] = None,
        operation: Optional[str] = None,
        group_by: Optional[str] = None,
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.df.empty:
            return {"error": "The dataset is empty."}

        if task not in ["calculate", "filter", "describe", "group", "visualize"]:
            return {"error": f"Task '{task}' is not supported."}

        if task == "calculate":
            if not column_1 or not operation:
                return {"error": "For 'calculate', provide column_1 and operation."}
            if column_1 not in self.df.columns:
                return {"error": f"Column '{column_1}' not found in dataset."}

            if operation in ["add", "subtract", "multiply", "divide"]:
                if not column_2 or column_2 not in self.df.columns:
                    return {"error": f"Column '{column_2}' must be provided for '{operation}'."}
                if operation == "add":
                    self.df[f"{column_1}_plus_{column_2}"] = self.df[column_1] + self.df[column_2]
                elif operation == "subtract":
                    self.df[f"{column_1}_minus_{column_2}"] = self.df[column_1] - self.df[column_2]
                elif operation == "multiply":
                    self.df[f"{column_1}_times_{column_2}"] = self.df[column_1] * self.df[column_2]
                elif operation == "divide":
                    self.df[f"{column_1}_per_{column_2}"] = self.df[column_1] / self.df[column_2].replace(0, np.nan)
                return {
                    "result": f"New column created using '{operation}' between '{column_1}' and '{column_2}'.",
                    "data_preview": self.df.head().to_dict(),
                }

            if operation in ["mean", "median", "std", "sum", "min", "max"]:
                value = getattr(self.df[column_1], operation)()
                return {"result": f"{operation.capitalize()} of '{column_1}' is {float(value):.2f}"}

            return {"error": f"Operation '{operation}' is not recognized."}

        if task == "filter":
            if not filter_column or filter_column not in self.df.columns:
                return {"error": f"Invalid filter column '{filter_column}' specified."}
            filtered_data = self.df[self.df[filter_column] == filter_value]
            return {
                "result": f"Filtered {len(filtered_data)} rows.",
                "data_preview": filtered_data.head().to_dict(),
            }

        if task == "describe":
            return {
                "result": "Summary statistics generated.",
                "describe": self.df.describe(include="all").to_dict(),
            }

        if task == "group":
            if not group_by or group_by not in self.df.columns:
                return {"error": f"Group-by column '{group_by}' not found in dataset."}
            if not column_1 or operation not in ["mean", "sum", "count", "min", "max"]:
                return {"error": "Invalid column or operation specified for grouping."}
            grouped_data = self.df.groupby(group_by)[column_1].agg(operation).reset_index()
            return {
                "result": f"Data grouped by '{group_by}' with '{operation}' on '{column_1}'.",
                "grouped_data": grouped_data.to_dict(),
            }

        if task == "visualize":
            if not column_1 or column_1 not in self.df.columns:
                return {"error": "For 'visualize', provide a valid column_1."}
            plt.figure(figsize=(8, 5))
            sns.histplot(data=self.df, x=column_1, kde=True)
            plt.title(f"Distribution of {column_1}")
            plt.tight_layout()
            plt.show()
            return {"result": f"Histogram for '{column_1}' displayed."}

        return {"error": "Invalid task specified."}

    def dimensionality_reduction(
        self, method: str = "pca", n_components: int = 2, visualize: bool = True
    ) -> Dict[str, Any]:
        numeric_data = self.df.select_dtypes(include=np.number).dropna()
        if numeric_data.empty:
            return {"error": "No numeric data available for dimensionality reduction."}

        if method == "pca":
            model = PCA(n_components=n_components)
            reduced_data = model.fit_transform(numeric_data)
            explained_variance = float(model.explained_variance_ratio_.sum())
        elif method == "tsne":
            model = TSNE(n_components=n_components, random_state=42)
            reduced_data = model.fit_transform(numeric_data)
            explained_variance = None
        elif method == "umap":
            model = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = model.fit_transform(numeric_data)
            explained_variance = None
        else:
            return {"error": f"Invalid method: {method}. Choose from 'pca', 'tsne', 'umap'."}

        if visualize and n_components <= 3:
            fig = plt.figure(figsize=(10, 8))
            if n_components == 2:
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c="blue", alpha=0.5)
                plt.title(f"{method.upper()} Visualization (2D)")
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c="blue", alpha=0.5)
                ax.set_title(f"{method.upper()} Visualization (3D)")
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            plt.tight_layout()
            plt.show()

        return {
            "reduced_data_preview": reduced_data[:5].tolist(),
            "explained_variance": explained_variance,
        }

    def save_dataframe_to_csv(self, file_name: str = "output") -> Dict[str, str]:
        """Save DataFrame using the output manager with automatic directory handling.
        
        Args:
            file_name: Name for the file (without extension).
            
        Returns:
            Dictionary with save status and file path.
        """
        manager = get_outputs_manager()
        file_path = manager.save_dataframe_report(self.df, file_name, format_type="csv")
        return {
            "message": f"DataFrame saved successfully",
            "file_path": str(file_path),
        }

    def save_plot(self, plot_name: str = "plot", close_figure: bool = True) -> Dict[str, str]:
        """Save the current matplotlib figure using the output manager.
        
        Args:
            plot_name: Name for the plot file (without extension).
            close_figure: Whether to close the figure after saving.
            
        Returns:
            Dictionary with save status and file path.
        """
        manager = get_outputs_manager()
        file_path = manager.save_plot(plot_name=plot_name, close_figure=close_figure)
        return {
            "message": f"Plot saved successfully",
            "file_path": str(file_path),
        }
