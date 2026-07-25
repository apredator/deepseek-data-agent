"""
Data Analysis Module for the AI Agent
Handles loading, processing, and analyzing data
"""

import os
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataAnalyzer:
    """Main data analyzer class"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data analyzer

        Args:
            data_path: Path to data file (CSV, Excel, JSON)
        """
        self.data: Optional[pd.DataFrame] = None
        self.data_path = data_path
        self.analysis_results: Dict[str, Any] = {}

        if data_path:
            self.load_data(data_path)

    def load_data(self, file_path: str) -> bool:
        """
        Load data from file

        Args:
            file_path: Path to data file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".csv":
                self.data = pd.read_csv(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                self.data = pd.read_excel(file_path)
            elif file_ext == ".json":
                self.data = pd.read_json(file_path)
            else:
                print(f"Unsupported file format: {file_ext}")
                return False

            self.data_path = file_path
            print(f"✓ Data loaded successfully. Shape: {self.data.shape}")
            return True

        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False

    def get_summary(self) -> str:
        """Get summary statistics of data"""
        if self.data is None:
            return "No data loaded"

        summary = f"""
DATA SUMMARY
============
Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns

Columns: {list(self.data.columns)}

Data Types:
{self.data.dtypes}

Missing Values:
{self.data.isnull().sum()}

Basic Statistics:
{self.data.describe()}
        """
        return summary

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        if self.data is None:
            return {}

        stats = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_stats": self.data.describe().to_dict(),
            "correlation": self.data.corr(numeric_only=True).to_dict(),
        }

        return stats

    def filter_data(
        self,
        column: str,
        operator: str,
        value: Any,
    ) -> Optional[pd.DataFrame]:
        """
        Filter data by condition

        Args:
            column: Column name
            operator: Comparison operator ('==', '>', '<', '>=', '<=', '!=')
            value: Value to compare

        Returns:
            Filtered dataframe or None
        """
        if self.data is None:
            return None

        try:
            if operator == "==":
                return self.data[self.data[column] == value]
            elif operator == ">":
                return self.data[self.data[column] > value]
            elif operator == "<":
                return self.data[self.data[column] < value]
            elif operator == ">=":
                return self.data[self.data[column] >= value]
            elif operator == "<=":
                return self.data[self.data[column] <= value]
            elif operator == "!=":
                return self.data[self.data[column] != value]
            else:
                print(f"Unknown operator: {operator}")
                return None
        except Exception as e:
            print(f"Error filtering data: {str(e)}")
            return None

    def group_and_aggregate(
        self,
        group_by: str,
        agg_column: str,
        agg_func: str = "mean",
    ) -> Optional[pd.Series]:
        """
        Group data and aggregate

        Args:
            group_by: Column to group by
            agg_column: Column to aggregate
            agg_func: Aggregation function ('mean', 'sum', 'count', 'min', 'max')

        Returns:
            Aggregated series or None
        """
        if self.data is None:
            return None

        try:
            return self.data.groupby(group_by)[agg_column].agg(agg_func)
        except Exception as e:
            print(f"Error in grouping: {str(e)}")
            return None

    def plot_histogram(
        self,
        column: str,
        bins: int = 30,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> bool:
        """
        Create histogram

        Args:
            column: Column to plot
            bins: Number of bins
            title: Plot title
            save_path: Path to save plot

        Returns:
            True if successful
        """
        if self.data is None:
            return False

        try:
            plt.figure(figsize=(10, 6))
            plt.hist(self.data[column], bins=bins, edgecolor="black")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(title or f"Histogram of {column}")
            plt.grid(alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Plot saved to {save_path}")
            else:
                plt.show()

            plt.close()
            return True

        except Exception as e:
            print(f"Error creating histogram: {str(e)}")
            return False

    def plot_scatter(
        self,
        x_column: str,
        y_column: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> bool:
        """
        Create scatter plot

        Args:
            x_column: X-axis column
            y_column: Y-axis column
            title: Plot title
            save_path: Path to save plot

        Returns:
            True if successful
        """
        if self.data is None:
            return False

        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[x_column], self.data[y_column], alpha=0.6)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(title or f"{x_column} vs {y_column}")
            plt.grid(alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Plot saved to {save_path}")
            else:
                plt.show()

            plt.close()
            return True

        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            return False

    def plot_correlation_heatmap(
        self,
        save_path: Optional[str] = None,
    ) -> bool:
        """
        Create correlation heatmap

        Args:
            save_path: Path to save plot

        Returns:
            True if successful
        """
        if self.data is None:
            return False

        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                numeric_data.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
            )
            plt.title("Correlation Heatmap")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Heatmap saved to {save_path}")
            else:
                plt.show()

            plt.close()
            return True

        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            return False

    def export_results(self, output_path: str) -> bool:
        """
        Export analysis results to CSV

        Args:
            output_path: Path to save results

        Returns:
            True if successful
        """
        if self.data is None:
            return False

        try:
            self.data.to_csv(output_path, index=False)
            print(f"✓ Results exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
            return False