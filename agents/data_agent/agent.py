"""
Data Agent Module.
Contains the DataAgent class responsible for reading, checking, cleaning,
and saving inventory CSV files.
"""

import os
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class DataAgent:
    """
    A modular agent responsible for basic data processing operations on inventory CSV files.
    """

    def __init__(self) -> None:
        """
        Initializes the DataAgent with an empty DataFrame placeholder.
        """
        self.df: Optional[pd.DataFrame] = None

    def read_csv(self, file_path_or_buffer: Any) -> pd.DataFrame:
        """
        Reads a CSV file from a file path or a file-like buffer.

        Args:
            file_path_or_buffer: An absolute/relative file path, a URL, or a
                                 file-like object/buffer (e.g. from an upload).

        Returns:
            pd.DataFrame: The loaded pandas DataFrame.
        """
        self.df = pd.read_csv(file_path_or_buffer)
        return self.df

    def check_missing_values(self) -> Dict[str, Any]:
        """
        Checks for missing/null values in the loaded dataset.

        Returns:
            Dict[str, Any]: A report summarizing missing values:
                - 'missing_by_column' (Dict[str, int]): Count of missing values per column.
                - 'total_missing' (int): Total number of missing values.
                - 'has_missing' (bool): True if any missing values exist, False otherwise.

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call read_csv() before checking missing values.")

        missing_series = self.df.isnull().sum()
        missing_by_column = missing_series.to_dict()
        total_missing = int(missing_series.sum())

        return {
            "missing_by_column": missing_by_column,
            "total_missing": total_missing,
            "has_missing": total_missing > 0,
        }

    def impute_missing_values(self) -> pd.DataFrame:
        """
        Imputes missing values in the loaded dataset:
        - Numerical columns are filled with their median.
        - Categorical columns are filled with their mode (most frequent value).
        If a column is entirely null, fills it with a default value (0.0 for numeric, 'unknown' for categorical).

        Returns:
            pd.DataFrame: The imputed DataFrame.

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call read_csv() before cleaning.")

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].isnull().all():
                    self.df[col] = 0.0
                else:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
            else:
                if self.df[col].isnull().all():
                    self.df[col] = "unknown"
                else:
                    mode_series = self.df[col].mode()
                    mode_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
                    self.df[col] = self.df[col].fillna(mode_val)

        return self.df

    def remove_duplicates(self) -> Tuple[int, pd.DataFrame]:
        """
        Removes duplicate rows from the loaded dataset, then imputes missing values
        (filling numeric with median, categorical with mode).

        Returns:
            Tuple[int, pd.DataFrame]: A tuple containing:
                - int: The number of duplicate rows removed.
                - pd.DataFrame: The cleaned DataFrame.

        Raises:
            ValueError: If no data has been loaded yet.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call read_csv() before removing duplicates.")

        # Remove duplicates first so they don't skew median/mode calculations
        initial_row_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        final_row_count = len(self.df)

        removed_count = initial_row_count - final_row_count

        # Impute missing values next
        self.impute_missing_values()

        return removed_count, self.df

    def save_cleaned_csv(self, output_path: str) -> str:
        """
        Saves the current DataFrame to a CSV file. Creates parent directories if they don't exist.
        Verifies that no missing (NaN) values remain in the DataFrame before saving.

        Args:
            output_path: File path where the cleaned CSV will be written.

        Returns:
            str: The output file path.

        Raises:
            ValueError: If no data has been loaded yet or if missing values still remain.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call read_csv() before saving.")

        # Verify no missing values remain
        total_missing = int(self.df.isnull().sum().sum())
        if total_missing > 0:
            raise ValueError(
                f"Validation failed: The dataset still contains {total_missing} missing values. "
                "Ensure that impute_missing_values() is run before saving."
            )

        # Ensure directory structure exists
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    # This block allows the script to be run directly as a demonstration.
    # It demonstrates the data cleaning pipeline using the raw sample dataset.
    print("=== Data Agent Demonstration ===")
    
    # Path to the sample inventory raw CSV
    raw_path = os.path.join("data", "raw", "sample_inventory.csv")
    cleaned_path = os.path.join("data", "processed", "cleaned_inventory.csv")
    
    if os.path.exists(raw_path):
        agent = DataAgent()
        
        print(f"\n1. Reading CSV from: {raw_path}")
        df = agent.read_csv(raw_path)
        print(f"Loaded dataset containing {len(df)} rows.")
        
        print("\n2. Checking for missing values...")
        report = agent.check_missing_values()
        print("Missing Values Summary:")
        print(f" - Total Missing Values: {report['total_missing']}")
        print(f" - Missing per Column: {report['missing_by_column']}")
        print(f" - Has missing: {report['has_missing']}")
        
        print("\n3. Removing duplicate rows...")
        duplicates_removed, df_cleaned = agent.remove_duplicates()
        print(f"Removed {duplicates_removed} duplicate rows.")
        print(f"Cleaned dataset row count: {len(df_cleaned)}")
        
        print(f"\n4. Saving cleaned CSV to: {cleaned_path}")
        agent.save_cleaned_csv(cleaned_path)
        print("Dataset successfully cleaned and saved!")
    else:
        print(f"\n[Error] Raw sample inventory file not found at: {raw_path}")
        print("Please ensure the CSV file is placed in the 'data/raw' directory.")

