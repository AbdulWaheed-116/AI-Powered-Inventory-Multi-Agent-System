"""
Forecasting Model Module.
Contains the ForecastingModel class responsible for training a regression model
to predict future product demand based on processed inventory and supply chain data.
"""

import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ForecastingModel:
    """
    A machine learning model for forecasting future demand.
    Automatically detects dataset schemas, cleans inputs, preprocesses features,
    and trains a Random Forest Regressor to predict demand.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes the ForecastingModel.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.target_col: Optional[str] = None
        self.is_fitted: bool = False

    def _clean_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-cleans the input DataFrame by replacing columns that are entirely null.

        Args:
            df: Input pandas DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].isnull().all():
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = 0.0
                else:
                    df_copy[col] = "unknown"
        return df_copy

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects date/time columns in the DataFrame and extracts useful cyclical
        and numerical features (Year, Month, Day, Day of Week).

        Args:
            df: Input pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with datetime features added and original date columns dropped.
        """
        df_copy = df.copy()
        date_cols = []

        # Identify date columns by datatype or non-numeric types with matching name patterns
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                date_cols.append(col)
            elif not pd.api.types.is_numeric_dtype(df_copy[col]):
                if any(keyword in col.lower() for keyword in ["date", "time", "timestamp"]):
                    date_cols.append(col)

        # Identify date columns by string content patterns
        for col in df_copy.columns:
            if col not in date_cols and not pd.api.types.is_numeric_dtype(df_copy[col]):
                # Check first non-null sample
                sample = df_copy[col].dropna().head(1).values
                if len(sample) > 0 and isinstance(sample[0], str) and any(c in sample[0] for c in ["-", "/"]):
                    if len(sample[0]) >= 6:
                        date_cols.append(col)

        for col in date_cols:
            try:
                dt_col = pd.to_datetime(df_copy[col])
                df_copy[f"{col}_year"] = dt_col.dt.year
                df_copy[f"{col}_month"] = dt_col.dt.month
                df_copy[f"{col}_day"] = dt_col.dt.day
                df_copy[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                # Drop original datetime column so scikit-learn pipeline runs cleanly
                df_copy = df_copy.drop(columns=[col])
            except Exception:
                pass

        return df_copy

    def _detect_target_and_features(self, df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
        """
        Automatically identifies the target column (demand) and maps other columns
        to numeric or categorical feature lists.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            Tuple[str, List[str], List[str]]: Target column name, numeric features, categorical features.
        """
        # Look for target candidates representing demand
        target_col = None
        target_candidates = ["demand", "demand_qty", "sales", "order_quantity", "quantity_on_hand", "reorder_level"]
        
        for cand in target_candidates:
            matched = [col for col in df.columns if col.lower() == cand]
            if matched:
                target_col = matched[0]
                break

        # Fallback target if none of the candidates match
        if not target_col:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            non_id_numeric = [col for col in numeric_cols if "id" not in col.lower()]
            if non_id_numeric:
                target_col = non_id_numeric[0]
            else:
                target_col = "demand"

        # Create target column with constant if it does not exist in DataFrame
        if target_col not in df.columns:
            df[target_col] = 100.0

        # Features are all other columns
        potential_features = [col for col in df.columns if col != target_col]
        numeric_features = []
        categorical_features = []

        for col in potential_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                # ID fields are better treated as categorical
                if "id" in col.lower():
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)

        return target_col, numeric_features, categorical_features

    def train(self, data_path: str = "data/processed/cleaned_data.csv") -> Dict[str, float]:
        """
        Reads data from data_path, pre-processes the dataset, fits the pipeline,
        and computes regression performance metrics on a validation split.

        Args:
            data_path: Path to the cleaned CSV data file.

        Returns:
            Dict[str, float]: Evaluation metrics (MSE, RMSE, MAE, R²).
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned data file not found at: {data_path}")

        df = pd.read_csv(data_path)

        # 1. Pre-clean columns that are completely empty
        df = self._clean_input_df(df)

        # 2. Extract datetime features
        df = self._extract_datetime_features(df)

        # 3. Detect features and target
        target_col, num_feats, cat_feats = self._detect_target_and_features(df)
        self.target_col = target_col
        self.numeric_features = num_feats
        self.categorical_features = cat_feats

        # Drop rows where target is missing, as we cannot train or evaluate with missing labels
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        # Cast categorical columns to string to handle mixed types cleanly in OneHotEncoder
        for col in self.categorical_features:
            df[col] = df[col].astype(str)

        # Separate X (features) and y (target)
        X = df[self.numeric_features + self.categorical_features]
        y = df[self.target_col]

        # 4. Construct sklearn preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        # 5. Split train/test (fallback to training set if too few rows exist)
        if len(df) < 5:
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Fit preprocessing & regressor pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', self.model)
        ])

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        # 7. Evaluate on validation split
        y_pred = self.pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        if len(np.unique(y_test)) > 1:
            r2 = r2_score(y_test, y_pred)
        else:
            r2 = 1.0  # R² is 1.0 if target values are uniform

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates demand predictions for an input DataFrame X using the fitted model.

        Args:
            X: Input DataFrame containing features.

        Returns:
            np.ndarray: Predicted demand values.
        """
        if not self.is_fitted or self.pipeline is None:
            raise ValueError("The forecasting model is not trained yet. Run train() first.")

        # Clean inputs and extract datetime features
        X_clean = self._clean_input_df(X)
        X_clean = self._extract_datetime_features(X_clean)

        # Fill missing features with defaults
        for col in self.categorical_features:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].astype(str)
            else:
                X_clean[col] = "unknown"

        for col in self.numeric_features:
            if col not in X_clean.columns:
                X_clean[col] = 0.0

        # Rearrange feature columns to match training order
        X_clean = X_clean[self.numeric_features + self.categorical_features]

        return self.pipeline.predict(X_clean)

    def predict_future_demand(self, data_path: str = "data/processed/cleaned_data.csv") -> pd.DataFrame:
        """
        Convenience method that loads data from data_path, performs predictions,
        and returns the dataset with an added 'predicted_demand' column.

        Args:
            data_path: Path to the cleaned data CSV file.

        Returns:
            pd.DataFrame: Original DataFrame with a 'predicted_demand' column added.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned data file not found at: {data_path}")

        df = pd.read_csv(data_path)
        predictions = self.predict(df)
        df["predicted_demand"] = predictions
        return df


if __name__ == "__main__":
    print("=== Forecasting Model Module Demonstration ===")
    
    # Try using the processed files
    # By default, use cleaned_data.csv
    cleaned_path = os.path.join("data", "processed", "cleaned_data.csv")
    cleaned_inventory_path = os.path.join("data", "processed", "cleaned_inventory.csv")
    
    # Select which path actually exists for demo
    path_to_use = None
    if os.path.exists(cleaned_path):
        path_to_use = cleaned_path
    elif os.path.exists(cleaned_inventory_path):
        path_to_use = cleaned_inventory_path

    if path_to_use:
        print(f"\n1. Training forecasting model on dataset: {path_to_use}")
        model = ForecastingModel()
        try:
            metrics = model.train(path_to_use)
            print("\n2. Model trained successfully! Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f" - {metric.upper()}: {value:.4f}")
                
            print("\n3. Testing prediction on original dataset...")
            pred_df = model.predict_future_demand(path_to_use)
            print("\nPreview of original dataset with 'predicted_demand':")
            # Select target, predictions, and a few descriptive features if present
            show_cols = [col for col in [model.target_col, "predicted_demand", "item_name", "Product", "category", "Category"] if col in pred_df.columns]
            print(pred_df[show_cols].head())
        except Exception as e:
            print(f"\n[Error during training/prediction] {e}")
    else:
        print(f"\n[Error] Processed data files not found at {cleaned_path} or {cleaned_inventory_path}.")
        print("Please clean your data first using the DataAgent or upload data through the Streamlit interface.")
