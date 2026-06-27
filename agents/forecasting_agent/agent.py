"""
Forecasting Agent Module.
Contains the ForecastingAgent class responsible for training and generating
predictions using the existing forecasting model.
"""

import os
from typing import Any, Dict, Optional
import pandas as pd

# Import the existing forecasting model
from models.forecasting_model import ForecastingModel


class ForecastingAgent:
    """
    A modular agent responsible for demand forecasting.
    Uses the existing ForecastingModel to train on cleaned data and predict demand.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes the ForecastingAgent with an instance of ForecastingModel.
        """
        self.model = ForecastingModel(random_state=random_state)
        self.last_metrics: Optional[Dict[str, float]] = None

    def run_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a forecasting task.

        Args:
            task_details: A dictionary containing task parameters.
                Expected keys:
                - 'data_path' (str, optional): Path to the cleaned CSV file.
                  Defaults to 'data/processed/cleaned_data.csv'.

        Returns:
            Dict[str, Any]: A dictionary containing the execution results:
                - 'status': 'success' or 'failed'
                - 'metrics': Evaluation metrics from model training (MSE, RMSE, MAE, R2).
                - 'target_column': The name of the target column (e.g., 'Demand').
                - 'predictions_summary': Dictionary containing summary stats (mean, min, max, total).
                - 'predictions': List of predicted values.
                - 'row_count': Total number of rows processed.
                - 'error': Error message, if failed.
        """
        data_path = task_details.get("data_path", os.path.join("data", "processed", "cleaned_data.csv"))

        if not os.path.exists(data_path):
            return {
                "status": "failed",
                "error": f"Cleaned data file not found at: {data_path}"
            }

        try:
            # 1. Train the model using existing train method
            metrics = self.model.train(data_path=data_path)
            self.last_metrics = metrics

            # 2. Predict future demand (generates prediction for all rows of the cleaned data)
            pred_df = self.model.predict_future_demand(data_path=data_path)

            # 3. Extract predictions list
            predictions = pred_df["predicted_demand"].tolist()

            # 4. Compute prediction statistics
            pred_series = pred_df["predicted_demand"]
            predictions_summary = {
                "mean": float(pred_series.mean()),
                "min": float(pred_series.min()),
                "max": float(pred_series.max()),
                "std": float(pred_series.std()) if len(pred_series) > 1 else 0.0,
                "total_predicted_demand": float(pred_series.sum())
            }

            return {
                "status": "success",
                "metrics": metrics,
                "target_column": self.model.target_col,
                "predictions_summary": predictions_summary,
                "predictions": predictions,
                "row_count": len(pred_df)
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"An error occurred during forecasting task: {str(e)}"
            }
