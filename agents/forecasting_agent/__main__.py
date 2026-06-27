"""
Forecasting Agent Package CLI Entry Point.
Allows running the forecasting agent demonstration directly from the command line.
"""

import os
import sys

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.forecasting_agent.agent import ForecastingAgent
from agents.forecasting_agent.task_manager import TaskManager


def run_demo() -> None:
    """
    Runs a demonstration of the ForecastingAgent by loading the default
    cleaned dataset, training the model, generating predictions, and printing metrics.
    """
    print("=== Forecasting Agent Demonstration ===")

    # Path to the processed inventory cleaned CSV
    cleaned_path = os.path.join("data", "processed", "cleaned_data.csv")

    if not os.path.exists(cleaned_path):
        print(f"\n[Error] Cleaned data file not found at: {cleaned_path}")
        print("Please clean your data first using the DataAgent or check the file directory.")
        return

    # Initialize Agent and TaskManager
    agent = ForecastingAgent()
    manager = TaskManager(agent=agent)

    print(f"\n1. Creating forecasting task using file: {cleaned_path}")
    task_id = manager.create_task({"data_path": cleaned_path})
    print(f"Task created with ID: {task_id}")

    # Inspect task
    task = manager.get_task(task_id)
    print(f"Task status before execution: {task['status']}")

    print("\n2. Executing task...")
    completed_task = manager.execute_task(task_id)

    print(f"Task status after execution: {completed_task['status']}")

    if completed_task["status"] == "COMPLETED":
        result = completed_task["result"]
        print("\n3. Forecasting completed successfully! Evaluation Metrics:")
        metrics = result["metrics"]
        for metric, val in metrics.items():
            print(f" - {metric.upper():<5}: {val:.4f}")

        print("\n4. Predictions Summary Stats:")
        summary = result["predictions_summary"]
        print(f" - Target Column: {result['target_column']}")
        print(f" - Mean Predicted Demand: {summary['mean']:.2f}")
        print(f" - Min Predicted Demand: {summary['min']:.2f}")
        print(f" - Max Predicted Demand: {summary['max']:.2f}")
        print(f" - Std Dev of Predicted Demand: {summary['std']:.2f}")
        print(f" - Total Predicted Demand: {summary['total_predicted_demand']:.2f}")
        print(f" - Total Processed Rows: {result['row_count']}")

        print("\n5. First 5 Prediction Values:")
        predictions = result["predictions"]
        for i, val in enumerate(predictions[:5]):
            print(f" - Row {i}: {val:.2f}")
        if len(predictions) > 5:
            print(f" - ... and {len(predictions) - 5} more rows.")
    else:
        print(f"\n[Error] Task execution failed: {completed_task['error']}")


if __name__ == "__main__":
    run_demo()
