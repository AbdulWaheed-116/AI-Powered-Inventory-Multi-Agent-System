"""
Host Agent Package CLI Entry Point.
Allows running the host agent pipeline demonstration directly from the command line.
"""

import os
import sys

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.host_agent.agent import HostAgent
from agents.host_agent.task_manager import TaskManager


def run_demo() -> None:
    """
    Runs a demonstration of the HostAgent by orchestrating DataAgent
    and ForecastingAgent to clean a raw dataset, train a model, and forecast demand.
    """
    print("=== Host Agent Coordinated Pipeline Demonstration ===")

    # Input and output paths
    raw_path = os.path.join("data", "raw", "pharma_supply_chain_messy_2.csv")
    cleaned_path = os.path.join("data", "processed", "cleaned_data.csv")
    inventory_path = os.path.join("data", "processed", "inventory_recommendations.csv")

    if not os.path.exists(raw_path):
        print(f"\n[Error] Raw dataset file not found at: {raw_path}")
        print("Please place 'pharma_supply_chain_messy_2.csv' in the 'data/raw' directory.")
        return

    # Initialize Agent and TaskManager
    agent = HostAgent()
    manager = TaskManager(agent=agent)

    print(f"\n1. Creating pipeline task with parameters:")
    print(f" - Raw Input: {raw_path}")
    print(f" - Cleaned Output: {cleaned_path}")
    print(f" - Recommendations Output: {inventory_path}")

    task_id = manager.create_task({
        "raw_data_path": raw_path,
        "cleaned_data_path": cleaned_path,
        "inventory_output_path": inventory_path,
        "service_level_z": 1.65
    })
    print(f"Task created with ID: {task_id}")

    # Inspect task before run
    task = manager.get_task(task_id)
    print(f"Task status before execution: {task['status']}")

    print("\n2. Executing task (running coordinated pipeline)...")
    completed_task = manager.execute_task(task_id)

    print(f"Task status after execution: {completed_task['status']}")

    if completed_task["status"] == "COMPLETED":
        result = completed_task["result"]
        data_res = result["data_agent_results"]
        forecast_res = result["forecasting_agent_results"]

        print("\n3. Data Cleaning Phase Completed successfully:")
        print(f" - Initial Rows: {data_res['initial_row_count']}")
        print(f" - Duplicates Removed: {data_res['duplicates_removed']}")
        print(f" - Cleaned Rows: {data_res['cleaned_row_count']}")
        print(f" - Output Saved To: {data_res['output_path']}")

        missing_val_rep = data_res["missing_values_report"]
        print(" - Missing Values Check:")
        print(f"   * Total missing values found: {missing_val_rep['total_missing']}")
        print(f"   * Missing per column: {missing_val_rep['missing_by_column']}")

        print("\n4. Demand Forecasting Phase Completed successfully! Evaluation Metrics:")
        metrics = forecast_res["metrics"]
        for metric, val in metrics.items():
            print(f" - {metric.upper():<5}: {val:.4f}")

        print("\n5. Predictions Summary Stats:")
        summary = forecast_res["predictions_summary"]
        print(f" - Target Column: {forecast_res['target_column']}")
        print(f" - Mean Predicted Demand: {summary['mean']:.2f}")
        print(f" - Min Predicted Demand: {summary['min']:.2f}")
        print(f" - Max Predicted Demand: {summary['max']:.2f}")
        print(f" - Std Dev of Predicted Demand: {summary['std']:.2f}")
        print(f" - Total Predicted Demand: {summary['total_predicted_demand']:.2f}")
        print(f" - Total Processed Rows: {forecast_res['row_count']}")

        print("\n6. First 5 Demand Predictions:")
        predictions = forecast_res["predictions"]
        for i, val in enumerate(predictions[:5]):
            print(f" - Row {i}: {val:.2f}")
        if len(predictions) > 5:
            print(f" - ... and {len(predictions) - 5} more rows.")

        print("\n7. Inventory Recommendation Phase Completed successfully! Summary:")
        inventory_res = result["inventory_agent_results"]
        inv_summary = inventory_res["inventory_summary"]
        print(f" - Reorder Alerts Count             : {inv_summary['reorder_alerts_count']}")
        print(f" - Overstock Alerts Count           : {inv_summary['overstock_alerts_count']}")
        print(f" - Optimal Stock Count (OK)         : {inv_summary['ok_alerts_count']}")
        print(f" - Total Recommended Reorder Qty    : {inv_summary['total_recommended_reorder_quantity']} units")
        print(f" - Estimated Reorder Cost           : ${inv_summary['total_reorder_cost']:.2f}")
        print(f" - Average Safety Stock Level       : {inv_summary['average_safety_stock']:.2f} units")
        print(f" - Average Reorder Point (ROP)      : {inv_summary['average_reorder_point']:.2f} units")
        print(f" - Optimization Output Saved To     : {inventory_res['output_path']}")
    else:
        print(f"\n[Error] Pipeline execution failed: {completed_task['error']}")
        # If there's partial data/forecasting agent results, print them
        if completed_task["result"] and "data_agent_results" in completed_task["result"]:
            data_res = completed_task["result"]["data_agent_results"]
            if data_res.get("status") == "success":
                print("\nNote: Data cleaning phase succeeded.")
                print(f" - Cleaned data is saved at: {data_res.get('output_path')}")


if __name__ == "__main__":
    run_demo()
