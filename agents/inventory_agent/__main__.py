"""
Inventory Agent Package CLI Entry Point.
Allows running the inventory optimization recommendations demo directly from command line.
"""

import os
import sys

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.forecasting_agent.agent import ForecastingAgent
from agents.inventory_agent.agent import InventoryAgent
from agents.inventory_agent.task_manager import TaskManager


def run_demo() -> None:
    """
    Runs a complete demonstration of the InventoryAgent:
    1. Obtains demand predictions from the ForecastingAgent.
    2. Runs safety stock, ROP, EOQ, and reorder recommendation logic.
    3. Prints metrics summary and output file previews.
    """
    print("=== Inventory Agent Recommendation Demonstration ===")

    cleaned_path = os.path.join("data", "processed", "cleaned_data.csv")
    output_path = os.path.join("data", "processed", "inventory_recommendations.csv")

    if not os.path.exists(cleaned_path):
        print(f"\n[Error] Cleaned data file not found at: {cleaned_path}")
        print("Please run the DataAgent or HostAgent pipeline first to generate cleaned data.")
        return

    # Step 1: Generate Demand Forecasts using ForecastingAgent
    print(f"\n1. Requesting Demand Forecasts from ForecastingAgent...")
    forecasting_agent = ForecastingAgent()
    forecast_results = forecasting_agent.run_task({"data_path": cleaned_path})

    if forecast_results.get("status") != "success":
        print(f"[Error] Forecasting failed: {forecast_results.get('error')}")
        print("Falling back to historical demand values for inventory calculations...")
        predictions = None
    else:
        predictions = forecast_results.get("predictions")
        print("Demand forecasting succeeded!")
        print(f" - Mean Predicted Demand: {forecast_results['predictions_summary']['mean']:.2f}")
        print(f" - Row count: {forecast_results['row_count']}")

    # Step 2: Initialize InventoryAgent and TaskManager
    agent = InventoryAgent()
    manager = TaskManager(agent=agent)

    print(f"\n2. Creating inventory recommendation task...")
    task_id = manager.create_task({
        "data_path": cleaned_path,
        "predictions": predictions,
        "service_level_z": 1.65,  # 95% one-sided service level
        "output_path": output_path
    })
    print(f"Task created with ID: {task_id}")

    # Inspect task
    task = manager.get_task(task_id)
    print(f"Task status before execution: {task['status']}")

    print("\n3. Executing inventory recommendation task...")
    completed_task = manager.execute_task(task_id)

    print(f"Task status after execution: {completed_task['status']}")

    if completed_task["status"] == "COMPLETED":
        result = completed_task["result"]
        summary = result["inventory_summary"]

        print("\n4. Inventory Recommendations completed successfully!")
        print("\n=== Optimization Summary ===")
        print(f" - Total Items Processed              : {summary['total_items_processed']}")
        print(f" - Reorder Alerts (Stock <= ROP)     : {summary['reorder_alerts_count']}")
        print(f" - Overstock Alerts (Stock > ROP+2EOQ): {summary['overstock_alerts_count']}")
        print(f" - Optimal Stock Status (OK)          : {summary['ok_alerts_count']}")
        print(f" - Total Recommended Reorder Qty      : {summary['total_recommended_reorder_quantity']} units")
        print(f" - Total Recommended Reorder Cost     : ${summary['total_reorder_cost']:.2f}")
        print(f" - Total Current Stock Value          : ${summary['total_current_stock_value']:.2f}")
        print(f" - Average Safety Stock Level        : {summary['average_safety_stock']:.2f} units")
        print(f" - Average Reorder Point (ROP)        : {summary['average_reorder_point']:.2f} units")
        print(f" - Recommendations Saved To           : {result['output_path']}")

        # Read output and show preview
        try:
            import pandas as pd
            rec_df = pd.read_csv(result['output_path'])
            print("\n5. Recommendations Preview (First 5 Rows):")
            preview_cols = [
                "Product", "Inventory_Level", "predicted_demand", "Lead_Time",
                "Safety_Stock", "Reorder_Point", "EOQ", "Recommendation_Status", "Recommended_Order_Quantity"
            ]
            # Ensure columns exist in preview
            preview_cols = [col for col in preview_cols if col in rec_df.columns]
            print(rec_df[preview_cols].head())
        except Exception as e:
            print(f"\nCould not read recommendations file preview: {str(e)}")
    else:
        print(f"\n[Error] Task execution failed: {completed_task['error']}")


if __name__ == "__main__":
    run_demo()
