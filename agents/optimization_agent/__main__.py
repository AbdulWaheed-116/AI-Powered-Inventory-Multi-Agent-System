"""
Optimization Agent Package CLI Entry Point.
Allows running the inventory order optimization demo directly from the command line.
"""

import os
import sys

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.optimization_agent.agent import OptimizationAgent
from agents.optimization_agent.task_manager import TaskManager


def run_demo() -> None:
    """
    Runs a complete demonstration of the OptimizationAgent:
    1. Loads the Inventory Agent recommendations CSV.
    2. Runs cost-minimizing supplier selection and Lagrangian capacity constraints.
    3. Prints metrics summary and comparison tables.
    """
    print("=== Optimization Agent Replenishment Demonstration ===")

    recommendations_path = os.path.join("data", "processed", "inventory_recommendations.csv")
    output_path = os.path.join("data", "processed", "optimized_inventory.csv")

    if not os.path.exists(recommendations_path):
        print(f"\n[Error] Recommendations data file not found at: {recommendations_path}")
        print("Please run the InventoryAgent or HostAgent pipeline first to generate recommendations.")
        return

    # Initialize OptimizationAgent and TaskManager
    agent = OptimizationAgent()
    manager = TaskManager(agent=agent)

    print(f"\n1. Creating inventory optimization task...")
    task_id = manager.create_task({
        "data_path": recommendations_path,
        "output_path": output_path,
        "service_level_z": 1.65  # 95% service level
    })
    print(f"Task created with ID: {task_id}")

    # Inspect task before run
    task = manager.get_task(task_id)
    print(f"Task status before execution: {task['status']}")

    print("\n2. Executing optimization task...")
    completed_task = manager.execute_task(task_id)

    print(f"Task status after execution: {completed_task['status']}")

    if completed_task["status"] == "COMPLETED":
        result = completed_task["result"]
        summary = result["optimization_summary"]

        print("\n3. Inventory Optimization completed successfully!")
        print("\n=== Optimization Summary ===")
        print(f" - Total Items Processed             : {summary['total_items_processed']}")
        print(f" - Items Requiring Reorder           : {summary['reorder_items_count']}")
        print(f" - Original Reorder Quantity (EOQ)   : {summary['total_original_reorder_qty']} units")
        print(f" - Optimized Reorder Quantity       : {summary['total_optimized_reorder_qty']} units")
        print(f" - Original Reorder Cost             : ${summary['total_original_reorder_cost']:.2f}")
        print(f" - Optimized Reorder Cost           : ${summary['total_optimized_reorder_cost']:.2f}")
        print(f" - Estimated Total Savings           : ${summary['estimated_cost_savings']:.2f}")
        print(f" - Capacity Adjustments Resolved     : {summary['capacity_adjustments_count']} items")
        print(f" - Optimized Output Saved To         : {result['output_path']}")

        # Read output and show preview
        try:
            import pandas as pd
            opt_df = pd.read_csv(result['output_path'])
            
            # Show a table comparing original recommendations vs optimized recommendations for reordered items
            reorder_df = opt_df[opt_df["Recommendation_Status"] == "REORDER"]
            if not reorder_df.empty:
                print("\n4. Before & After Optimization Preview (First 5 Reordered Rows):")
                preview_cols = [
                    "Product", "Warehouse", "Supplier", "Optimized_Supplier", 
                    "Recommended_Order_Quantity", "Optimized_Order_Quantity", 
                    "Reorder_Cost", "Optimized_Reorder_Cost"
                ]
                preview_cols = [col for col in preview_cols if col in reorder_df.columns]
                print(reorder_df[preview_cols].head())
                
                # Check warehouse capacity enforcement
                print("\n5. Warehouse Capacity Constraints Audit:")
                for warehouse, group in opt_df.groupby("Warehouse"):
                    capacity = group["Storage_Capacity"].iloc[0]
                    initial_stock = group["Inventory_Level"].sum()
                    unopt_orders = group["Recommended_Order_Quantity"].sum()
                    opt_orders = group["Optimized_Order_Quantity"].sum()
                    
                    print(f" * {warehouse}:")
                    print(f"   - Capacity: {capacity} | Current Stock: {initial_stock}")
                    print(f"   - Original Projected Stock: {initial_stock + unopt_orders} "
                          f"({'EXCEEDED' if initial_stock + unopt_orders > capacity else 'OK'})")
                    print(f"   - Optimized Projected Stock: {initial_stock + opt_orders} "
                          f"({'EXCEEDED' if initial_stock + opt_orders > capacity else 'OK'})")
            else:
                print("\nNo reordered items in this batch to preview.")
        except Exception as e:
            print(f"\nCould not read optimization file preview: {str(e)}")
    else:
        print(f"\n[Error] Task execution failed: {completed_task['error']}")


if __name__ == "__main__":
    run_demo()
