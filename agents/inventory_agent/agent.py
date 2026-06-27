"""
Inventory Agent Module.
Contains the InventoryAgent class responsible for safety stock, ROP, EOQ,
and replenishment recommendations based on demand forecasts.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class InventoryAgent:
    """
    A modular agent responsible for generating inventory optimization recommendations.
    Uses standard supply chain formulas (Safety Stock, Reorder Point, and EOQ).
    """

    def __init__(self) -> None:
        """
        Initializes the InventoryAgent.
        """
        pass

    def run_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the inventory analysis task.

        Args:
            task_details: A dictionary containing task parameters.
                Expected keys:
                - 'data_path' (str, optional): Path to the cleaned inventory CSV.
                  Defaults to 'data/processed/cleaned_data.csv'.
                - 'predictions' (List[float], optional): Demand predictions list.
                - 'service_level_z' (float, optional): Z-score service level multiplier.
                  Defaults to 1.65.
                - 'output_path' (str, optional): Path to save recommendations CSV.
                  Defaults to 'data/processed/inventory_recommendations.csv'.

        Returns:
            Dict[str, Any]: Execution results containing status, output path, summary metrics.
        """
        data_path = task_details.get("data_path", os.path.join("data", "processed", "cleaned_data.csv"))
        predictions = task_details.get("predictions")
        service_level_z = task_details.get("service_level_z", 1.65)
        output_path = task_details.get("output_path", os.path.join("data", "processed", "inventory_recommendations.csv"))

        if not os.path.exists(data_path):
            return {
                "status": "failed",
                "error": f"Cleaned data file not found at: {data_path}"
            }

        try:
            # 1. Load the cleaned dataset
            df = pd.read_csv(data_path)

            # 2. Merge forecasting output into the dataframe
            if predictions is not None:
                if len(predictions) != len(df):
                    return {
                        "status": "failed",
                        "error": f"Predictions size ({len(predictions)}) does not match dataset row count ({len(df)})."
                    }
                df["predicted_demand"] = predictions
            else:
                # Check if predicted_demand column already exists in the dataset
                if "predicted_demand" not in df.columns:
                    if "Demand" in df.columns:
                        # Fallback to historical demand
                        df["predicted_demand"] = df["Demand"]
                    else:
                        return {
                            "status": "failed",
                            "error": "No 'predictions' provided and no 'predicted_demand' or 'Demand' column found in the dataset."
                        }

            # 3. Calculate safety stock grouped by Product to get demand standard deviation
            df["demand_std"] = df.groupby("Product")["predicted_demand"].transform("std")
            
            # For products with insufficient history (std is NaN or 0), default std to 20% of demand
            df["demand_std"] = df["demand_std"].fillna(df["predicted_demand"] * 0.2)
            df.loc[df["demand_std"] <= 0, "demand_std"] = df["predicted_demand"] * 0.2
            df.loc[df["demand_std"] < 1.0, "demand_std"] = 1.0  # Floor standard deviation

            # Safety Stock calculation: Z * std(Demand) * sqrt(Lead_Time)
            lead_time = df["Lead_Time"].fillna(1.0)
            df["Safety_Stock"] = (service_level_z * df["demand_std"] * np.sqrt(lead_time)).round(2)

            # 4. Calculate Reorder Point (ROP): (Demand * Lead_Time) + Safety_Stock
            df["Reorder_Point"] = ((df["predicted_demand"] * lead_time) + df["Safety_Stock"]).round(2)

            # 5. Economic Order Quantity (EOQ): sqrt(2 * D * S / H)
            # S: Ordering setup cost (proxy: Transportation_Cost), H: Holding_Cost
            S = df["Transportation_Cost"].fillna(50.0)
            H = df["Holding_Cost"].fillna(5.0)
            # Avoid division by zero or negative holding cost
            H = H.apply(lambda val: val if val > 0.0 else 5.0)

            df["EOQ"] = np.sqrt((2 * df["predicted_demand"] * S) / H)
            df["EOQ"] = df["EOQ"].fillna(50.0).round().astype(int)
            df.loc[df["EOQ"] < 1, "EOQ"] = 1  # Floor EOQ to at least 1 unit

            # 6. Generate inventory recommendations based on current inventory level vs ROP
            df["Recommendation_Status"] = "OK"
            df["Recommended_Order_Quantity"] = 0

            # Condition 1: REORDER (Stock <= ROP)
            reorder_mask = df["Inventory_Level"] <= df["Reorder_Point"]
            df.loc[reorder_mask, "Recommendation_Status"] = "REORDER"
            df.loc[reorder_mask, "Recommended_Order_Quantity"] = df.loc[reorder_mask, "EOQ"]

            # Condition 2: OVERSTOCK (Stock > ROP + 2 * EOQ)
            overstock_mask = df["Inventory_Level"] > (df["Reorder_Point"] + (2 * df["EOQ"]))
            df.loc[overstock_mask, "Recommendation_Status"] = "OVERSTOCK"

            # 7. Calculate estimated cost metrics
            item_cost = df["Cost"].fillna(0.0)
            df["Current_Stock_Value"] = (df["Inventory_Level"] * item_cost).round(2)
            df["Reorder_Cost"] = (df["Recommended_Order_Quantity"] * item_cost).round(2)

            # 8. Ensure output directory exists and save to CSV
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Clean up temporary calculation columns before saving
            save_df = df.drop(columns=["demand_std"])
            save_df.to_csv(output_path, index=False)

            # 9. Aggregate summary statistics
            inventory_summary = {
                "total_items_processed": len(df),
                "reorder_alerts_count": int(reorder_mask.sum()),
                "overstock_alerts_count": int(overstock_mask.sum()),
                "ok_alerts_count": int((df["Recommendation_Status"] == "OK").sum()),
                "total_recommended_reorder_quantity": int(df["Recommended_Order_Quantity"].sum()),
                "total_reorder_cost": float(df["Reorder_Cost"].sum()),
                "total_current_stock_value": float(df["Current_Stock_Value"].sum()),
                "average_safety_stock": float(df["Safety_Stock"].mean()),
                "average_reorder_point": float(df["Reorder_Point"].mean())
            }

            return {
                "status": "success",
                "inventory_summary": inventory_summary,
                "output_path": output_path
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"An error occurred during inventory recommendation calculations: {str(e)}"
            }
