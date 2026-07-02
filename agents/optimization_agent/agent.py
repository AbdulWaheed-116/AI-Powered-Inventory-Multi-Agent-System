"""
Optimization Agent Module.
Contains the OptimizationAgent class responsible for reorder quantity optimization,
supplier selection, cost tradeoffs, and warehouse capacity constraint enforcement.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class OptimizationAgent:
    """
    A modular agent responsible for optimizing reorder quantities and supplier selection,
    considering purchase cost, ordering cost, holding cost, and warehouse capacity.
    """

    def __init__(self) -> None:
        """
        Initializes the OptimizationAgent.
        """
        pass

    def run_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the inventory optimization task.

        Args:
            task_details: A dictionary containing task parameters.
                Expected keys:
                - 'data_path' (str, optional): Path to the inventory recommendations CSV.
                  Defaults to 'data/processed/inventory_recommendations.csv'.
                - 'output_path' (str, optional): Path to save optimized recommendations CSV.
                  Defaults to 'data/processed/optimized_inventory.csv'.
                - 'service_level_z' (float, optional): Z-score service level multiplier.
                  Defaults to 1.65.

        Returns:
            Dict[str, Any]: Execution results containing status, output path, and summary metrics.
        """
        data_path = task_details.get("data_path", os.path.join("data", "processed", "inventory_recommendations.csv"))
        output_path = task_details.get("output_path", os.path.join("data", "processed", "optimized_inventory.csv"))
        service_level_z = task_details.get("service_level_z", 1.65)

        if not os.path.exists(data_path):
            return {
                "status": "failed",
                "error": f"Recommendations file not found at: {data_path}. Please run InventoryAgent first."
            }

        try:
            # 1. Load inventory recommendations dataset
            df = pd.read_csv(data_path)

            # Check required columns
            required_cols = [
                "Product", "Warehouse", "Supplier", "Inventory_Level", "Holding_Cost",
                "Transportation_Cost", "Cost", "Storage_Capacity", "predicted_demand",
                "Safety_Stock", "Reorder_Point", "EOQ", "Recommendation_Status",
                "Recommended_Order_Quantity", "Reorder_Cost"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    "status": "failed",
                    "error": f"Input dataset is missing required columns: {missing_cols}"
                }

            # 2. Extract supplier profiles dynamically
            supplier_profiles = self._extract_supplier_profiles(df)

            # 3. Calculate demand standard deviation for safety stock calculations
            df["demand_std"] = df.groupby("Product")["predicted_demand"].transform("std")
            df["demand_std"] = df["demand_std"].fillna(df["predicted_demand"] * 0.2)
            df.loc[df["demand_std"] <= 0, "demand_std"] = df["predicted_demand"] * 0.2
            df.loc[df["demand_std"] < 1.0, "demand_std"] = 1.0

            # 4. Perform supplier selection and unconstrained order quantity optimization
            df = self._optimize_supplier_and_quantity(df, supplier_profiles, service_level_z)

            # 5. Enforce warehouse capacity constraints using Lagrangian binary search
            df = self._apply_capacity_constraints(df)

            # 6. Recalculate cost metrics
            df["Optimized_Reorder_Cost"] = (df["Optimized_Order_Quantity"] * df["Optimized_Cost"]).round(2)
            df["Optimized_Stock_Value"] = (df["Inventory_Level"] * df["Optimized_Cost"]).round(2)
            
            # 7. Save results
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Clean temporary columns before saving
            save_df = df.drop(columns=["demand_std"])
            save_df.to_csv(output_path, index=False)

            # 8. Generate summary metrics
            capacity_violations_resolved = int(
                ((df["Recommended_Order_Quantity"] != df["Optimized_Order_Quantity"]) & 
                 (df["Recommendation_Status"] == "REORDER")).sum()
            )
            
            unopt_total_cost = float(df["Reorder_Cost"].sum())
            opt_total_cost = float(df["Optimized_Reorder_Cost"].sum())
            cost_savings = max(0.0, unopt_total_cost - opt_total_cost)

            summary = {
                "total_items_processed": len(df),
                "reorder_items_count": int((df["Recommendation_Status"] == "REORDER").sum()),
                "total_original_reorder_qty": int(df["Recommended_Order_Quantity"].sum()),
                "total_optimized_reorder_qty": int(df["Optimized_Order_Quantity"].sum()),
                "total_original_reorder_cost": unopt_total_cost,
                "total_optimized_reorder_cost": opt_total_cost,
                "estimated_cost_savings": float(round(cost_savings, 2)),
                "capacity_adjustments_count": capacity_violations_resolved,
                "output_path": output_path
            }

            return {
                "status": "success",
                "optimization_summary": summary,
                "output_path": output_path
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"An error occurred during inventory optimization: {str(e)}"
            }

    def _extract_supplier_profiles(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups the dataset to compile available suppliers and logistics parameters per product.
        """
        supplier_profiles: Dict[str, List[Dict[str, Any]]] = {}
        
        # We fill missing supplier-specific values using reasonable defaults
        df_temp = df.copy()
        df_temp["Cost"] = df_temp["Cost"].fillna(50.0)
        df_temp["Lead_Time"] = df_temp["Lead_Time"].fillna(2.0)
        df_temp["Transportation_Cost"] = df_temp["Transportation_Cost"].fillna(50.0)

        for (product, supplier), group in df_temp.groupby(["Product", "Supplier"]):
            cost = float(group["Cost"].mean())
            lead_time = float(group["Lead_Time"].mean())
            trans_cost = float(group["Transportation_Cost"].mean())

            if product not in supplier_profiles:
                supplier_profiles[product] = []

            supplier_profiles[product].append({
                "supplier_name": supplier,
                "cost": cost,
                "lead_time": lead_time,
                "transportation_cost": trans_cost
            })

        return supplier_profiles

    def _optimize_supplier_and_quantity(
        self, df: pd.DataFrame, supplier_profiles: Dict[str, List[Dict[str, Any]]], service_level_z: float
    ) -> pd.DataFrame:
        """
        Chooses the cheapest supplier based on purchase, ordering, and lead-time safety stock holding costs.
        """
        # Create columns for optimized values and explicitly cast to avoid pandas int64 dtype errors on float assignments
        df["Optimized_Supplier"] = df["Supplier"].astype(str)
        df["Optimized_Lead_Time"] = df["Lead_Time"].astype(float)
        df["Optimized_Safety_Stock"] = df["Safety_Stock"].astype(float)
        df["Optimized_Reorder_Point"] = df["Reorder_Point"].astype(float)
        df["Optimized_Cost"] = df["Cost"].astype(float)
        df["Optimized_Transportation_Cost"] = df["Transportation_Cost"].astype(float)
        df["Optimized_Order_Quantity"] = 0
        df["Optimized_Unconstrained_EOQ"] = 0

        for idx, row in df.iterrows():
            product = row["Product"]
            status = row["Recommendation_Status"]
            
            # Get supplier options
            options = supplier_profiles.get(product, [])
            if not options:
                # Fallback if no profile exists
                options = [{
                    "supplier_name": row["Supplier"],
                    "cost": row["Cost"] if pd.notna(row["Cost"]) else 50.0,
                    "lead_time": row["Lead_Time"] if pd.notna(row["Lead_Time"]) else 2.0,
                    "transportation_cost": row["Transportation_Cost"] if pd.notna(row["Transportation_Cost"]) else 50.0
                }]

            best_supplier = options[0]
            min_total_cost = float("inf")
            best_ss = row["Safety_Stock"]
            best_eoq = row["EOQ"]

            # Evaluate each supplier option
            for opt in options:
                lead_time = opt["lead_time"]
                cost = opt["cost"]
                trans_cost = opt["transportation_cost"]
                holding_cost = row["Holding_Cost"] if (pd.notna(row["Holding_Cost"]) and row["Holding_Cost"] > 0) else 5.0
                demand = row["predicted_demand"]
                std_dev = row["demand_std"]

                # 1. Recalculate Safety Stock for supplier lead time
                ss = service_level_z * std_dev * np.sqrt(lead_time)

                # 2. Recalculate EOQ for supplier transportation cost
                eoq = np.sqrt((2 * demand * trans_cost) / holding_cost)
                eoq = max(1.0, np.round(eoq))

                # 3. Calculate total relevant cost
                # Purchase Cost + Ordering/Setup Cost + Average Inventory Holding Cost + Safety Stock Holding Cost
                total_cost = (demand * cost) + (demand / eoq * trans_cost) + holding_cost * (eoq / 2.0 + ss)

                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_supplier = opt
                    best_ss = ss
                    best_eoq = int(eoq)

            # Store optimized supplier characteristics
            df.at[idx, "Optimized_Supplier"] = best_supplier["supplier_name"]
            df.at[idx, "Optimized_Lead_Time"] = best_supplier["lead_time"]
            df.at[idx, "Optimized_Safety_Stock"] = round(best_ss, 2)
            
            demand_val = row["predicted_demand"]
            rop = (demand_val * best_supplier["lead_time"]) + best_ss
            df.at[idx, "Optimized_Reorder_Point"] = round(rop, 2)
            df.at[idx, "Optimized_Cost"] = best_supplier["cost"]
            df.at[idx, "Optimized_Transportation_Cost"] = best_supplier["transportation_cost"]
            df.at[idx, "Optimized_Unconstrained_EOQ"] = best_eoq

            # Only order if Inventory Agent recommended REORDER
            if status == "REORDER":
                df.at[idx, "Optimized_Order_Quantity"] = best_eoq
            else:
                df.at[idx, "Optimized_Order_Quantity"] = 0

        return df

    def _apply_capacity_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces warehouse capacity limits across products using Lagrangian multiplier binary search.
        """
        for warehouse, group in df.groupby("Warehouse"):
            # Get capacity
            capacity = group["Storage_Capacity"].iloc[0]
            if pd.isna(capacity) or capacity <= 0:
                capacity = 10000.0  # Fallback warehouse capacity

            current_stock = group["Inventory_Level"].sum()
            unconstrained_orders = group["Optimized_Order_Quantity"].sum()
            total_projected_stock = current_stock + unconstrained_orders

            # If capacity is violated
            if total_projected_stock > capacity:
                available_capacity = max(0.0, capacity - current_stock)
                
                # Identify items that have active orders in this warehouse
                active_order_mask = (group["Optimized_Order_Quantity"] > 0)
                active_indices = group[active_order_mask].index.tolist()

                if not active_indices:
                    continue

                if available_capacity <= 0:
                    # Warehouse is already full, cancel all active orders
                    df.loc[active_indices, "Optimized_Order_Quantity"] = 0
                    continue

                # Run binary search to find Lagrange multiplier lambda
                low_lambda = 0.0
                high_lambda = 1e6
                best_lambda = 0.0

                for _ in range(50):
                    mid_lambda = (low_lambda + high_lambda) / 2.0
                    sum_q = 0.0

                    for idx in active_indices:
                        d = df.loc[idx, "predicted_demand"]
                        s = df.loc[idx, "Optimized_Transportation_Cost"]
                        h = df.loc[idx, "Holding_Cost"]
                        if pd.isna(h) or h <= 0:
                            h = 5.0

                        q_val = np.sqrt((2 * d * s) / (h + 2 * mid_lambda))
                        sum_q += q_val

                    if sum_q > available_capacity:
                        # We need smaller order quantities, so increase lambda
                        low_lambda = mid_lambda
                        best_lambda = mid_lambda
                    else:
                        # We are within capacity, try smaller lambda to utilize more capacity
                        high_lambda = mid_lambda

                # Apply capacity-constrained quantities
                for idx in active_indices:
                    d = df.loc[idx, "predicted_demand"]
                    s = df.loc[idx, "Optimized_Transportation_Cost"]
                    h = df.loc[idx, "Holding_Cost"]
                    if pd.isna(h) or h <= 0:
                        h = 5.0

                    q_val = np.sqrt((2 * d * s) / (h + 2 * best_lambda))
                    # Clamp to at least 1 unit and at most the unconstrained EOQ
                    unconstrained_eoq = df.loc[idx, "Optimized_Unconstrained_EOQ"]
                    df.loc[idx, "Optimized_Order_Quantity"] = int(np.clip(np.round(q_val), 1, unconstrained_eoq))

                # Fine-tune rounding errors to ensure we strictly respect warehouse capacity
                current_orders_sum = df.loc[active_indices, "Optimized_Order_Quantity"].sum()
                if current_orders_sum > available_capacity:
                    # Sort active orders by quantity descending to scale down largest first
                    sorted_active = sorted(active_indices, key=lambda i: df.loc[i, "Optimized_Order_Quantity"], reverse=True)
                    for idx in sorted_active:
                        if current_orders_sum <= available_capacity:
                            break
                        val = df.loc[idx, "Optimized_Order_Quantity"]
                        if val > 1:
                            df.loc[idx, "Optimized_Order_Quantity"] = val - 1
                            current_orders_sum -= 1

        return df
