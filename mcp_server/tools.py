import os
import sys
import json
import datetime
from typing import Any, Dict, Optional
import pandas as pd

# Ensure project root is in path for absolute/relative imports compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.host_agent.agent import HostAgent

# File Paths
RAW_DIR = os.path.join(project_root, "data", "raw")
PROCESSED_DIR = os.path.join(project_root, "data", "processed")
METRICS_PATH = os.path.join(PROCESSED_DIR, "pipeline_metrics.json")
OPTIMIZED_PATH = os.path.join(PROCESSED_DIR, "optimized_inventory.csv")
REPORT_PATH = os.path.join(PROCESSED_DIR, "inventory_business_report.md")


def run_inventory_pipeline_impl(
    raw_data_path: Optional[str] = None,
    cleaned_data_path: str = "data/processed/cleaned_data.csv",
    inventory_output_path: str = "data/processed/inventory_recommendations.csv",
    service_level_z: float = 1.65
) -> Dict[str, Any]:
    """Runs the primary Host Agent pipeline which cleans data, forecasts demand, and calculates inventory target levels.

    Args:
        raw_data_path: Optional path to the raw input CSV. If not provided, it resolves to the latest CSV in the data/raw/ folder.
        cleaned_data_path: Path where the cleaned CSV data should be saved (relative to project root).
        inventory_output_path: Path where replenishment recommendations CSV should be saved (relative to project root).
        service_level_z: The Z-score multiplier representing target service level (e.g., 1.65 for 95% service level).

    Returns:
        A dictionary containing the status of the execution ('success' or 'failed') and details of each agent's results.
    """
    try:
        # Dynamically resolve raw data path if not provided
        if not raw_data_path:
            if os.path.exists(RAW_DIR):
                csv_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
                if csv_files:
                    csv_files.sort(key=os.path.getmtime, reverse=True)
                    raw_data_path = csv_files[0]

        if not raw_data_path or not os.path.exists(raw_data_path):
            return {
                "status": "failed",
                "error": "Required raw dataset is missing. Please place a CSV file in data/raw/ first."
            }

        abs_raw = os.path.abspath(raw_data_path)
        abs_clean = os.path.abspath(os.path.join(project_root, cleaned_data_path))
        abs_inv = os.path.abspath(os.path.join(project_root, inventory_output_path))

        os.makedirs(os.path.dirname(abs_clean), exist_ok=True)
        os.makedirs(os.path.dirname(abs_inv), exist_ok=True)

        # Initialize and run HostAgent
        agent = HostAgent()
        result = agent.run_task({
            "raw_data_path": abs_raw,
            "cleaned_data_path": abs_clean,
            "inventory_output_path": abs_inv,
            "service_level_z": service_level_z
        })

        # Cache metrics on success to maintain alignment with Streamlit App
        if result.get("status") == "success":
            forecasting_metrics = result.get("forecasting_agent_results", {}).get("metrics", {})
            if forecasting_metrics:
                pipeline_meta = {
                    "r2": forecasting_metrics.get("r2", 0.0),
                    "rmse": forecasting_metrics.get("rmse", 0.0),
                    "mae": forecasting_metrics.get("mae", 0.0),
                    "mse": forecasting_metrics.get("mse", 0.0),
                    "last_updated": datetime.datetime.now().isoformat()
                }
                os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
                with open(METRICS_PATH, "w", encoding="utf-8") as f:
                    json.dump(pipeline_meta, f, indent=2)

        return result
    except Exception as e:
        return {"status": "failed", "error": f"Error running Host Agent: {str(e)}"}


def get_forecast_metrics_impl() -> Dict[str, Any]:
    """Reads and returns the forecasting model evaluation metrics from 'pipeline_metrics.json'.

    Returns:
        A dictionary containing model metrics (R2, RMSE, MAE, MSE) and the last updated timestamp.
    """
    if not os.path.exists(METRICS_PATH):
        return {
            "status": "failed",
            "error": f"Forecast metrics file not found at '{METRICS_PATH}'. Please run the inventory pipeline first."
        }
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Error reading metrics file: {str(e)}"
        }


def get_inventory_status_impl(
    product: Optional[str] = None,
    warehouse: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Reads and filters optimized inventory status records from 'optimized_inventory.csv'.

    Args:
        product: Optional filter for a specific product name.
        warehouse: Optional filter for a specific warehouse name.
        status: Optional filter for recommendation status (e.g. 'REORDER', 'OK', 'OVERSTOCK').

    Returns:
        A dictionary containing overall network metrics summary and a list of matching inventory records.
    """
    if not os.path.exists(OPTIMIZED_PATH):
        return {
            "status": "failed",
            "error": f"Optimized inventory file not found at '{OPTIMIZED_PATH}'. Please run the pipeline first."
        }
    try:
        df = pd.read_csv(OPTIMIZED_PATH)

        # High level summary
        total_products = int(df["Product"].nunique())
        total_records = len(df)
        reorder_alerts = int((df["Recommendation_Status"] == "REORDER").sum())
        total_optimized_qty = int(df["Optimized_Order_Quantity"].sum())
        original_cost = float(df["Reorder_Cost"].sum())
        optimized_cost = float(df["Optimized_Reorder_Cost"].sum())
        cost_savings = max(0.0, original_cost - optimized_cost)

        summary = {
            "total_products": total_products,
            "total_records": total_records,
            "reorder_alerts": reorder_alerts,
            "total_optimized_quantity": total_optimized_qty,
            "original_cost": original_cost,
            "optimized_cost": optimized_cost,
            "cost_savings": cost_savings
        }

        # Apply optional filters
        filtered_df = df.copy()
        if product:
            filtered_df = filtered_df[filtered_df["Product"].str.contains(product, case=False, na=False)]
        if warehouse:
            filtered_df = filtered_df[filtered_df["Warehouse"].str.contains(warehouse, case=False, na=False)]
        if status:
            filtered_df = filtered_df[filtered_df["Recommendation_Status"].str.upper() == status.upper()]

        # Convert records to list of dicts (limit to first 100 to avoid token limits if too large)
        records = filtered_df.head(100).to_dict(orient="records")
        truncated = len(filtered_df) > 100

        return {
            "status": "success",
            "summary": summary,
            "records_count": len(filtered_df),
            "records_truncated": truncated,
            "records": records
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Error reading optimized inventory: {str(e)}"
        }


def get_business_report_impl() -> Dict[str, Any]:
    """Reads and returns the compiled business report markdown file 'inventory_business_report.md'.

    Returns:
        A dictionary containing the report content as a string.
    """
    if not os.path.exists(REPORT_PATH):
        return {
            "status": "failed",
            "error": f"Business report not found at '{REPORT_PATH}'. Please run the pipeline first."
        }
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
        return {
            "status": "success",
            "report_content": report_content
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Error reading business report: {str(e)}"
        }
