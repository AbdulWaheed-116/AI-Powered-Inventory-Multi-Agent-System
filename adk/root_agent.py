"""
Root Agent Module.
Defines the orchestrator agent and register tools that wrap calls to core agents.
"""

import os
import sys
from typing import Any, Dict

# Ensure project root is in path for absolute/relative imports compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from google.adk import Agent
from google.adk.tools.function_tool import ToolContext
from agents.host_agent.agent import HostAgent
from agents.optimization_agent.agent import OptimizationAgent
from agents.reporting_agent.agent import ReportingAgent


# Define wrapper tools for the existing pipeline agents

def run_host_agent_pipeline(
    ctx: ToolContext,
    raw_data_path: str = "",
    cleaned_data_path: str = "data/processed/cleaned_data.csv",
    inventory_output_path: str = "data/processed/inventory_recommendations.csv",
    service_level_z: float = 1.65
) -> Dict[str, Any]:
    """Runs the primary Host Agent pipeline which cleans data, forecasts demand, and calculates inventory target levels.

    Args:
        raw_data_path: Path to the raw input CSV. If empty, the active dataset is resolved dynamically.
        cleaned_data_path: Path where the cleaned CSV data should be saved.
        inventory_output_path: Path where the replenishment recommendations CSV should be saved.
        service_level_z: The Z-score multiplier representing target service level (e.g., 1.65 for 95% service level).

    Returns:
        A dictionary containing the status of the execution ('success' or 'failed') and details of each agent's results.
    """
    try:
        agent = HostAgent()
        
        # Dynamically resolve raw data path if not provided
        if not raw_data_path:
            # 1. Try to get it from tool context state passed from Streamlit
            if ctx and ctx.state:
                raw_data_path = ctx.state.get("raw_data_path")
            
            # 2. Fallback: Search the data/raw/ directory for any CSV files
            if not raw_data_path:
                raw_dir = os.path.abspath(os.path.join(project_root, "data", "raw"))
                if os.path.exists(raw_dir):
                    csv_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
                    if csv_files:
                        csv_files.sort(key=os.path.getmtime, reverse=True)
                        raw_data_path = csv_files[0]
                        
        if not raw_data_path or not os.path.exists(raw_data_path):
            return {
                "status": "failed", 
                "error": "Required raw dataset is missing. Please upload a CSV file using the Streamlit pipeline first."
            }

        abs_raw = os.path.abspath(raw_data_path)
        abs_clean = os.path.abspath(os.path.join(project_root, cleaned_data_path))
        abs_inv = os.path.abspath(os.path.join(project_root, inventory_output_path))
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(abs_clean), exist_ok=True)
        os.makedirs(os.path.dirname(abs_inv), exist_ok=True)
        
        result = agent.run_task({
            "raw_data_path": abs_raw,
            "cleaned_data_path": abs_clean,
            "inventory_output_path": abs_inv,
            "service_level_z": service_level_z
        })
        return result
    except Exception as e:
        return {"status": "failed", "error": f"Error running Host Agent: {str(e)}"}


def run_optimization_agent_pipeline(
    data_path: str = "data/processed/inventory_recommendations.csv",
    output_path: str = "data/processed/optimized_inventory.csv",
    service_level_z: float = 1.65
) -> Dict[str, Any]:
    """Runs the Optimization Agent on the replenishment recommendations to select suppliers and enforce warehouse capacity constraints.

    Args:
        data_path: Path to the inventory recommendations CSV (output of the Host Agent).
        output_path: Path where the final optimized inventory CSV should be saved.
        service_level_z: The Z-score multiplier representing target service level.

    Returns:
        A dictionary containing the optimization metrics, cost savings, and capacity mitigations summary.
    """
    try:
        agent = OptimizationAgent()
        abs_data = os.path.abspath(os.path.join(project_root, data_path))
        abs_out = os.path.abspath(os.path.join(project_root, output_path))
        
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        
        result = agent.run_task({
            "data_path": abs_data,
            "output_path": abs_out,
            "service_level_z": service_level_z
        })
        return result
    except Exception as e:
        return {"status": "failed", "error": f"Error running Optimization Agent: {str(e)}"}


def run_reporting_agent_pipeline(
    data_path: str = "data/processed/optimized_inventory.csv",
    output_path: str = "data/processed/inventory_business_report.md"
) -> Dict[str, Any]:
    """Runs the Reporting Agent on the optimized inventory data to calculate KPIs and compile the markdown business report.

    Args:
        data_path: Path to the optimized inventory recommendations CSV (output of the Optimization Agent).
        output_path: Path where the compiled business report markdown file should be saved.

    Returns:
        A dictionary containing the report summary KPIs and the output file path.
    """
    try:
        agent = ReportingAgent()
        abs_data = os.path.abspath(os.path.join(project_root, data_path))
        abs_out = os.path.abspath(os.path.join(project_root, output_path))
        
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        
        result = agent.run_task({
            "data_path": abs_data,
            "output_path": abs_out
        })
        return result
    except Exception as e:
        return {"status": "failed", "error": f"Error running Reporting Agent: {str(e)}"}


def read_inventory_business_report(
    ctx: ToolContext,
    report_path: str = "data/processed/inventory_business_report.md"
) -> str:
    """Reads the generated business report and returns a comprehensive summary of all processed results, including optimization metrics and stock level recommendations.

    Args:
        report_path: Path to the markdown business report file.

    Returns:
        The content of the markdown business report, forecasting evaluation metrics, and a summary of optimized inventory records.
    """
    try:
        abs_report = os.path.abspath(os.path.join(project_root, report_path))
        abs_metrics = os.path.abspath(os.path.join(project_root, "data", "processed", "pipeline_metrics.json"))
        abs_opt = os.path.abspath(os.path.join(project_root, "data", "processed", "optimized_inventory.csv"))

        if not os.path.exists(abs_report):
            return "Error: Business report not found on disk. Please run the pipeline first."

        # 1. Read the business report markdown
        with open(abs_report, "r", encoding="utf-8") as f:
            report_content = f.read()

        # 2. Read the pipeline metrics JSON
        metrics_content = "No metrics cache found."
        if os.path.exists(abs_metrics):
            with open(abs_metrics, "r", encoding="utf-8") as f:
                metrics_content = f.read()

        # 3. Read and summarize the optimized inventory CSV
        opt_summary = "No optimized inventory file found."
        if os.path.exists(abs_opt):
            import pandas as pd
            df_opt = pd.read_csv(abs_opt)
            
            # Formulate a condensed summary of active reorders
            opt_summary = (
                f"Total Unique Products: {df_opt['Product'].nunique()}\n"
                f"Active Warehouses: {df_opt['Warehouse'].unique().tolist()}\n"
                f"Total Reorder Alerts Count: {int((df_opt['Recommendation_Status'] == 'REORDER').sum())}\n"
                f"Total Optimized Order Quantity: {int(df_opt['Optimized_Order_Quantity'].sum())}\n"
                f"Total Optimized Reorder Cost: ${df_opt['Optimized_Reorder_Cost'].sum():,.2f}\n"
                f"Active Replenishment Orders Details:\n"
            )
            
            reorders_df = df_opt[df_opt["Optimized_Order_Quantity"] > 0]
            if not reorders_df.empty:
                opt_summary += reorders_df[["Product", "Warehouse", "Optimized_Supplier", "Optimized_Order_Quantity", "Optimized_Reorder_Cost"]].to_string(index=False)
            else:
                opt_summary += "No new orders are currently recommended."

        return (
            f"--- EXECUTIVE BUSINESS REPORT ---\n{report_content}\n\n"
            f"--- PIPELINE METRICS ---\n{metrics_content}\n\n"
            f"--- OPTIMIZED INVENTORY SUMMARY ---\n{opt_summary}"
        )
    except Exception as e:
        return f"Error reading report and processed results: {str(e)}"


# Instantiate the Root Agent with instruction and tools list
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Root Agent (Orchestrator) for the Multi-Agent Inventory Optimization System.\n"
        "Your task is to help the user analyze and run the inventory pipeline.\n\n"
        "Execution Rules:\n"
        "1. Check if the processed results already exist on disk (you can assume they exist if you can successfully read them using `read_inventory_business_report`).\n"
        "2. If processed results exist, do NOT run the pipeline (i.e. do NOT call `run_host_agent_pipeline`, `run_optimization_agent_pipeline`, or `run_reporting_agent_pipeline`). Instead, use `read_inventory_business_report` to answer the user's questions about KPIs, safety stocks, reorders, suppliers, or warehouse caps.\n"
        "3. Only run the pipeline (calling all three tools: `run_host_agent_pipeline` first, then `run_optimization_agent_pipeline`, and finally `run_reporting_agent_pipeline`) in the following cases:\n"
        "   - The user explicitly asks for a 'rerun', 'refresh', 're-execute', or 'start over'.\n"
        "   - The processed results do not exist on disk (calling `read_inventory_business_report` returns an error saying they do not exist).\n"
        "4. Always answer questions based on the latest report contents. Keep responses professional, data-driven, and business-focused."
    ),
    tools=[
        run_host_agent_pipeline,
        run_optimization_agent_pipeline,
        run_reporting_agent_pipeline,
        read_inventory_business_report
    ]
)
