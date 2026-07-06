import os
import sys
from typing import Optional

# Ensure project root is in path for absolute/relative imports compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp.server.fastmcp import FastMCP
from mcp_server.tools import (
    run_inventory_pipeline_impl,
    get_forecast_metrics_impl,
    get_inventory_status_impl,
    get_business_report_impl
)

# Initialize FastMCP Server
mcp = FastMCP("inventory_optimization_server")


@mcp.tool()
def run_inventory_pipeline(
    raw_data_path: Optional[str] = None,
    cleaned_data_path: str = "data/processed/cleaned_data.csv",
    inventory_output_path: str = "data/processed/inventory_recommendations.csv",
    service_level_z: float = 1.65
) -> str:
    """Runs the primary Host Agent pipeline which cleans data, forecasts demand, and calculates inventory target levels.

    Args:
        raw_data_path: Optional path to the raw input CSV. If not provided, it resolves to the latest CSV in the data/raw/ folder.
        cleaned_data_path: Path where the cleaned CSV data should be saved (relative to project root).
        inventory_output_path: Path where replenishment recommendations CSV should be saved (relative to project root).
        service_level_z: The Z-score multiplier representing target service level (e.g., 1.65 for 95% service level).
    """
    result = run_inventory_pipeline_impl(
        raw_data_path=raw_data_path,
        cleaned_data_path=cleaned_data_path,
        inventory_output_path=inventory_output_path,
        service_level_z=service_level_z
    )
    import json
    return json.dumps(result, indent=2)


@mcp.tool()
def get_forecast_metrics() -> str:
    """Reads and returns the forecasting model evaluation metrics from 'pipeline_metrics.json'."""
    result = get_forecast_metrics_impl()
    import json
    return json.dumps(result, indent=2)


@mcp.tool()
def get_inventory_status(
    product: Optional[str] = None,
    warehouse: Optional[str] = None,
    status: Optional[str] = None
) -> str:
    """Reads and returns optimized inventory status records from 'optimized_inventory.csv', with optional filters.

    Args:
        product: Optional filter for a specific product name.
        warehouse: Optional filter for a specific warehouse name.
        status: Optional filter for recommendation status (e.g. 'REORDER', 'OK', 'OVERSTOCK').
    """
    result = get_inventory_status_impl(
        product=product,
        warehouse=warehouse,
        status=status
    )
    import json
    return json.dumps(result, indent=2)


@mcp.tool()
def get_business_report() -> str:
    """Reads and returns the compiled business report markdown file 'inventory_business_report.md'."""
    result = get_business_report_impl()
    import json
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    print("Starting Inventory Optimization MCP server...")
    mcp.run(transport="stdio")
