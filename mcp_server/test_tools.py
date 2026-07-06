import os
import sys

# Ensure project root is in path for absolute/relative imports compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp_server.tools import (
    run_inventory_pipeline_impl,
    get_forecast_metrics_impl,
    get_inventory_status_impl,
    get_business_report_impl
)


def test_get_forecast_metrics():
    print("\n--- Testing get_forecast_metrics ---")
    result = get_forecast_metrics_impl()
    print(f"Status: {result.get('status')}")
    if result.get("status") == "success":
        print("Metrics retrieved:")
        for k, v in result.get("metrics", {}).items():
            print(f"  {k}: {v}")
    else:
        print(f"Error: {result.get('error')}")


def test_get_inventory_status():
    print("\n--- Testing get_inventory_status (unfiltered) ---")
    result = get_inventory_status_impl()
    print(f"Status: {result.get('status')}")
    if result.get("status") == "success":
        summary = result.get("summary", {})
        print("Summary Stats:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"Total matching records: {result.get('records_count')}")
        print(f"First record details: {result.get('records')[0] if result.get('records') else 'None'}")
    else:
        print(f"Error: {result.get('error')}")

    # Test filtering
    print("\n--- Testing get_inventory_status (filtered by Product='Vitamin C' and Status='REORDER') ---")
    result_filtered = get_inventory_status_impl(product="Vitamin C", status="REORDER")
    print(f"Status: {result_filtered.get('status')}")
    if result_filtered.get("status") == "success":
        print(f"Filtered records count: {result_filtered.get('records_count')}")
        if result_filtered.get("records"):
            print("Filtered records sample:")
            for record in result_filtered.get("records")[:2]:
                print(f"  Product: {record.get('Product')}, Warehouse: {record.get('Warehouse')}, Status: {record.get('Recommendation_Status')}, Optimized Order Qty: {record.get('Optimized_Order_Quantity')}")
    else:
        print(f"Error: {result_filtered.get('error')}")


def test_get_business_report():
    print("\n--- Testing get_business_report ---")
    result = get_business_report_impl()
    print(f"Status: {result.get('status')}")
    if result.get("status") == "success":
        content = result.get("report_content", "")
        # Print the first few lines of the report
        lines = content.splitlines()[:10]
        print("Report Preview (First 10 lines):")
        for line in lines:
            print(f"  {line}")
    else:
        print(f"Error: {result.get('error')}")


def test_run_inventory_pipeline():
    print("\n--- Testing run_inventory_pipeline ---")
    print("Executing inventory pipeline via HostAgent...")
    # Using small dataset or default setup
    result = run_inventory_pipeline_impl(service_level_z=1.65)
    print(f"Status: {result.get('status')}")
    if result.get("status") == "success":
        print("Pipeline execution succeeded.")
        print("Data Agent Rows Cleaned:", result.get("data_agent_results", {}).get("cleaned_row_count"))
        print("Forecasting Mean Demand:", result.get("forecasting_agent_results", {}).get("predictions_summary", {}).get("mean"))
        print("Inventory Recommendations Row Count:", len(result.get("inventory_agent_results", {}).get("recommendations", [])) if result.get("inventory_agent_results") else "N/A")
    else:
        print(f"Pipeline execution failed: {result.get('error')}")


if __name__ == "__main__":
    print("Starting independent tool tests...")
    
    # 1. Test reading tools first
    test_get_forecast_metrics()
    test_get_inventory_status()
    test_get_business_report()
    
    # 2. Test pipeline execution tool
    test_run_inventory_pipeline()
    
    # 3. Test reading tools again to verify fresh cache updates
    print("\nRe-testing metrics to verify pipeline update...")
    test_get_forecast_metrics()
    
    print("\nAll independent tool tests completed.")
