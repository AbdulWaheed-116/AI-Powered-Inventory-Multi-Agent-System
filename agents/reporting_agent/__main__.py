"""
Reporting Agent CLI Entry Point.
Allows running the business reporting demo directly or starting an HTTP server for A2A communication.
"""

import os
import sys
import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.reporting_agent.agent import ReportingAgent
from agents.reporting_agent.task_manager import TaskManager


class ReportingAgentServerHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler routing A2A capability requests to ReportingAgent.
    """
    agent_instance = ReportingAgent()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[REPORTING Agent Server] " + (format % args) + "\n")

    def do_POST(self) -> None:
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found. Use '/'")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            request_body = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self.send_json_response({"status": "failed", "error": f"Invalid JSON payload: {str(e)}"}, 400)
            return

        action = request_body.get("action")
        payload = request_body.get("payload", {})

        if not action:
            self.send_json_response({"status": "failed", "error": "Missing key 'action' in request body."}, 400)
            return

        if action == "generate_report":
            try:
                response = self.agent_instance.run_task(payload)
                self.send_json_response(response, 200)
            except Exception as e:
                self.send_json_response({"status": "failed", "error": f"ReportingAgent error: {str(e)}"}, 500)
        else:
            self.send_json_response({"status": "failed", "error": f"Unknown action '{action}' for reporting_agent."}, 400)

    def send_json_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_demo() -> None:
    """
    Runs a complete demonstration of the ReportingAgent:
    1. Loads the Optimization Agent outputs.
    2. Generates the business metrics report.
    3. Prints a summary of KPIs on the console.
    """
    print("=== Reporting Agent Business Analysis Demonstration ===")

    optimized_path = os.path.join("data", "processed", "optimized_inventory.csv")
    report_output_path = os.path.join("data", "processed", "inventory_business_report.md")

    if not os.path.exists(optimized_path):
        print(f"\n[Error] Optimized inventory dataset not found at: {optimized_path}")
        print("Please run the OptimizationAgent pipeline first to generate optimized inventory data.")
        return

    agent = ReportingAgent()
    manager = TaskManager(agent=agent)

    print("\n1. Initializing business reporting task...")
    task_id = manager.create_task({
        "data_path": optimized_path,
        "output_path": report_output_path
    })
    print(f"Task created with ID: {task_id}")

    # Inspect initial task status
    task = manager.get_task(task_id)
    print(f"Task status before execution: {task['status']}")

    print("\n2. Executing report generation task...")
    completed_task = manager.execute_task(task_id)
    print(f"Task status after execution: {completed_task['status']}")

    if completed_task["status"] == "COMPLETED":
        result = completed_task["result"]
        summary = result["report_summary"]

        print("\n=== Business Report Summary ===")
        print(f" - Total Sourcing Cost Saved         : ${summary['total_cost_savings']:,.2f} ({summary['pct_cost_savings']:.1f}%)")
        print(f" - Original Replenishment Cost       : ${summary['total_original_reorder_cost']:,.2f}")
        print(f" - Optimized Replenishment Cost      : ${summary['total_optimized_reorder_cost']:,.2f}")
        print(f" - Current Stock Asset Value         : ${summary['current_stock_value']:,.2f}")
        print(f" - Post-Order Stock Asset Value      : ${summary['optimized_stock_value']:,.2f}")
        print(f" - Total Unique Warehouses Evaluated : {summary['unique_warehouses']}")
        print(f" - Total Unique Products Processed   : {summary['unique_products']}")
        print(f" - Items Reordered (Original)        : {summary['items_reordered_original']}")
        print(f" - Items Reordered (Optimized)       : {summary['items_reordered_optimized']}")
        print(f" - Average Lead Time (Orig vs Opt)   : {summary['avg_original_lead_time']} vs {summary['avg_optimized_lead_time']} days")
        print(f" - Average Safety Stock (Orig vs Opt): {summary['avg_original_safety_stock']} vs {summary['avg_optimized_safety_stock']} units")
        print(f" - Business Report File Generated To : {result['output_path']}")

        print("\nReport generation completed successfully!")
    else:
        print(f"\n[Error] Task execution failed: {completed_task['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporting Agent Entrypoint")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run Reporting Agent as a standalone HTTP server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8005,
        help="Port to run the A2A HTTP Server on (default: 8005)."
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host interface for server to bind (default: localhost)."
    )

    args = parser.parse_args()

    if args.server:
        server = HTTPServer((args.host, args.port), ReportingAgentServerHandler)
        print(f"=== Starting Standalone A2A HTTP Server for 'reporting_agent' ===")
        print(f"Endpoint: http://{args.host}:{args.port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down A2A reporting agent server gracefully...")
            server.server_close()
    else:
        run_demo()


if __name__ == "__main__":
    main()
