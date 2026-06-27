"""
A2A Server Module.
Implements a generic HTTP server wrapper to host any agent as a network service.
"""

import os
import sys
import json
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

# Ensure project root is in the path to allow relative imports of modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the agents
from agents.data_agent.agent import DataAgent
from agents.forecasting_agent.agent import ForecastingAgent
from agents.host_agent.agent import HostAgent, AgentRegistry
from agents.inventory_agent.agent import InventoryAgent
from agents.optimization_agent.agent import OptimizationAgent


class A2AAgentServerHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler that routes JSON-RPC capability calls to
    the wrapped agent instance.
    """
    agent_name: str = ""
    agent_instance: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        """
        Custom logging prepended with the server's agent name.
        """
        sys.stderr.write(f"[{self.agent_name.upper()} Server] " + (format % args) + "\n")

    def do_POST(self) -> None:
        """
        Handles incoming POST requests to run agent capabilities.
        """
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

        # Route the action based on the agent wrapped by this handler
        response = self.handle_action(action, payload)
        
        # Return success/failed payload with HTTP 200
        self.send_json_response(response, 200)

    def send_json_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """
        Encapsulates sending a JSON response.
        """
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def handle_action(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps standard message action names to the wrapped agent's core functionalities.
        """
        agent = self.agent_instance
        agent_name = self.agent_name

        if agent_name == "data_agent":
            if action == "clean_data":
                raw_path = payload.get("raw_path")
                cleaned_path = payload.get("cleaned_path")
                if not raw_path or not cleaned_path:
                    return {"status": "failed", "error": "Missing 'raw_path' or 'cleaned_path' in payload."}
                
                try:
                    df = agent.read_csv(raw_path)
                    missing_report = agent.check_missing_values()
                    duplicates_removed, df_cleaned = agent.remove_duplicates()
                    agent.save_cleaned_csv(cleaned_path)
                    
                    return {
                        "status": "success",
                        "initial_row_count": len(df) + duplicates_removed,
                        "duplicates_removed": duplicates_removed,
                        "cleaned_row_count": len(df_cleaned),
                        "missing_values_report": missing_report,
                        "output_path": cleaned_path
                    }
                except Exception as e:
                    return {"status": "failed", "error": f"DataAgent cleaning error: {str(e)}"}
            else:
                return {"status": "failed", "error": f"Unknown action '{action}' for data_agent."}

        elif agent_name == "forecasting_agent":
            if action == "forecast_demand":
                try:
                    return agent.run_task(payload)
                except Exception as e:
                    return {"status": "failed", "error": f"ForecastingAgent training error: {str(e)}"}
            else:
                return {"status": "failed", "error": f"Unknown action '{action}' for forecasting_agent."}

        elif agent_name == "host_agent":
            if action == "coordinate_pipeline":
                try:
                    return agent.run_task(payload)
                except Exception as e:
                    return {"status": "failed", "error": f"HostAgent coordination error: {str(e)}"}
            else:
                return {"status": "failed", "error": f"Unknown action '{action}' for host_agent."}

        elif agent_name == "inventory_agent":
            if action == "calculate_inventory_recommendations":
                try:
                    return agent.run_task(payload)
                except Exception as e:
                    return {"status": "failed", "error": f"InventoryAgent recommendations error: {str(e)}"}
            else:
                return {"status": "failed", "error": f"Unknown action '{action}' for inventory_agent."}

        elif agent_name == "optimization_agent":
            if action == "optimize_inventory":
                try:
                    return agent.run_task(payload)
                except Exception as e:
                    return {"status": "failed", "error": f"OptimizationAgent error: {str(e)}"}
            else:
                return {"status": "failed", "error": f"Unknown action '{action}' for optimization_agent."}

        else:
            return {"status": "failed", "error": f"Unrecognized agent registry context: '{agent_name}'."}


def main() -> None:
    """
    CLI Entrypoint to run a server instance.
    """
    parser = argparse.ArgumentParser(description="A2A HTTP Server Wrapper for Agents")
    parser.add_argument(
        "--agent",
        choices=["host_agent", "data_agent", "forecasting_agent", "inventory_agent", "optimization_agent"],
        required=True,
        help="Specify which agent class to run as an HTTP service."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Specify port to listen on. Defaults to values in common.utils."
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host interface to bind the socket to. (default: localhost)"
    )

    args = parser.parse_args()

    # Load port from standard registry if not overridden
    from common.utils import AGENT_PORTS
    port = args.port or AGENT_PORTS[args.agent]

    # Instantiate the agent class
    if args.agent == "host_agent":
        # Pass a registry with use_network=True to coordinate over network calls
        registry = AgentRegistry(use_network=True)
        agent_instance = HostAgent(registry=registry)
    elif args.agent == "data_agent":
        agent_instance = DataAgent()
    elif args.agent == "forecasting_agent":
        agent_instance = ForecastingAgent()
    elif args.agent == "inventory_agent":
        agent_instance = InventoryAgent()
    elif args.agent == "optimization_agent":
        agent_instance = OptimizationAgent()
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    # Bind variables to handler class
    A2AAgentServerHandler.agent_name = args.agent
    A2AAgentServerHandler.agent_instance = agent_instance

    server = HTTPServer((args.host, port), A2AAgentServerHandler)
    print(f"=== Starting A2A HTTP Server for '{args.agent}' ===")
    print(f"Endpoint: http://{args.host}:{port}/")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down A2A server gracefully...")
        server.server_close()


if __name__ == "__main__":
    main()
