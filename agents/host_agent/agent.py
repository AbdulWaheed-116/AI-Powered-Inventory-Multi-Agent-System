"""
Host Agent Module.
Contains the HostAgent class responsible for orchestrating sub-agents
and the AgentRegistry class that facilitates Agent-to-Agent (A2A) communication.
"""

import os
from typing import Any, Dict, Optional
from agents.data_agent.agent import DataAgent
from agents.forecasting_agent.agent import ForecastingAgent
from agents.inventory_agent.agent import InventoryAgent


class AgentRegistry:
    """
    A central registry managing agent capabilities and enabling Agent-to-Agent (A2A)
    communication. Supports payload-based message passing between registered agents.
    """

    def __init__(self, use_network: bool = False) -> None:
        """
        Initializes the AgentRegistry map. Allows specifying if network calls should be used.
        """
        self._agents: Dict[str, Any] = {}
        # Enable network mode if parameter is True or via environment variable
        env_a2a = os.environ.get("A2A_MODE", "0") == "1" or os.environ.get("A2A_ENABLED", "").lower() == "true"
        self.use_network = use_network or env_a2a
        
        if self.use_network:
            try:
                from common.a2a_client import A2AClient
                self.client = A2AClient()
            except ImportError:
                self.client = None
        else:
            self.client = None

    def register(self, name: str, agent_instance: Any) -> None:
        """
        Registers an agent instance by name.
        """
        self._agents[name] = agent_instance

    def get_agent(self, name: str) -> Any:
        """
        Retrieves the agent instance by name.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered in the system registry.")
        return self._agents[name]

    def call_agent(self, agent_name: str, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Foundation for A2A communication. Sends a message (payload) requesting a specific
        action from a target agent, executing it and returning the output payload.
        
        If network mode is enabled, dispatches the call over HTTP via A2AClient.
        Otherwise, routes to the local registered instance in-memory.
        """
        if self.use_network and self.client is not None:
            print(f"[A2A Client] Routing request to '{agent_name}' ({action}) over network...")
            return self.client.call_agent(agent_name, action, payload)

        agent = self.get_agent(agent_name)

        if agent_name == "data_agent":
            return self._dispatch_data_agent(agent, action, payload)
        elif agent_name == "forecasting_agent":
            return self._dispatch_forecasting_agent(agent, action, payload)
        elif agent_name == "inventory_agent":
            return self._dispatch_inventory_agent(agent, action, payload)
        else:
            raise NotImplementedError(
                f"A2A dispatch mechanism not implemented for agent: '{agent_name}'"
            )

    def _dispatch_data_agent(self, agent: DataAgent, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapts the DataAgent's methods into a unified message-passing format.
        """
        if action == "clean_data":
            raw_path = payload.get("raw_path")
            cleaned_path = payload.get("cleaned_path")

            if not raw_path:
                return {"status": "failed", "error": "Missing key 'raw_path' in payload."}
            if not cleaned_path:
                return {"status": "failed", "error": "Missing key 'cleaned_path' in payload."}

            try:
                # 1. Load CSV
                df = agent.read_csv(raw_path)
                
                # 2. Check for missing values before cleaning
                missing_report = agent.check_missing_values()

                # 3. Clean dataset (imputes missing, removes duplicates)
                duplicates_removed, df_cleaned = agent.remove_duplicates()

                # 4. Save to processed directory
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
                return {
                    "status": "failed",
                    "error": f"DataAgent error: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "error": f"Action '{action}' is not supported by DataAgent."
            }

    def _dispatch_forecasting_agent(
        self, agent: ForecastingAgent, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapts the ForecastingAgent's methods into a unified message-passing format.
        """
        if action == "forecast_demand":
            try:
                # The ForecastingAgent already expects a task payload structure.
                response = agent.run_task(payload)
                return response
            except Exception as e:
                return {
                    "status": "failed",
                    "error": f"ForecastingAgent error: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "error": f"Action '{action}' is not supported by ForecastingAgent."
            }

    def _dispatch_inventory_agent(
        self, agent: InventoryAgent, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapts the InventoryAgent's methods into a unified message-passing format.
        """
        if action == "calculate_inventory_recommendations":
            try:
                return agent.run_task(payload)
            except Exception as e:
                return {
                    "status": "failed",
                    "error": f"InventoryAgent error: {str(e)}"
                }
        else:
            return {
                "status": "failed",
                "error": f"Action '{action}' is not supported by InventoryAgent."
            }


class HostAgent:
    """
    Host Agent responsible for orchestrating the overall inventory planning pipeline.
    Coordinates sub-agents via the central AgentRegistry using A2A message patterns.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None) -> None:
        """
        Initializes HostAgent. Uses a provided registry or registers default sub-agents.
        """
        self.registry = registry or AgentRegistry()
        
        # Self-register default agents
        self.registry.register("data_agent", DataAgent())
        self.registry.register("forecasting_agent", ForecastingAgent())
        self.registry.register("inventory_agent", InventoryAgent())

    def run_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinates the pipeline execution: Data Cleaning followed by Demand Forecasting
        followed by Inventory Recommendations.

        Args:
            task_details: Configuration containing raw, clean, and inventory CSV paths.
                Expected keys:
                - 'raw_data_path' (str): Path to raw CSV.
                - 'cleaned_data_path' (str): Path to output cleaned CSV.
                - 'inventory_output_path' (str): Path to output inventory recommendations CSV.
                - 'service_level_z' (float): Z-score service level.

        Returns:
            Dict[str, Any]: Coordinated pipeline execution results.
        """
        raw_path = task_details.get("raw_data_path", os.path.join("data", "raw", "pharma_supply_chain_messy_2.csv"))
        cleaned_path = task_details.get("cleaned_data_path", os.path.join("data", "processed", "cleaned_data.csv"))
        inventory_path = task_details.get("inventory_output_path", os.path.join("data", "processed", "inventory_recommendations.csv"))
        service_level_z = task_details.get("service_level_z", 1.65)

        # Step 1: Request Data Cleaning via A2A
        data_payload = {
            "raw_path": raw_path,
            "cleaned_path": cleaned_path
        }
        data_response = self.registry.call_agent("data_agent", "clean_data", data_payload)

        if data_response.get("status") != "success":
            return {
                "status": "failed",
                "error": f"Pipeline failed at DataAgent step: {data_response.get('error')}"
            }

        # Step 2: Request Demand Forecasting via A2A
        forecast_payload = {
            "data_path": cleaned_path
        }
        forecast_response = self.registry.call_agent(
            "forecasting_agent", "forecast_demand", forecast_payload
        )

        if forecast_response.get("status") != "success":
            return {
                "status": "failed",
                "data_agent_results": data_response,
                "error": f"Pipeline failed at ForecastingAgent step: {forecast_response.get('error')}"
            }

        # Step 3: Request Inventory Recommendations via A2A
        inventory_payload = {
            "data_path": cleaned_path,
            "predictions": forecast_response.get("predictions"),
            "service_level_z": service_level_z,
            "output_path": inventory_path
        }
        inventory_response = self.registry.call_agent(
            "inventory_agent", "calculate_inventory_recommendations", inventory_payload
        )

        if inventory_response.get("status") != "success":
            return {
                "status": "failed",
                "data_agent_results": data_response,
                "forecasting_agent_results": forecast_response,
                "error": f"Pipeline failed at InventoryAgent step: {inventory_response.get('error')}"
            }

        return {
            "status": "success",
            "data_agent_results": data_response,
            "forecasting_agent_results": forecast_response,
            "inventory_agent_results": inventory_response
        }
