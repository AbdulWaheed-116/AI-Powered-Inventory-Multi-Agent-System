"""
A2A Client Module.
Provides the A2AClient class for network-based Agent-to-Agent communication.
"""

from typing import Any, Dict
from common.utils import get_agent_url, send_json_post


class A2AClient:
    """
    A client to dispatch messages to agent network endpoints.
    Allows executing actions on remote agents and retrieving result payloads.
    """

    def __init__(self, host: str = "localhost") -> None:
        """
        Initializes the A2AClient.
        """
        self.host = host

    def call_agent(self, agent_name: str, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches a capability request over HTTP to the specified agent.

        Args:
            agent_name: Name of the target agent (e.g. 'data_agent', 'forecasting_agent').
            action: The capability/action name to execute (e.g. 'clean_data').
            payload: Input parameters matching the capability schema.

        Returns:
            Dict[str, Any]: Output payload returned by the target agent.
        """
        try:
            url = get_agent_url(agent_name, self.host)
        except ValueError as e:
            return {"status": "failed", "error": str(e)}

        request_body = {
            "action": action,
            "payload": payload
        }
        
        # Route POST request to HTTP server and return response
        return send_json_post(url, request_body)
