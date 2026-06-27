"""
Common Utilities Module.
Contains port maps, URL generators, and unified JSON POST request helpers.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict

# Standard network ports assigned to each agent
AGENT_PORTS = {
    "host_agent": 8000,
    "data_agent": 8001,
    "forecasting_agent": 8002,
    "inventory_agent": 8003,
    "optimization_agent": 8004
}


def get_agent_url(agent_name: str, host: str = "localhost") -> str:
    """
    Builds the base HTTP URL for a given agent name.
    """
    port = AGENT_PORTS.get(agent_name)
    if not port:
        raise ValueError(f"Unknown agent name: {agent_name}")
    return f"http://{host}:{port}"


def send_json_post(url: str, data: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    """
    Sends a HTTP POST request with JSON payload to a URL.
    Returns the parsed JSON response dict. Uses only the python standard library.
    """
    req_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = response.read().decode("utf-8")
            return json.loads(res_data)
    except urllib.error.HTTPError as e:
        try:
            err_data = e.read().decode("utf-8")
            return {"status": "failed", "error": f"HTTP Error {e.code}: {err_data}"}
        except Exception:
            return {"status": "failed", "error": f"HTTP Error {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {
            "status": "failed",
            "error": f"Network connection failed: {e.reason}. Is the agent server running?"
        }
    except Exception as e:
        return {"status": "failed", "error": f"Request failed: {str(e)}"}
