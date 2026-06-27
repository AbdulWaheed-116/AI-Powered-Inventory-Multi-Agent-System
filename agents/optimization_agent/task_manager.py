"""
Task Manager Module for Optimization Agent.
Manages tasks for the OptimizationAgent, including task creation,
state transitions (PENDING, RUNNING, COMPLETED, FAILED), and result retrieval.
"""

import uuid
import datetime
from typing import Any, Dict, List, Optional
from .agent import OptimizationAgent


class TaskManager:
    """
    Manages tasks for the OptimizationAgent. Handles scheduling, state tracking,
    execution, and storing execution results.
    """

    def __init__(self, agent: Optional[OptimizationAgent] = None) -> None:
        """
        Initializes the TaskManager.
        If no agent is provided, instantiates a default OptimizationAgent.
        """
        self.agent = agent or OptimizationAgent()
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, input_data: Dict[str, Any]) -> str:
        """
        Creates a new optimization task and registers it with a PENDING state.

        Args:
            input_data: Configuration details for the task (e.g. data_path, service_level_z).

        Returns:
            str: A unique UUID for the task.
        """
        task_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()

        self.tasks[task_id] = {
            "task_id": task_id,
            "status": "PENDING",
            "created_at": now,
            "updated_at": now,
            "input_data": input_data,
            "result": None,
            "error": None
        }
        return task_id

    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Executes a task by changing its status to RUNNING, invoking the
        OptimizationAgent, and updating the state to COMPLETED or FAILED.

        Args:
            task_id: The ID of the task to run.

        Returns:
            Dict[str, Any]: The task dictionary with execution details.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task with ID {task_id} not found.")

        task = self.tasks[task_id]
        if task["status"] in ["RUNNING", "COMPLETED"]:
            return task

        task["status"] = "RUNNING"
        task["updated_at"] = datetime.datetime.now().isoformat()

        # Execute agent calculations
        response = self.agent.run_task(task["input_data"])

        task["updated_at"] = datetime.datetime.now().isoformat()
        if response.get("status") == "success":
            task["status"] = "COMPLETED"
            task["result"] = response
            task["error"] = None
        else:
            task["status"] = "FAILED"
            task["error"] = response.get("error", "An unknown error occurred.")
            task["result"] = response

        return task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves task details by task ID.

        Args:
            task_id: Unique task identifier.

        Returns:
            Optional[Dict[str, Any]]: Task dictionary if found, else None.
        """
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        Lists all tasks managed by this manager.

        Returns:
            List[Dict[str, Any]]: List of task dictionaries.
        """
        return list(self.tasks.values())
