"""
Runner Module.
Handles environment configuration and executes the RootAgent using InMemoryRunner.
Supports programmatically querying the agent or starting an interactive terminal CLI.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load env variables from project root using absolute path
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

from google.adk.runners import InMemoryRunner, print_event
from google.genai import types
from adk.root_agent import root_agent


async def ensure_session(runner: InMemoryRunner, user_id: str, session_id: str) -> None:
    """Helper to ensure the session exists in the session service."""
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    if not session:
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )


async def run_query(
    query: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    verbose: bool = True
) -> None:
    """Runs a single query through the Root Agent and prints progress events in real-time.

    Args:
        query: The instruction or query for the agent.
        user_id: User identifier.
        session_id: Session identifier.
        verbose: Whether to show detailed tool call event traces.
    """
    # Verify dataset existence (scan for any CSV file in data/raw/)
    raw_dir = os.path.join(project_root, "data", "raw")
    has_csv = False
    if os.path.exists(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if csv_files:
            has_csv = True
            
    if not has_csv:
        print(f"\n[Error] Required raw dataset is missing in {raw_dir}")
        print("Please upload the dataset using the existing Streamlit pipeline first.")
        return

    runner = InMemoryRunner(agent=root_agent)
    runner.auto_create_session = True
    
    # Pre-create session to avoid SessionNotFoundError
    await ensure_session(runner, user_id, session_id)
    
    # Construct UserContent as expected by ADK v2.3.0
    new_message = types.UserContent(
        parts=[types.Part(text=query)]
    )
    
    print(f"\n>>> Query: {query}")
    print("-" * 60)
    
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            print_event(event, verbose=verbose)
    except Exception as e:
        err_msg = str(e)
        if any(kw in err_msg for kw in ["429", "RESOURCE_EXHAUSTED", "Quota exceeded", "ResourceExhausted"]):
            print("\n[WARNING] Gemini API Quota Exceeded (429 RESOURCE_EXHAUSTED)")
            print("Your API key has exceeded its free tier rate limits or daily quota.")
            print("Mitigations:")
            print("1. Wait 10-30 seconds if it is a transient rate limit.")
            print("2. Upgrade your API key to pay-as-you-go on Google AI Studio (https://aistudio.google.com/).")
            print(f"Details:\n{err_msg}\n")
        else:
            print(f"\nError: {err_msg}")
        
    print("-" * 60)


async def interactive_cli() -> None:
    """Starts an interactive session with the Root Agent in the terminal."""
    # Verify dataset existence (scan for any CSV file in data/raw/)
    raw_dir = os.path.join(project_root, "data", "raw")
    has_csv = False
    if os.path.exists(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if csv_files:
            has_csv = True
            
    if not has_csv:
        print(f"\n[Error] Required raw dataset is missing in {raw_dir}")
        print("Please upload the dataset using the existing Streamlit pipeline first.")
        return

    print("=============================================================")
    print("     AI Inventory Optimization System - Root Agent CLI")
    print("=============================================================")
    print("Type your request or query (e.g., 'run the optimization pipeline', ")
    print("'what are the top supplier changes?', or 'exit' to quit).\n")
    
    user_id = "cli_user"
    session_id = "cli_session"
    
    # Initialize runner once to keep the session history
    runner = InMemoryRunner(agent=root_agent)
    runner.auto_create_session = True
    
    # Pre-create CLI session
    await ensure_session(runner, user_id, session_id)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Exiting Root Agent CLI. Goodbye!")
                break
                
            new_message = types.UserContent(
                parts=[types.Part(text=query)]
            )
            
            print("-" * 60)
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                print_event(event, verbose=False)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["429", "RESOURCE_EXHAUSTED", "Quota exceeded", "ResourceExhausted"]):
                print("\n[WARNING] Gemini API Quota Exceeded (429 RESOURCE_EXHAUSTED)")
                print("Your API key has exceeded its free tier rate limits or daily quota.")
                print("Mitigations:")
                print("1. Wait 10-30 seconds if it is a transient rate limit.")
                print("2. Upgrade your API key to pay-as-you-go on Google AI Studio (https://aistudio.google.com/).")
                print(f"Details:\n{err_msg}\n")
            else:
                print(f"\nError: {err_msg}")


if __name__ == "__main__":
    # If command line arguments are provided, run as a one-off query.
    # Otherwise, start interactive CLI mode.
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        asyncio.run(run_query(query_str))
    else:
        asyncio.run(interactive_cli())

