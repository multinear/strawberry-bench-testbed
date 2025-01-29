import sys
from pathlib import Path
import time

# Add the project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# flake8: noqa: E402, E405
from engine import ask


def run_task(input):
    """
    Process the input using your GenAI-powered application logic.

    Args:
        input: The input data to process

    Returns:
        dict: A dictionary containing the output and processing details
    """
    model = input["model"]
    start_time = time.time()
    response = ask(input["question"], model, input.get("params", {}))
    end_time = time.time()
    response_time = end_time - start_time

    model_name = model.replace("openrouter/", "")

    details = {
        'model': model_name,
        'response_time': response_time,
        'prompt_tokens': response["prompt_tokens"],
        'completion_tokens': response["completion_tokens"],
    }
    if response.get("reasoning"):
        details["reasoning"] = response["reasoning"]

    return {'output': response["answer"], 'details': details}
