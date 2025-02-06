import sys
from pathlib import Path
import time
from openai import OpenAI
from pydantic import BaseModel, Field

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
    response = ask(
        input["question"], model, input.get("params", {}), input.get("extra", {})
    )
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


class ExtractPoemResponse(BaseModel):
    poem_text: str = Field(
        description="The poem text, without any additional text before or after"
    )


def evaluate_custom(input, output, spec):
    """
    Evaluate the output against the custom spec: All poem words start with the letter s
    """
    # print(f"Evaluating: {input['question']} against {spec}")

    # Extract just the poem

    client = OpenAI()
    model = "gpt-4o-mini"

    prompt = "Extract just the poem text from the following text."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": output},
        ],
        temperature=0.0,  # reduce randomness
        response_format=ExtractPoemResponse,
    )

    parsed_response = completion.choices[0].message.parsed

    poem_text = parsed_response.poem_text

    # Split into words and clean up
    words = [word.strip('.,!?-:;"\'()').lower() for word in poem_text.split()]
    # Filter out empty strings
    words = [word for word in words if word]

    # Find non-s words
    non_s_words = [word for word in words if not word.startswith('s')]

    # Calculate score (1 if all words start with 's', 0 otherwise)
    score = 1.0 if len(non_s_words) == 0 else 0.0

    return {
        "score": score,
        "metadata": {
            "evaluations": [
                {
                    "criterion": "All poem words start with the letter s",
                    "score": score,
                    "rationale": (
                        "All words start with 's'"
                        if score == 1.0
                        else f"Found {len(non_s_words)} words that don't start with 's': {', '.join(non_s_words)}"
                    ),
                },
            ]
        },
    }
