# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")

# flake8: noqa: E402, E405
import os
from litellm import completion  # , RateLimitError
from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from rich.console import Console
import time
# from collections import deque
# from openai import OpenAIError
import logging
from google import genai


console = Console()
logger = logging.getLogger(__name__)
load_dotenv()


# Sliding window of request timestamps
# request_timestamps = deque(maxlen=5)  # Keep track of last 5 requests
# MAX_REQUESTS_PER_SECOND = 1

# def check_rate_limit():
#     now = time.time()

#     print(request_timestamps)

#     # Remove timestamps older than 1 second
#     while request_timestamps and now - request_timestamps[0] > 1:
#         request_timestamps.popleft()
    
#     if len(request_timestamps) >= MAX_REQUESTS_PER_SECOND:
#         # Calculate how long to wait
#         wait_time = 1 - (now - request_timestamps[0])
#         if wait_time > 0:
#             print(f"Rate limit exceeded, waiting for {wait_time} seconds")
#             time.sleep(wait_time + 0.5)  # Add 0.5s buffer
#             return check_rate_limit()  # Recursively check again after waiting
    
#     request_timestamps.append(now)

# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(OpenAIError)),
#     before_sleep=before_sleep_log(logger, log_level=logging.INFO)
# )
def ask(question, model, params={}):
    # try:
    if True:
        #check_rate_limit()  # Check rate limit before making request
        if model.startswith("gemini-v1alpha/"):
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={'api_version':'v1alpha'})
            response = client.models.generate_content(
                model=model.replace("gemini-v1alpha/", ""),
                contents=question,
                config={'thinking_config': {'include_thoughts': True}}
            )
            answer = response.text
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            reasoning = response.candidates[0].content.parts[0].text
        else:
            response = completion(
                model=model,
                messages=[{"content": question, "role": "user"}],
                **params
            )
            answer = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            # Check for provider_specific_fields
            reasoning = None
            if hasattr(response.choices[0].message, 'provider_specific_fields') and response.choices[0].message.provider_specific_fields:
                reasoning = response.choices[0].message.provider_specific_fields.get('reasoning')

        # sleep for 1 second to avoid hitting the rate limit
        time.sleep(1)

        return {
            "answer": answer,
            "reasoning": reasoning,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    # except Exception:
    #     console.print_exception()
    #     # raise
