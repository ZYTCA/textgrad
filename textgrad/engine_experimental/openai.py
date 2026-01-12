try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
from typing import List, Union
from textgrad.engine_experimental.engine_utils import open_ai_like_formatting
from textgrad.engine_experimental.base import EngineLM, cached
import diskcache as dc
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

class OpenAIEngine(EngineLM):
    """
    OpenAIEngine is a wrapper around the OpenAI API.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(self, model_string: str,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 is_multimodal: bool = False,
                 cache: Union[dc.Cache, bool] = False,
                 base_url: str = None):

        proxy_base_url = os.getenv("PROXY_LLM_BASE_URL")
        self.base_url = proxy_base_url or base_url

        self.validate()

        super().__init__(
            model_string=model_string,
            system_prompt=system_prompt,
            is_multimodal=is_multimodal,
            cache=cache
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = api_key or "proxy"
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def validate(self) -> None:
        if self.base_url:
            return
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")

    def openai_call(self, user_content, system_prompt, temperature, max_tokens, top_p):
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        return response.choices[0].message.content

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_single_prompt(
            self, content: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        return self.openai_call(content, system_prompt, temperature, max_tokens, top_p)

    @cached
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _generate_from_multiple_input(
            self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        formatted_content = open_ai_like_formatting(content)

        return self.openai_call(formatted_content, system_prompt, temperature, max_tokens, top_p)

    def __call__(self, content, **kwargs):
        return self.generate(content, **kwargs)



class OpenAICompatibleEngine(OpenAIEngine):
    """
        This is the same as engine.openai.ChatOpenAI, but we pass in an external OpenAI client.
        """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    client = None

    def __init__(self,
                 client,
                 model_string: str,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 is_multimodal: bool = False,
                 cache: Union[dc.Cache, bool] = False):

            self.client = client

            super().__init__(
                model_string=model_string,
                system_prompt=system_prompt,
                is_multimodal=is_multimodal,
                cache=cache
            )
