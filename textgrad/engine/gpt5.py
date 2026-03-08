import os
import platformdirs
from openai import OpenAI
from .openai import BaseOpenAIEngine
import base64
from .engine_utils import get_image_type_from_bytes
import json

class ChatGPT5(BaseOpenAIEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    
    # Configuration
    BASE_URL = "https://yunwu.ai/v1"

    def __init__(
        self,
        model_string: str = "gpt-5-nano",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs,
    ):
        """
        :param model_string: Model name to use.
        :param system_prompt: System prompt.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_gpt5_{model_string}.db")

        super().__init__(cache_path, system_prompt, model_string, is_multimodal)

        api_key = os.getenv("OPENAI_KEY")
        if api_key is None:
             api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError("Please set the OPENAI_KEY or OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=30.0
        )

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=1500,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        
        cache_key = sys_prompt_arg + prompt
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        input_messages = []
        if sys_prompt_arg:
            input_messages.append({
                "role": "system",
                "content": [{"type": "input_text", "text": sys_prompt_arg}]
            })
            
        input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        })

        response = self.client.responses.create(
            model=self.model_string,
            input=input_messages,
        )

        response_text = response.output_text
        self._save_cache(cache_key, response_text)
        return response_text

    def _format_content(self, content):
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode("utf-8")
                formatted_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/{image_type};base64,{base64_image}"
                    }
                )
            elif isinstance(item, str):
                formatted_content.append({"type": "input_text", "text": item})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content,
        system_prompt=None,
        temperature=0,
        max_tokens=1500,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        input_messages = []
        if sys_prompt_arg:
            input_messages.append({
                "role": "system",
                "content": [{"type": "input_text", "text": sys_prompt_arg}]
            })
        
        input_messages.append({
            "role": "user",
            "content": formatted_content
        })

        response = self.client.responses.create(
            model=self.model_string,
            input=input_messages,
        )

        response_text = response.output_text
        self._save_cache(cache_key, response_text)
        return response_text
