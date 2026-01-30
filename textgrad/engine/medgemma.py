import os
import platformdirs
from openai import OpenAI
from .openai import BaseOpenAIEngine
import re

class ChatMedGemma(BaseOpenAIEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    
    # Configuration from src/medgemma_api.py
    API_KEY = "78ff48759075851637df19727d6f6297c869682c1a4bfabac2389c5011d9483e"
    BASE_URL = "https://med-api.anipath.com/v1"

    def __init__(
        self,
        model_string: str = "MedAIBase/MedGemma1.5:4b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        **kwargs,
    ):
        """
        :param model_string: Model name to use.
        :param system_prompt: System prompt.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_medgemma_{model_string}.db")

        super().__init__(cache_path, system_prompt, model_string, is_multimodal)

        # Initialize the client with custom configuration
        self.client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.BASE_URL,
            default_headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

    def generate(self, content, system_prompt=None, **kwargs):
        response = super().generate(content, system_prompt=system_prompt, **kwargs)
        # Remove thinking process wrapped in <unused94> and <unused95>
        cleaned_response = re.sub(r'<unused94>.*?<unused95>', '', response, flags=re.DOTALL).strip()
        return cleaned_response
