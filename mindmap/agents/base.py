from abc import ABC, abstractmethod

from typing import Any, Optional
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from mindmap.utils.llm import _acomplete_chat, _acomplete_vision

import tiktoken
import re


class AgentInput(BaseModel):
    task: str
    use_text_model: bool


class BaseAgent(ABC):

    def __init__(self, instructions: Optional[str], model: str = 'gpt-4o') -> None:
        self._instructions = instructions
        self._model = model
        self.__price = 0.0
        
    @staticmethod
    def compute_num_tokens(text: str) -> int:
        encoder = tiktoken.encoding_for_model("gpt-4o-2024-11-20")
        return len(encoder.encode(text))
    
    @property
    def price(self) -> float:
        return self.__price

    @abstractmethod
    async def arun(self, input: Any) -> Any:
        raise NotImplementedError(f"Implement arun method")
    
    @staticmethod
    def _extract_from_pattern(text: str, word: str) -> str:
        pattern = fr'<{word}>(.*?)</{word}>'
        matches = re.findall(pattern, text)
        try:
            return matches[0]
        except IndexError:
            sub_pattern = f"```{word}"
            if sub_pattern in text:
                start = len(sub_pattern) + text.find(sub_pattern)
                end = len(text)
                if text[start:].find("```"):
                    end = start + text[start:].find("```")
                text = text[start:end]
            return text
 
    async def _acall_text(self, input_text: str, instructions: Optional[str] = None, model: str = "gpt-4o") -> Optional[str]:
        if self._instructions is None:
            raise ValueError(f"You have to define the instructions")
        if instructions is None:
            instructions = self._instructions

        chat_completion: ChatCompletion = await _acomplete_chat(system_prompt=instructions, user_prompt=input_text, model=model)
        self._update_price_after_chat_completion(completion=chat_completion)
        string_response = chat_completion.choices[0].message.content
        return string_response
    
    def _update_price_after_chat_completion(self, completion: ChatCompletion):

        usage = completion.usage
        
        if usage is None:
            return 0.0
        
        input_pricing = 2.5 if not 'mini' in self._model else 0.15
        output_pricing = 10.0 if not 'mini' in self._model else 0.6
        cached_input = 1.25 if not 'mini' in self._model else 0.75
        
        cached_price = (
            usage.prompt_tokens_details.cached_tokens * cached_input 
            if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens 
            else 0
        )

        input_price = usage.prompt_tokens * input_pricing
        output_price = usage.completion_tokens * output_pricing
        
        self.__price += (cached_price + input_price + output_price) / 1e6
