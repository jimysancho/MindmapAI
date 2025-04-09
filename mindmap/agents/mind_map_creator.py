from mindmap.agents.base import BaseAgent
from mindmap.agents.text_prompts import MIND_MAP_CREATION_PROMPT, MERGE_MIND_MAP_PROMPT
from mindmap.utils.logger import logger
from mindmap.agents.types import SummarizerInfo, MindMapCreatorInfo

from typing import Optional, List
from typing_extensions import Coroutine
from pydantic import BaseModel

import asyncio


class MindMapConfig(BaseModel):
    model: str
    max_number_of_tokens: int | None = None


class MindMapCreatorAgent(BaseAgent):
    
    def __init__(self, config: MindMapConfig, instructions: Optional[str] = None) -> None:
        if instructions is None:
            instructions = MIND_MAP_CREATION_PROMPT
        super().__init__(instructions=instructions, model=config.model)
        self._MAX_NUMBER_OF_TOKENS = config.max_number_of_tokens or 30_000
        
    async def arun(self, summarizer_output: SummarizerInfo) -> MindMapCreatorInfo:
    
        chapter_to_summaries = summarizer_output.chapter_title_to_summary
                
        tasks: List[Coroutine] = []
        mind_maps = []

        token_count = 0
        for chapter_summary in chapter_to_summaries.values():
            temp_token_count = self.compute_num_tokens(chapter_summary)
            if token_count + temp_token_count > self._MAX_NUMBER_OF_TOKENS:
                mind_maps.extend(await asyncio.gather(*tasks))
                token_count = temp_token_count
                tasks = [self._mind_map_chapter(summary_text=chapter_summary)]
            else:
                token_count += temp_token_count
                tasks.append(self._mind_map_chapter(summary_text=chapter_summary))

        if tasks:
            mind_maps.extend(await asyncio.gather(*tasks))

        chapter_to_mindmap = {chapter: 
            self._extract_from_pattern(text=mind_map, word='markdown')
            for chapter, mind_map in zip(chapter_to_summaries, mind_maps)
        }

        return MindMapCreatorInfo(chapter_title_to_mind_map=chapter_to_mindmap, price=self.price)
        
    async def _mind_map_chapter(self, summary_text: str) -> str | None:
        return await self._acall_text(input_text=f"This is the input text: {summary_text}", model=self._model)
