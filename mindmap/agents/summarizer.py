from mindmap.agents.base import BaseAgent
from mindmap.agents.text_prompts import SUMMARY_PROMPT, MERGE_SUMMARIES_PROMPT
from mindmap.agents.types import IndexerInfo, SummarizerInfo
from mindmap.utils.logger import logger


from typing import Optional, Dict, List
from typing_extensions import Coroutine
from pydantic import BaseModel

import asyncio


class SummarizerConfig(BaseModel):
    model: str = "gpt-4o-2024-11-20"
    merge_model: str = "gpt-4o-2024-11-20"
    max_tokens_to_summarize: int | None = None
    max_tokens_per_minute: int | None = None


class AgentSummarizer(BaseAgent):
    
    def __init__(self, config: SummarizerConfig, instructions: Optional[str] = None) -> None:
        if instructions is None:
            instructions = SUMMARY_PROMPT
        super().__init__(instructions=instructions, model=config.model)
        self._MAX_NUMBER_OF_TOKENS_TO_SUMMARIZE = config.max_tokens_to_summarize or 10_000
        self._MAX_NUMBER_OF_TOKENS_PER_MINUTE = config.max_tokens_per_minute or 200_000
        self._MERGE_MODEL = config.merge_model
    
    async def arun(self, input: IndexerInfo) -> SummarizerInfo:
        
        chapter_to_chunks = self._chapter_to_chunk(input)
                
        tasks: List[Coroutine] = []
        summaries = []

        token_count = 0
        for chapter_chunks in chapter_to_chunks.values():
            temp_token_count = sum(self.compute_num_tokens(chunk) for chunk in chapter_chunks)
            if token_count + temp_token_count > self._MAX_NUMBER_OF_TOKENS_PER_MINUTE:
                summaries.extend(await asyncio.gather(*tasks))
                await asyncio.sleep(3)
                token_count = temp_token_count
                tasks = [self._summarize_chapter(chapter_contents=chapter_chunks)]
            else:
                token_count += temp_token_count
                tasks.append(self._summarize_chapter(chapter_contents=chapter_chunks))

        if tasks:
            summaries.extend(await asyncio.gather(*tasks))
            
        logger.info(f"Number of summaries -> {len(summaries)}. Number of chapters: {len(chapter_to_chunks)}")
    
        chapter_to_summary = {chapter: summary for chapter, summary in zip(chapter_to_chunks, summaries)}
        return SummarizerInfo(chapter_title_to_summary=chapter_to_summary, price=self.price)
    
    async def _summarize_chapter(self, chapter_contents: List[str]) -> str | None:

        summaries = await asyncio.gather(*[
            self._acall_text(
                input_text=f"This is the content of the chapter: {content}", 
                model=self._model
            ) for content in chapter_contents
        ])
        
        if len(summaries) > 1:
            logger.warning(f"Chapter too long. Summarizing the chunks -> {len(summaries)}")
            user_prompt = "These are the summaries:" + "\n".join([f"\nSUMMARY {n+1}\n{s}" for n, s in enumerate(summaries)])
            merged_summary = await self._acall_text(input_text=user_prompt, instructions=MERGE_SUMMARIES_PROMPT, model=self._MERGE_MODEL)
        else:
            merged_summary = summaries[0]
    
        return merged_summary

    def _chapter_to_chunk(self, input: IndexerInfo):
        chapter_to_chunks: Dict[str, List[str]] = {}

        for chapter_title, chapter_text in input.chapter_to_text.items():
            chapter_to_chunks[chapter_title] = []
            if (n_tokens := self.compute_num_tokens(text=chapter_text)) > self._MAX_NUMBER_OF_TOKENS_TO_SUMMARIZE:
                n_chunks = n_tokens // self._MAX_NUMBER_OF_TOKENS_TO_SUMMARIZE + 1
                chapter_to_chunks[chapter_title] = [
                    chapter_text[k: k + len(chapter_text) // n_chunks] for k in range(n_chunks)
                ]
            else:
                chapter_to_chunks[chapter_title] = [chapter_text]
                
        return chapter_to_chunks
