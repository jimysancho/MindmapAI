from mindmap.agents.base import BaseAgent
from mindmap.agents.types import IndexerInfo, QuoteExtractorInfo
from mindmap.agents.text_prompts import EXTRACT_QUOTES_FROM_TEXT_PROMPT
from mindmap.utils.logger import logger

from typing import Optional, Dict, List
from typing_extensions import Coroutine
from pydantic import BaseModel

import asyncio


class QuoteExtractorConfig(BaseModel):
    model: str
    max_tokens_to_extract_quotes: int | None = None
    max_tokens_per_minute: int | None = None
    

class QuoteExtractorAgent(BaseAgent):

    _NO_QUOTES = 'NO QUOTES'
    
    def __init__(self, config: QuoteExtractorConfig, instructions: Optional[str] = None) -> None:
        instructions = instructions or EXTRACT_QUOTES_FROM_TEXT_PROMPT
        super().__init__(instructions=instructions, model=config.model)
        self._MAX_NUMBER_OF_TOKENS_TO_SUMMARIZE = config.max_tokens_to_extract_quotes or 10_000
        self._MAX_NUMBER_OF_TOKENS_PER_MINUTE = config.max_tokens_per_minute or 200_000

    async def arun(self, input: IndexerInfo) -> QuoteExtractorInfo:
        
        chapter_to_chunks = self._chapter_to_chunk(input)
                
        tasks: List[Coroutine] = []
        quotes = []

        token_count = 0
        for chapter_title, chapter_chunks in chapter_to_chunks.items():
            temp_token_count = sum(self.compute_num_tokens(chunk) for chunk in chapter_chunks)
            if token_count + temp_token_count > self._MAX_NUMBER_OF_TOKENS_PER_MINUTE:
                quotes.extend(await asyncio.gather(*tasks))
                await asyncio.sleep(3)
                token_count = temp_token_count
                tasks = [
                    self._extract_quotes_from_chapter(chapter_title=chapter_title, chapter_contents=chapter_chunks)
                ]
            else:
                token_count += temp_token_count
                tasks.append(
                    self._extract_quotes_from_chapter(chapter_title=chapter_title, chapter_contents=chapter_chunks)
                )

        if tasks:
            quotes.extend(await asyncio.gather(*tasks))
            
        logger.debug(f"Number of Quotes -> {len(quotes)}. Number of chapters: {len(chapter_to_chunks)}")
    
        chapter_to_quotes = {chapter: quotes for chapter, quotes in zip(chapter_to_chunks, quotes)}
        return QuoteExtractorInfo(chapter_to_quotes=chapter_to_quotes, price=self.price)
    
    def _chapter_to_chunk(self, input: IndexerInfo) -> Dict[str, List[str]]:
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
    
    async def _extract_quotes_from_chapter(
        self, chapter_title: str, chapter_contents: List[str]
    ) -> List[str | None]:
        input_text = f"This is the title of the chapter: {chapter_title}"
        quotes = await asyncio.gather(*[
            self._acall_text(
                input_text=f"{input_text}\nThis is the content of the chapter: {content}",
                instructions=self._instructions, 
                model=self._model
            ) for content in chapter_contents
        ])
        
        return [quote for quote in quotes if quote and self._NO_QUOTES.lower() not in quote.lower()]
