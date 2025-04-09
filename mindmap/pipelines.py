from mindmap.agents import (
    Indexer, 
    AgentSummarizer, 
    MindMapCreatorAgent,
    QuoteExtractorAgent,
    IndexerInfo,
    SummarizerInfo,
    MindMapCreatorInfo, 
    QuoteExtractorInfo,
    IndexerConfig,
    SummarizerConfig, 
    MindMapConfig, 
    QuoteExtractorConfig
)

from mindmap.utils.logger import logger

from pydantic import BaseModel
from typing import Tuple, Any, Dict

import asyncio


class PipelineConfig(BaseModel):
    
    summary_model: str = "gpt-4o"
    mind_map_model: str = "gpt-4o"
    merge_mind_map_model: str = "gpt-4o"
    merge_summary_model: str = "gpt-4o"
    quote_extraction_model: str = "gpt-4o-mini"
    image_descriptor_model: str = "gpt-4o"

    tokens_to_summarize: int | None = None
    summary_tokens_per_minute: int | None = None
    mind_map_tokens_per_minute: int | None = None
    quote_extraction_tokens_per_minute: int | None = None
    tokens_to_extract_quotes: int | None = None
    max_number_of_tokens_index_proportion: int = 30_000
    index_tokens_batch: int = 10_000
    batch_images_descriptor: int = 10

    callback_index_pages_batch: int = 20
    min_pages_per_chapter_callback: int = 50
    max_words_per_chapter_callback: int = 10

    
    
class PipelineOutput(BaseModel):

    mind_map: MindMapCreatorInfo
    price: float
    

class Pipeline(BaseModel):
    
    pipeline_config: PipelineConfig

    def get_intermidiate_results(self) -> Dict[str, Any]:
        try:
            return self.__intermidiate_results
        except AttributeError:
            return {}

    def get_pipeline_agents(self) -> Tuple[Indexer, AgentSummarizer, QuoteExtractorAgent, MindMapCreatorAgent] | None:
        try:
            return self._indexer, self._summarizer, self._quote_extractor, self._mind_map_creator
        except AttributeError:
            logger.warning(f"Pipeline intitialized but not run")
            return None
    
    async def arun(self, path: str) -> PipelineOutput:
        logger.info(f"Pipeline to be run for book: {path}")
        try:
            mind_map_info, price = await self._create_mind_map_of_book(path=path)
        except Exception as e:
            logger.error(f"Error in pipeline.arun -> {e}")
            raise e
        return PipelineOutput(mind_map=mind_map_info, price=price)
    
    async def _create_mind_map_of_book(self, path: str) -> Tuple[MindMapCreatorInfo, float]:

        indexer_config: IndexerConfig = IndexerConfig(
            path=path,
            max_number_of_tokens_index_proportion=self.pipeline_config.max_number_of_tokens_index_proportion,
            index_batch=self.pipeline_config.index_tokens_batch,
            vision_model=self.pipeline_config.image_descriptor_model,
            batch_images=self.pipeline_config.batch_images_descriptor,
            callback_index_pages_batch=self.pipeline_config.callback_index_pages_batch,
            min_pages_per_chapter_callback=self.pipeline_config.min_pages_per_chapter_callback,
            max_words_per_chapter_callback=self.pipeline_config.max_words_per_chapter_callback
        )
        
        summarizer_config: SummarizerConfig = SummarizerConfig(
            model=self.pipeline_config.summary_model, 
            merge_model=self.pipeline_config.merge_summary_model, 
            max_tokens_to_summarize=self.pipeline_config.tokens_to_summarize, 
            max_tokens_per_minute=self.pipeline_config.summary_tokens_per_minute
        )
        
        mind_map_config: MindMapConfig = MindMapConfig(
            model=self.pipeline_config.mind_map_model, 
            max_number_of_tokens=self.pipeline_config.mind_map_tokens_per_minute
        )
        
        quote_extractor_config: QuoteExtractorConfig = QuoteExtractorConfig(
            model=self.pipeline_config.quote_extraction_model, 
            max_tokens_per_minute=self.pipeline_config.quote_extraction_tokens_per_minute, 
            max_tokens_to_extract_quotes=self.pipeline_config.tokens_to_extract_quotes
        )

        try:
            self._indexer = Indexer(indexer_config=indexer_config)
            self._summarizer = AgentSummarizer(config=summarizer_config)
            self._mind_map_creator = MindMapCreatorAgent(config=mind_map_config)
            self._quote_extractor = QuoteExtractorAgent(config=quote_extractor_config)
        except Exception as e:
            logger.error(f"Could not initialize some agent -> {e}")
            raise e

        try:
            indexer_info: IndexerInfo = await self._indexer.arun()
            logger.info(f"Indexer finished for book: {path}")
            summarizer_info, quote_extraction_info = await self._summarize_quote_extraction_in_paralel(indexer_info=indexer_info)
            logger.info(f"Summarizer and Quote Extraction finished for book: {path}")
            mind_map_info: MindMapCreatorInfo = await self._mind_map_creator.arun(summarizer_info)
            logger.info(f"MindMapCreator finished for book: {path}")
            mind_map_info.update_mind_map(quotes=quote_extraction_info)
        except Exception as e:
            logger.error(f"Error -> {e}")
            raise e
        
        self.__intermidiate_results = {
            'Indexer': indexer_info,
            'Summarizer': summarizer_info,
            'QuoteExtraction': quote_extraction_info,
        }

        price = indexer_info.price + summarizer_info.price + mind_map_info.price + quote_extraction_info.price
        logger.info(f"Pipeline for book: '{path}' finished. Final price: {price}")

        return mind_map_info, price

    async def _summarize_quote_extraction_in_paralel(self, indexer_info: IndexerInfo) -> Tuple[SummarizerInfo, QuoteExtractorInfo]:
        summarizer_info_task: asyncio.Task[SummarizerInfo] = asyncio.create_task(self._summarizer.arun(indexer_info))
        quote_extraction_info_task: asyncio.Task[QuoteExtractorInfo] = asyncio.create_task(self._quote_extractor.arun(indexer_info))

        try:
            results: Tuple[SummarizerInfo, QuoteExtractorInfo] = await asyncio.gather(
                summarizer_info_task, quote_extraction_info_task
            )
        except Exception as e:
            logger.error(f"Could not summarize and extract quotes -> {e}")
            raise e

        summarizer_info, quote_extraction_info = results
        return summarizer_info, quote_extraction_info
