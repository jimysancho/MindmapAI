from pydantic import BaseModel

from typing import Dict, List


class IndexerInfo(BaseModel):

    chapter_domain: Dict[str, List[int]]
    chapter_to_text: Dict[str, str]
    page_to_image_description: Dict[int, str]
    beginning_offset: int
    num_tokens: int
    price: float = 0.0
    
class SummarizerInfo(BaseModel):
    chapter_title_to_summary: Dict[str, str]
    price: float
    
    
class QuoteExtractorInfo(BaseModel):
    
    chapter_to_quotes: Dict[str, List[str]]
    price: float

class MindMapCreatorInfo(BaseModel):
    chapter_title_to_mind_map: Dict[str, str]
    price: float
    
    def update_mind_map(self, quotes: QuoteExtractorInfo) -> None:
        for chapter, chapter_mind_map in self.chapter_title_to_mind_map.items():
            quotes_from_chapter = quotes.chapter_to_quotes[chapter]
            chapter_mind_map += "\n## Quotes\n" + "\n".join(quotes_from_chapter)
            self.chapter_title_to_mind_map[chapter] = chapter_mind_map
