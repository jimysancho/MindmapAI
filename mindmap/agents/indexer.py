from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult

from typing import Dict, List, Tuple, Any
from openai import AsyncClient
from openai.types.chat import ChatCompletion

from pydantic import BaseModel

import os
import tiktoken
import asyncio

from mindmap.agents.types import IndexerInfo
from mindmap.utils.helpers import (
    remove_punctuation,
    encode_image,
    _turn_page_into_image
)
from mindmap.utils.logger import logger
from mindmap.utils.llm import _acomplete_chat, _acomplete_vision


class NoIndexFoundException(Exception):
    pass


class BookTooLongException(Exception):
    pass


class CallbackIndexExtractionException(Exception):
    pass


class IndexerConfig(BaseModel):
    path: str
    max_number_of_tokens_index_proportion: int = 30_000
    index_batch: int = 10_000
    vision_model: str = "gpt-4o-2024-11-20"
    extract_index_model: str = "gpt-4o-2024-11-20"
    batch_images: int = 10
    callback_index_pages_batch: int = 20
    min_pages_per_chapter_callback: int = 50
    max_words_per_chapter_callback: int = 10


class Indexer:

    _NO_INDEX_FIELD = "NO INDEX"
    _CHAPTER_PAGE_SEPARATOR = "CHAPTER_PAGE_SEP"
    _NO_CHAPTER_FOUND = "NO CHAPTER"
    _NO_FIGURE_FIELD = "NO FIGURE"
    _BOLD_DOCLING_SEP = "## "
    
    def __init__(self, indexer_config: IndexerConfig) -> None:

        self._path = indexer_config.path
        self._extract_index_model = indexer_config.extract_index_model
        self._CALLBACK_INDEX_BATCH = indexer_config.callback_index_pages_batch
        self._MAX_NUMBER_OF_TOKENS = indexer_config.max_number_of_tokens_index_proportion
        self._INDEX_BATCH = indexer_config.index_batch
        self._VISION_MODEL = indexer_config.vision_model
        self._BATCH_OF_IMAGES = indexer_config.batch_images
        self._MIN_PAGES_PER_CHAPTER_CALLBACK = indexer_config.min_pages_per_chapter_callback
        self._MAX_WORDS_PER_CHAPTER_CALLBACK = indexer_config.max_words_per_chapter_callback

        self.__intialized = False
        # TODO -> max number of pages depending on the product?

        self.__book_info: Dict[str, Any] = {}
        self.__price = 0.0
        
    @property
    def docling_output(self) -> ConversionResult:
        return self._docling_output
    
    @property
    def docling_output_dict(self) -> dict:
        try:
            return self._docling_output_dict
        except AttributeError:
            logger.error(f"Intialize the Indexer first")
            raise AttributeError
    
    @property
    def docling_markdown(self) -> str:
        try:
            return self._docling_output_markdown
        except AttributeError:
            logger.error(f"Intialize the Indexer first")
            raise AttributeError

    @property
    def book_info(self) -> IndexerInfo:
        return IndexerInfo(**self.__book_info)

    @property
    def num_tokens(self) -> int:
        encoder = tiktoken.encoding_for_model('gpt-4o')
        return len(encoder.encode(self._docling_output_markdown))
    
    @staticmethod
    def _compute_num_tokens(text: str) -> int:
        encoder = tiktoken.encoding_for_model('gpt-4o')
        return len(encoder.encode(text))
    
    async def _initialize(self) -> None:
        try:
            self._docling_output: ConversionResult = await asyncio.to_thread(self._extract_docling_output)
        except Exception as e:
            logger.error(f"Fatal error. Could not initialize Indexer -> {e}")
            raise e
        self._docling_output_dict: dict = self._docling_output.document.export_to_dict()
        self._docling_output_markdown: str = self._docling_output.document.export_to_markdown()
        self._number_of_pages = sorted(self._docling_output_dict['pages'].values(), key=lambda x: -x['page_no'])[0]['page_no']
        
        self.__intialized = True
        logger.info(f"Indexer initialized for book: '{self._path}' -> Number of pages: {self._number_of_pages}")

    def _extract_docling_output(self) -> ConversionResult:
        logger.info(f"Extracting text from book: {self._path}")
        converter = DocumentConverter()
        return converter.convert(self._path)
    
    async def arun(self) -> IndexerInfo:
        
        if not self.__intialized:
            try:
                await self._initialize()
            except Exception as e:
                raise e

        chapter_domain_task: asyncio.Task[Dict[str, List[int]]] = asyncio.create_task(self._chapter_domain())
        page_description_task: asyncio.Task[Dict[int, str]] = asyncio.create_task(self._descriptions_of_images())
        try:
            results: Tuple[Dict[str, List[int]], Dict[int, str]] = await asyncio.gather(chapter_domain_task, page_description_task)
        except NoIndexFoundException as e:
            raise e
        except CallbackIndexExtractionException as e:
            raise e
        except Exception as e:
            logger.error(f"Could not extract chapter domain or page description -> {e}")
            raise e
    
        chapter_to_domain, page_to_description = results
        page_to_text = self._page_to_text()
        chapter_to_text = {}

        for chapter_title, chapter_domain in chapter_to_domain.items():
            try:
                begin_page, end_page = chapter_domain
            except ValueError:
                begin_page, end_page = chapter_domain[0], None
                
            chapter_content = []
                
            for page, content in page_to_text.items():
                if begin_page <= page and (end_page is None or page <= end_page):
                    if page in page_to_description:
                        chapter_content.append(f"This is the description of an image: {page_to_description[page]}")
                    chapter_content.append(content)

            chapter_to_text[chapter_title] = "\n".join(chapter_content)
    
        self.__book_info["chapter_to_text"] = chapter_to_text
        self.__book_info["page_to_image_description"] = page_to_description
        self.__book_info["price"] = self.__price
        return self.book_info

    async def _chapter_domain(self) -> Dict[str, List[int]]:
        
        try:
            chapter_to_page, offset_page = await self._extract_book_info()
        except NoIndexFoundException as e:
            logger.error(f"Book: '{self._path}' has no index. Trying callback markdown callback")
            chapter_to_page = await self._callback_for_index_extraction()
            offset_page = 0
        except CallbackIndexExtractionException as e:
            logger.error(f"Book: '{self._path}' callback failed. Trying last index extraction")
            # NOTE -> create a Chunker to extract main themes from pages (maybe is too expensive)
            raise e
        except Exception as e:
            logger.error(f"Book: '{self._path}'. Could not extract chapter domain -> {e}")
            raise e

        if all(
            len([char for char in chapter.split(self._CHAPTER_PAGE_SEPARATOR) if char]) ==  1 
            for chapter in chapter_to_page
        ):
            logger.error(f"Book: '{self._path}'. No chapter has been identified properly...")
            try:
                chapter_to_page = await self._callback_for_index_extraction()
            except Exception as e:
                logger.error(f"Callback failed for book: {self._path} -> {e}")
                raise e
            offset_page = 0

        chapter_to_domain = {}
        prev_chapter, prev_page = None, None

        for chapter_line in chapter_to_page:
            
            try:
                chapter_title, init_page = chapter_line.split(self._CHAPTER_PAGE_SEPARATOR)
            except ValueError as e:
                logger.error(f"Chapter title and page not properly extracted for book: '{self._path}' -> {chapter_line}")
                number_of_items = chapter_line.split(self._CHAPTER_PAGE_SEPARATOR)
                # TODO -> maybe we can use this
                raise e

            try:
                init_page_int = int(init_page) + offset_page
            except ValueError:
                proposed_page = self._look_for_missing_page_of_chapter(start_page=prev_page, chapter_title=chapter_title)
                init_page_int = proposed_page or ((prev_page + 1) if prev_page is not None else 1)
                logger.warning(f"Book: '{self._path}' has a chapter with no page. Proposed page for chapter: {chapter_title} -> {proposed_page}")
            chapter_title = chapter_title.strip()

            if prev_chapter:
                chapter_to_domain[prev_chapter].append(init_page_int)

            chapter_to_domain[chapter_title] = [init_page_int]
            prev_chapter = chapter_title
            prev_page = init_page_int
            
        self.__book_info.update({
            "chapter_domain": chapter_to_domain,
            "beginning_offset": offset_page,
            "num_tokens": self.num_tokens
        })

        return chapter_to_domain
    
    async def _extract_book_info(self) -> Tuple[List[str], int]:

        page_to_text = self._page_to_text()
        max_page = max(list(page_to_text.keys()))
        try:
            chapter_to_page = await self._extract_chapters()
            if len(chapter_to_page) == 1 and self._NO_INDEX_FIELD in chapter_to_page[0]:
                raise NoIndexFoundException
        except NoIndexFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Could not extract chapters for book: {self._path} -> {e}")
            raise e

        offset_page_freq: Dict[int, int] = {}

        for chapter in chapter_to_page:
            chapter_title = chapter.split(self._CHAPTER_PAGE_SEPARATOR)[0].strip()
            try:
                chapter_page = int(chapter.split(self._CHAPTER_PAGE_SEPARATOR)[1].strip())
            except IndexError:
                logger.warning(f"Book: '{self._path}' has a chapter with no separator -> {chapter}")
                chapter_page = 0
            except ValueError:
                logger.warning(f"Book: '{self._path}' has a chapter with no page detected -> {chapter}")
                chapter_page = 0

            for page_number, page_text in page_to_text.items():
                begin_text = " ".join(page_text.split(" ")[:self._MAX_WORDS_PER_CHAPTER_CALLBACK])

                if (chapter_title.lower() in begin_text.lower().replace("\n", " ") 
                    and not f"{chapter_title.lower()} " + str(chapter_page) in begin_text.lower()):
                    if page_number < max_page // 5 or page_number > (max_page - max_page // 5):
                        continue
                    offset_page = page_number - chapter_page
                    if offset_page not in offset_page_freq:
                        offset_page_freq[offset_page] = 0
                    offset_page_freq[offset_page] += 1

        offset_page = self._return_most_repeated_offset(offset_page_freq)
        logger.info(f"Book: '{self._path}' has {len(chapter_to_page)} chapters. Offset -> {offset_page}")
        return chapter_to_page, offset_page
        
    async def _extract_chapters(self) -> List[str]:
        # NOTE -> consider the case of very detailed indexes. it may be really useful 
        # to extract a hierarchical index
        proportion = self._compute_index_proportion()

        try:
            index_exists = await self._llm_extract_index(
                text=self._docling_output_markdown[:proportion]
            )
            if self._NO_INDEX_FIELD in index_exists:
                index_exists = await self._llm_extract_index(
                    text=self._docling_output_markdown[-proportion:]
                )
        except Exception as e:
            logger.error(f"Could call llm for book: {self._path} -> {e}")
            raise e
            
        return [c.strip() for c in index_exists.split("\n") if c]
        
    async def _llm_extract_index(self, text: str) -> str:
        client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])

        try:
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Extract JUST the table of contents from the text following this format: \nchapter title {self._CHAPTER_PAGE_SEPARATOR} page\nA new line for every chapter. Just extract the table of contents, without saying anything else. If there is no table of contents, return {self._NO_INDEX_FIELD}"}, 
                    {"role": "user", "content": f"This is the text: \n{text}"}
                ],
                temperature=0, 
                model=self._extract_index_model
            )
        except Exception as e:
            logger.error(f"Could not call openai api for book: {self._path} -> {e}")
            raise e

        self._update_price_after_chat_completion(completion=response)

        return response.choices[0].message.content or self._NO_INDEX_FIELD
    
    def _compute_index_proportion(self) -> int:

        max_tokens = self._MAX_NUMBER_OF_TOKENS
        current_tokens = 0
        index = 0
        index_batch = self._INDEX_BATCH

        while index < len(self._docling_output_markdown):
            next_chunk = self._docling_output_markdown[:index + index_batch]
            current_tokens += self._compute_num_tokens(text=next_chunk)

            if current_tokens > max_tokens:
                break

            index += index_batch

        return index

    def _page_to_text(self) -> Dict[int, str]:

        texts = self._docling_output_dict['texts']
        current_page = 1
        current_text = ""
        page_to_text = {}

        for text in texts:
            for pr in text['prov']:
                page = pr['page_no']
                if page != current_page:
                    page_to_text[current_page] = current_text + "\n"
                    current_text = ""
                    current_page = page

            current_text += text['text'] + "\n"

        if current_text:
            page_to_text[current_page] = current_text + "\n"

        return page_to_text
    
    async def _descriptions_of_images(self) -> Dict[int, str]:
        pages_with_pictures = self._pages_of_pictures()
        logger.info(f"Book: '{self._path}'. Total number of pages with figures: {len(pages_with_pictures)}")
        
        if len(pages_with_pictures) > self._number_of_pages // 4:
            logger.warning(f"Book: '{self._path}' has too many images detected by docling")
            return {}

        descriptions = {}
    
        for batch_range in range(0, len(pages_with_pictures), self._BATCH_OF_IMAGES):
    
            batch = pages_with_pictures[batch_range: batch_range + self._BATCH_OF_IMAGES]
            try:
                batch_descriptions = await asyncio.gather(
                    *[self._picture_to_text(page_number=page) for page in batch]
                )
            except Exception as e:
                logger.error(f"Something went wrong when computing images for book: {self._path} -> {e}")
                continue
            
            descriptions.update({
                page: description
                for page, description in zip(batch, batch_descriptions)
                if self._NO_FIGURE_FIELD not in description
            })
        return descriptions

    async def _picture_to_text(self, page_number: int) -> str:

        image_path = _turn_page_into_image(
            document_path=self._path, page_number=page_number
        )
        encoded_image = encode_image(image_path=image_path)
        try:
            response: ChatCompletion = await _acomplete_vision(
                encoded_image=encoded_image,
                prompt=f"Give a detailed but short description of the figure found in the following book page. If there is no figure, return {self._NO_FIGURE_FIELD}",
                model=self._VISION_MODEL
            )
            os.remove(image_path)
            self._update_price_after_chat_completion(completion=response)
            return response.choices[0].message.content or self._NO_FIGURE_FIELD
        except Exception as e:
            logger.error(f"Book: '{self._path}' -> Could not extract text from image -> {e}")
            return self._NO_FIGURE_FIELD

    def _pages_of_pictures(self) -> List[int]:
        return list(set([picture['prov'][0]['page_no']
                for picture in self._docling_output_dict['pictures']]))
        
    @staticmethod
    def _is_page_in_domain(page: int, domain: Tuple[int, int | None]) -> bool:
        start, end = domain
        if end is None:
            if start <= page:
                return True
            return False
        if start <= page <= end:
            return True
        return False
    
    def _return_most_repeated_offset(self, offset_dict: Dict[int, int]) -> int:
        try:
            total = sum(list(offset_dict.values()))
            most_repeated_offset, frequency = sorted(offset_dict.items(), key=lambda x: -x[1])[0]
            if frequency < total // 2:
                logger.warning(f"Book: '{self._path}'. Offset frequency threshold not met -> {offset_dict}")
                return 0
            logger.info(f"Book: '{self._path}'. Offset frequency: {offset_dict}")
            return most_repeated_offset
        except IndexError:
            logger.warning(f"Book: '{self._path}'. IndexError -> {offset_dict}")
            return 0
        except ZeroDivisionError:
            logger.warning(f"Book: '{self._path}'. Zero Division Error")
            return 0

    def _look_for_missing_page_of_chapter(
        self, start_page: int | None, chapter_title: str
    ) -> int | None:

        if start_page is None:
            start_page = 0
        page_to_text = self._page_to_text()
        for page, text in page_to_text.items():
            if page <= start_page:
                continue
            begin_text = text[:1_000]

            if (
                remove_punctuation(chapter_title.lower().strip()) 
                in remove_punctuation(begin_text.lower().replace("\n", " ").replace("\n\n", " "))
            ):
                return page
        return None
    
    async def _callback_for_index_extraction(self) -> List[str]:
        try:
            markdown_callback = self._markdown_index_extraction_callback()
            if len(markdown_callback):
                return markdown_callback
            logger.warning(f"Book: '{self._path}'. Markdown callback did not yield any chapters")
            llm_callback = await self._llm_index_extraction_callback()
            return llm_callback
        except Exception as e:
            logger.error(f"Book: '{self._path}' callback index extraction failed -> {e}")
            raise CallbackIndexExtractionException(f"Callback failed -> {e}")

    def _markdown_index_extraction_callback(self) -> List[str]:

        markdown = self._docling_output_markdown
        possible_chapters = markdown.split(self._BOLD_DOCLING_SEP)
        if len(possible_chapters) < self._number_of_pages // self._MIN_PAGES_PER_CHAPTER_CALLBACK:
            logger.warning(f"Book: '{self._path}'. Possible chapters: {len(possible_chapters)} < {self._number_of_pages // self._MIN_PAGES_PER_CHAPTER_CALLBACK}")
            return []

        chapters = [
            chapter.split("\n")[0] for chapter in possible_chapters 
            if len(chapter.split("\n")[0].split(" ")) < self._MAX_WORDS_PER_CHAPTER_CALLBACK
        ]

        chapters_with_page = []

        for chapter in chapters:
            found = False
            for page, page_text in self._page_to_text().items():
                begining_of_page_text = " ".join(page_text.split(" ")[:self._MAX_WORDS_PER_CHAPTER_CALLBACK])
                if (
                    remove_punctuation(text=chapter.lower().strip()) 
                    in remove_punctuation(text=begining_of_page_text.lower().strip())
                ):
                    chapters_with_page.append(
                        chapter + self._CHAPTER_PAGE_SEPARATOR + str(page)
                    )
                    found = True
                    break
            if not found:
                logger.warning(f"Possible chapter: {chapter} has not being found")
                    
        return chapters_with_page

    async def _llm_index_extraction_callback(self) -> List[str]:
        page_to_text = self._page_to_text()
        tasks = []
        
        batch_text = ""
        
        for index, (page, text) in enumerate(page_to_text.items()):
            initial_text = " ".join(text.split(" ")[:self._MAX_WORDS_PER_CHAPTER_CALLBACK])
            batch_text += f"This is the initial text of the page {page}: {initial_text}\n"
            if not (index % self._CALLBACK_INDEX_BATCH):
                tasks.append(
                    _acomplete_chat(
                        system_prompt=(
                            "You are going to receive the beginning text of various pages of a given book. " 
                            f"Your job is to extract the chapter titles of the book following this format: CHAPTER TITLE {self._CHAPTER_PAGE_SEPARATOR} PAGE. "
                            f"A new line for each chapter. If you do not find anything, return {self._NO_CHAPTER_FOUND}\n"
                        ),
                        user_prompt=f"This is the text of the book: \n{batch_text}",
                        model=self._extract_index_model
                    )
                )
                batch_text = ""
        
        try:
            results: List[ChatCompletion] = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in callback -> {e}")
            raise e
        all_chapters: List[str] = []
        for result in results: 
            self._update_price_after_chat_completion(result)
            all_chapters.append(
                result.choices[0].message.content or self._NO_CHAPTER_FOUND
            )
            
        final_chapters: List[str] = []
        for chapters in all_chapters:
            filtered_chapters = [
                chapter.strip() for chapter in chapters.split("\n")
                if self._NO_CHAPTER_FOUND not in chapter
            ]
            final_chapters.extend(filtered_chapters)
        
        return final_chapters

    def _update_price_after_chat_completion(self, completion: ChatCompletion) -> None:

        usage = completion.usage
        
        if usage is None:
            return
        
        input_pricing = 2.5
        output_pricing = 10
        cached_input = 1.25
        
        cached_price = usage.prompt_tokens_details.cached_tokens * cached_input if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens else 0
        input_price = usage.prompt_tokens * input_pricing
        output_price = usage.completion_tokens * output_pricing
        
        self.__price += (cached_price + input_price + output_price) / 1e6
