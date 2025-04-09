import string
import base64
import pymupdf
import uuid


def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError as e:
        raise e


def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from the given text.
    :param text: Input string
    :return: String without punctuation
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def _turn_page_into_image(
    document_path: str, page_number: int, zoom: float = 4.0
) -> str:

    pdf_document = pymupdf.open(document_path)
    page = pdf_document[page_number - 1]
    
    image_path = f"{document_path}_{uuid.uuid4()}_{page_number}.png"

    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    pixmap.save(image_path)
    pdf_document.close()
    return image_path
