"""
PDF Resume Parser using PDFMiner.
Extracts raw text from PDF files with layout preservation.
Handles multi-column layouts, tables, and embedded fonts.
"""
import io
import logging
from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import (
    LAParams, LTTextBox, LTTextLine, LTAnno,
    LTFigure, LTPage
)
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter

logger = logging.getLogger(__name__)


class PDFParserError(Exception):
    """Raised when PDF parsing fails."""
    pass


class PDFResumeParser:
    """
    Extracts text from PDF resumes using PDFMiner.
    Handles:
    - Single and multi-column layouts via LAParams tuning
    - Password-protected PDFs (raises explicit error)
    - Corrupted/malformed PDFs with graceful fallback
    - Large files (streaming, not loading entire file into memory)
    """

    # LAParams tuned for resume layouts:
    # - char_margin: distance to consider characters part of same word
    # - word_margin: distance to consider words part of same line
    # - boxes_flow: +1 = purely horizontal, -1 = purely vertical reading order
    DEFAULT_LAPARAMS = LAParams(
        char_margin=2.0,
        word_margin=0.1,
        boxes_flow=0.5,
        detect_vertical=False,
        all_texts=True,
    )

    def __init__(self, laparams: LAParams = None):
        self.laparams = laparams or self.DEFAULT_LAPARAMS

    def extract_text_from_bytes(self, content: bytes, filename: str = "resume.pdf") -> str:
        """
        Extract text from PDF bytes.
        Used when file is received from S3, HTTP upload, or message queue.
        """
        try:
            pdf_file = io.BytesIO(content)
            return self._extract_from_fileobj(pdf_file, filename)
        except Exception as e:
            logger.error(f"Failed to parse PDF bytes [{filename}]: {e}")
            raise PDFParserError(f"Cannot parse PDF '{filename}': {e}") from e

    def extract_text_from_path(self, file_path: Union[str, Path]) -> str:
        """Extract text from a local PDF file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        try:
            with open(path, "rb") as f:
                return self._extract_from_fileobj(f, path.name)
        except PDFParserError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF file [{path}]: {e}")
            raise PDFParserError(f"Cannot parse '{path}': {e}") from e

    def _extract_from_fileobj(self, fileobj, filename: str) -> str:
        """Core extraction logic using PDFMiner streaming interpreter."""
        output = io.StringIO()
        resource_manager = PDFResourceManager(caching=True)
        converter = TextConverter(
            resource_manager,
            output,
            laparams=self.laparams,
        )
        interpreter = PDFPageInterpreter(resource_manager, converter)

        page_count = 0
        try:
            for page in PDFPage.get_pages(
                fileobj,
                pagenos=None,
                maxpages=0,  # No limit
                password=b"",
                caching=True,
                check_extractable=True,
            ):
                interpreter.process_page(page)
                page_count += 1
        except Exception as e:
            if "password" in str(e).lower():
                raise PDFParserError(f"PDF '{filename}' is password-protected")
            logger.warning(f"Partial extraction error on page {page_count} of '{filename}': {e}")
            # Still return whatever we got before the error
        finally:
            converter.close()

        text = output.getvalue()
        output.close()

        if not text.strip():
            logger.warning(f"No text extracted from PDF '{filename}' ({page_count} pages)")

        logger.info(f"Extracted {len(text)} chars from '{filename}' ({page_count} pages)")
        return text

    def extract_pages_text(self, content: bytes) -> list[str]:
        """
        Extract text per page as a list.
        Useful for page-level NLP processing or section boundary detection.
        """
        pages_text = []
        pdf_file = io.BytesIO(content)
        try:
            for page_layout in extract_pages(pdf_file, laparams=self.laparams):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        page_text += element.get_text()
                pages_text.append(page_text)
        except Exception as e:
            raise PDFParserError(f"Page extraction failed: {e}") from e
        return pages_text

    def get_metadata(self, content: bytes) -> dict:
        """Extract PDF metadata (author, creation date, title, etc.)."""
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser as _PDFParser

        metadata = {}
        try:
            pdf_file = io.BytesIO(content)
            parser = _PDFParser(pdf_file)
            doc = PDFDocument(parser)
            if doc.info:
                for info_dict in doc.info:
                    for key, value in info_dict.items():
                        if isinstance(value, bytes):
                            try:
                                metadata[key] = value.decode("utf-8", errors="replace")
                            except Exception:
                                metadata[key] = str(value)
                        else:
                            metadata[key] = value
        except Exception as e:
            logger.debug(f"Could not extract PDF metadata: {e}")
        return metadata
