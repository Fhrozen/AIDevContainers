import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Tuple
from .text_utils import count_tokens
import logging

logger = logging.getLogger(__name__)    

def get_available_books() -> List[Dict[str, str]]:
    """Get list of available book XML files
    
    Returns:
        List of dicts with keys:
        - value: filename with extension (for internal use)
        - label: display name without extension
    """
    processed_dir = "texts/processed"
    books = []
    logger.info(f"Checking directory: {processed_dir}")
    for file in os.listdir(processed_dir):
        logger.info(f"Found file: {file}")
        if file.endswith('.xml'):
            books.append({
                'value': file,
                'label': file[:-4]  # Remove .xml extension for display
            })
    return books

def get_book_info(xml_path: str) -> Tuple[str, List[Dict]]:
    """Get book title and chapter information from XML file
    
    Returns:
        Tuple containing:
        - Book title (str)
        - List of chapter dicts with keys: id, title, text
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    book_title = root.get('title')
    chapters = []
    
    for chapter in root.findall('chapter'):
        chapter_info = {
            'id': chapter.get('id'),
            'title': chapter.get('title'),
            'text': chapter.text.strip() if chapter.text else ""
        }
        # Remove first line and strip whitespace
        chapter_info['text'] = chapter_info['text'][chapter_info['text'].find("\n") + 1:].strip()
        chapters.append(chapter_info)
        
    return book_title, chapters

def get_chapter_text(xml_path: str, chapter_id: str) -> str:
    """Get text content for a specific chapter"""
    _, chapters = get_book_info(xml_path)
    for chapter in chapters:
        if chapter['id'] == chapter_id:
            return chapter['text']
    return ""

def get_book_chapters(xml_path: str) -> List[Dict]:
    """Get list of chapters with id and title for dropdown"""
    _, chapters = get_book_info(xml_path)
    return [{'id': ch['id'], 'title': ch['title']} for ch in chapters]
