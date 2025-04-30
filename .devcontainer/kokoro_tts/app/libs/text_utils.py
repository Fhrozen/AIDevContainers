import tiktoken

def normalize_text(text: str) -> str:
    """Normalize text for TTS processing"""
    if not text:
        return ""
    # Basic normalization - can be expanded based on needs
    return text.strip()

def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Break text into chunks at natural boundaries"""
    chunks = []
    current_chunk = ""
    
    # Split on sentence boundaries first
    sentences = text.replace(".", ".|").replace("!", "!|").replace("?", "?|").replace(";", ";|").split("|")
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # If sentence is already too long, break on commas
        if len(sentence) > max_chars:
            parts = sentence.split(",")
            for part in parts:
                if len(current_chunk) + len(part) <= max_chars:
                    current_chunk += part + ","
                else:
                    # If part is still too long, break on whitespace
                    if len(part) > max_chars:
                        words = part.split()
                        for word in words:
                            if len(current_chunk) + len(word) > max_chars:
                                chunks.append(current_chunk.strip())
                                current_chunk = word + " "
                            else:
                                current_chunk += word + " "
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = part + ","
        else:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
