import re, tiktoken


def extract_chapters(*, book_txt: str) -> dict[int, dict[str, str]]:
    """
    Extracts chapters from a book string. Exptects chapters in format, e.g. : CHAPTER 3. <chapter title> \n <content...>
    NB - Overwrites the initial "empty" chapters from table of contents, with the actual chapter content as the book is iterated over.

    Returns a dictionary of the form:
    {
        chapter_number: {
            "title": <chapter title>,
            "content": <chapter content>
        }
    }
    """
    
    # Regex to capture chapter headings 
    pattern = re.compile(r"^CHAPTER\s+(\d+)\.?\s*([^\n]*)", re.MULTILINE)
    matches = list(pattern.finditer(book_txt))
    
    chapters = {}
    
    for chapter_i, match in enumerate(matches):
        chapter_num = int(match.group(1))
        chapter_title = match.group(2).strip()
        
        start_idx = match.end()                                                                     
        end_idx = matches[chapter_i+1].start() if chapter_i+1 < len(matches) else len(book_txt)     # End index is the start of next chapter or the end of text
        
        content = book_txt[start_idx:end_idx].strip()
        
        chapters[chapter_num] = {                                   
            "title": chapter_title,
            "content": content
        }
    
    return chapters


def tiktoken_chunks(*, txt:str, max_tokens=600, overlap=60, encoding="cl100k_base") -> tuple[list[str], list[int]]:
    enc = tiktoken.get_encoding(encoding)
    token_ids = enc.encode(txt)
    step = max_tokens - overlap
    out = []

    for i in range(0, len(token_ids), step):
        out.append(enc.decode(token_ids[i:i+max_tokens]))
    
    return out, token_ids