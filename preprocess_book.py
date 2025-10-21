import re, unicodedata
from openai import AzureOpenAI

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
            "chapter_title": chapter_title,
            "content": content
        }
    
    return chapters



def create_embeddings(*, embed_client:AzureOpenAI, model_deployed:str, texts:list[str]) -> list[list[float]]:
    resp = embed_client.embeddings.create(
        input=texts,
        model=model_deployed
    )

    embeddings = [emb_obj.embedding for emb_obj in resp.data]
    return embeddings


def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"[-\s]+", "-", s)
        return s

def make_slug_book_key(title: str, gutenberg_id:int, author: str|None=None, year: int|None=None, source: str|None=None, lang: str|None=None):
    parts = [_norm(title)]
    if author: parts.append(_norm(author))
    if gutenberg_id: parts.append(f"{gutenberg_id}")
    if year:   parts.append(str(year))
    if source: parts.append(_norm(source))        # e.g., "gutenberg"
    if lang:   parts.append(_norm(lang))          # e.g., "en"
    return "_".join(parts)



if __name__ == "__main__":

    print(make_slug_book_key("Moby Dick", 42, "Herman Melville", 1851, "gutenberg", "en"))
    print(make_slug_book_key("Pride and Prejudice", 42, "Jane Austen", 1813))
    print(make_slug_book_key("Don Quixote", gutenberg_id=42, source="project-gutenberg", lang="es"))
    print(make_slug_book_key("The Odyssey", gutenberg_id=42,))
