import re, tiktoken
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