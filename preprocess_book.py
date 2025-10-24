import re, unicodedata
from openai import AzureOpenAI
from bs4 import BeautifulSoup, ResultSet, Tag
from pathlib import Path
from data_classes.html_chapter import HtmlChapter


# def extract_chapters(*, book_txt: str) -> dict[int, dict[str, str]]:
#     """
#     Extracts chapters from a book string. Exptects chapters in format, e.g. : CHAPTER 3. <chapter title> \n <content...>
#     NB - Overwrites the initial "empty" chapters from table of contents, with the actual chapter content as the book is iterated over.

#     Returns a dictionary of the form:
#     {
#         chapter_number: {
#             "title": <chapter title>,
#             "content": <chapter content>
#         }
#     }
#     """
    
#     # Regex to capture chapter headings 
#     pattern = re.compile(r"^CHAPTER\s+(\d+)\.?\s*([^\n]*)", re.MULTILINE)
#     matches = list(pattern.finditer(book_txt))
    
#     chapters = {}
    
#     for chapter_i, match in enumerate(matches):
#         chapter_num = int(match.group(1))
#         chapter_title = match.group(2).strip()
        
#         start_idx = match.end()                                                                     
#         end_idx = matches[chapter_i+1].start() if chapter_i+1 < len(matches) else len(book_txt)     # End index is the start of next chapter or the end of text
        
#         content = book_txt[start_idx:end_idx].strip()
        
#         chapters[chapter_num] = {                                   
#             "chapter_title": chapter_title,
#             "content": content
#         }
    
#     return chapters


def extract_html(*, html_book:str) -> dict[int, HtmlChapter]:
    bs = BeautifulSoup(markup=html_book, features="lxml")
    chapters:ResultSet[Tag] = bs.find_all(attrs={"class": "chapter"})

    if len(chapters) == 0:
        return {}

    extr_html_chs = {}
    for i, ch in enumerate(chapters, start=0):
        txt = ch.get_text(separator=" ", strip=True)
        extr_html_chs[i] = HtmlChapter(content=txt, tag=ch.name, attrs=ch.attrs)        # type:ignore ; 0th chapter is often the contents table or similar

    return extr_html_chs


# def _strip_gutenberg_header_footer(*, book:str) -> str:
#     start = re.search(r"^CHAPTER 1\.", book, re.M)
#     end = re.search(r"End of the Project Gutenberg EBook of", book)

#     book = book[start.start(): end.start()] if start and end else book
#     return book.strip()


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
    book_p = r"C:\Users\alext\Documents\Code\RAG\mobyRag\books\the-enchanted-april_von-arnim-elizabeth_16389_en.html"
    book_str = ""
    with open(book_p, 'r', encoding='utf-8') as f:
        book_str = f.read()
    
    html_d = extract_html(html_book=book_str)
    for ch_num, ch in html_d.items():
        print(len(ch.content), ch.content[:50])

    # print(make_slug_book_key("Moby Dick", 42, "Herman Melville", 1851, "gutenberg", "en"))
    # print(make_slug_book_key("The Odyssey", gutenberg_id=42,))
