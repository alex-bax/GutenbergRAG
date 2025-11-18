import tiktoken
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
import matplotlib.pyplot as plt
# TODO: Chunking strategies to implement
 
# 1. Fixed size 
# 2. Sentence sized
# 3. LLM based

# TODO: Add overlap and chunk_size to Settings. Since these are hyperparams

def fixed_size_chunks(*, text:str, chunk_size, overlap=100, encoding="cl100k_base") -> list[str]:
    char_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator="\n"
    )

    chunks = char_splitter.split_text(text)

    # chunk_lens = [len(c) for c in chunks]
    # plt.bar(range(len(chunk_lens)), chunk_lens)
    return chunks



def token_chunking(*, txt:str, chunk_size=600, overlap=60, encoding="cl100k_base") -> list[str]:
    token_splitter = TokenTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )

    token_chunks = token_splitter.split_text(txt)
    print(f"\nToken-based: {len(token_chunks)} chunks")
    
    return token_chunks




