
# Chunking
To ensure the quality of the retrieved context, 2 different approaches were experimented with: 
- Fixed size chunking
- Semantic sized chunking *(current)*

### Fixed size chunking
A simple but naive way to split up the book text. Here done by chunking by some hard defined length. \
In this case about every 500 characters with `\n` as separator and an overlap between the chunks of 100.\
*Example of a fixed size chunk*\
<img src="../imgs/qdrant-fixed-coll1.png" alt="Diagram" height="325" >

*Evaluation results*
TODO! ADD RESULT CHARTS - fixed size chunking



### Semantic sized chunking
Even with overlap between the chunks, context are easily lost when using fixed chunk sizes.\
With semantic chunking, chunks are split based on their meaning, in turn making each chunk more relevant.\
This produces chunks with varying lengths, and requires use of an embedding model while building the collection.\
This implementation uses a custom made splitter, with the [Semantic splitter by LlamaIndex](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/semantic_splitter/) as the base.

In brief, the splitter works roughly by:
1. Split the document into small base units (often sentences).
2. Make embedding of each sentences.
3. For every sentence, compute semantic dissimilarity between it and its adjacent sentence (using cosine distance).
4. Collect all these distances and make a distribution
4. Define a `cutoff` based on the distribution and the breakpoint percentile threshold parameter.
5. Loop over all sentences, and insert a breakpoint only when the dissimilarity is > `cutoff`.

**Example of how the threshold is used:**\
Say we're given the sentences:
```markdown
S1: Holmes lit his pipe.
S2: He considered the evidence carefully.
S3: The fog lay thick over Baker Street.
S4: Meanwhile, in Paris, the minister resigned.
```
And their distances are
```markdown
S1–S2: 0.06
S2–S3: 0.08
S3–S4: 0.42   ← semantic jump
```
So if the cutoff is `0.3`:
* Sentences 1–3 → one chunk
* Sentence 4 → new chunk



TODO! Add collection fingerprint summary




Using 75% percentile dissimilarity yielded more balanced plots:\
!TODO ADD ALICE AND FRANKENSTEIN DISTR PLOTS

**Summary of the semantic vector collection**
To better understand how the semantic splitting is applied, \
I make a summary of the collection that after building the vector index. 
It helps show how the chunk sizes are distributed. Here `std` is the standard deviation and `p` is the percentile, so `p90` is "90% percentile". 

TODO: redo this
| Metric                              | Value        |
|-------------------------------------|--------------|
| book_count                          | 2            |
| total_chunks                        | 4            |
| book_chunk_count_median             | 2.0          |
| book_chunk_count_p90                | 2.0          |
| book_token_mean_median              | 116.75       |
| book_token_mean_p90                 | 120.95       |
| book_token_std_median               | 45.6083873865|
| book_token_std_p90                  | 47.5882863739|
| book_token_max_median               | 149.0        |
| book_token_max_p90                  | 154.6        |
| chunk_token_p10                     | 83.1         |
| chunk_token_p50                     | 115.0        |
| chunk_token_p90                     | 151.8        |
| chunk_token_p99                     | 155.58       |
| pct_books_token_std_gt_p90          | 50.0         |
| pct_books_token_max_gt_2xp90        | 0.0          |
| pct_books_chunk_count_gt_p99        | 0.0          |

Initially, I experimented with using the semantic splitter with its default parameters of 95 percentile dissimilarity as the break point threshold for splitting.  

The disadvantages of having few but very long chunks are:
- Bias: longer chunks can dominate, since they are more "matchable" due their length.
- Cost/latency: With the reranker + generation over large contexts, it gets slower and more expensive.
- Answer quality drift: long chunks can make topics/meaning too "bland", increasing hallucination risk or making citations fuzzy.

It created better results than when using *fixed sized* chunking as seen here:
TODO! *ADD 95p breakpoint eval results*


However the distribution of the chunk lengths were very unenven, as seen in "Alice's Adventure in Wonderland" and "Frankenstein":\
<img src="../stats/index_stats/charts/28-12-2025_2016/Alice&apos;s_Adventures_in_Wonderland.png" alt="Diagram" height="305" > <img src="../stats/index_stats/charts/28-12-2025_2016/Frankenstein;_Or,_The_Modern_Prometheus.png" alt="Diagram" height="305" >