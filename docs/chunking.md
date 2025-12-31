
# Chunking
To ensure the quality of the retrieved context, 2 different approaches were experimented with: 
- Fixed size chunking
- Semantic sized chunking *(current)*

### Fixed size chunking
A simple but naive way to split up the book text. Here done as chunking the text by some hard defined length. \
In this case about every 500 characters with `\n` as separator and an overlap between the chunks of 100 to keep some context from the previous chunk.\
*Example of a fixed size chunk*\
<img src="../imgs/qdrant-fixed-coll1.png" alt="Diagram" height="325" >

**Evaluation results**\
Having plotted the scores from running the [CI golden set](../evals/datasets/gb_ci_pipeline.csv) with DeepEval, it's clear that the chunks retrieved had "varying" relevance.. *yikes*\
<img src="../evals/plots/3112-2025_1424_31-12-2025_1438/Contextual_Relevancy.png" alt="Fixed chunking 500 length evluation chart" height="220" ><img src="../evals/plots/3112-2025_1424_31-12-2025_1438/Contextual_Precision.png" alt="Fixed chunking 500 length evluation chart" height="220" >

On the other hand answer generation went fine:\
<img src="../evals/plots/3112-2025_1424_31-12-2025_1438/Answer_Relevancy.png" alt="Fixed chunking 500 length evluation chart" height="205" ><img src="../evals/plots/3112-2025_1424_31-12-2025_1438/Faithfulness.png" alt="Fixed chunking 500 length evluation chart" height="205" >

From inspecting some of the test cases, we see that at e.g. index 7, with *Which musical instrument does Holmes play?*, the response was *"I dont know based on the given context."* which is the default answer from the prompt when no relevant context was given. Checking the context chunks, it's indeed true, there were no mention of music or instruments.  

**NB:**\
*I've defined the threshold as 0.7, which is subjective. However from experience this was the quality that semeed satisfying*\
*The full evaluation output can be seen at the bottom, under `metricScores`* [here](../evals/3112-2025_1424/.latest_test_run.json)


### Semantic sized chunking
From the poor evaluation score, more work was needed on the retrieval.\
Even with overlap between the chunks, context are easily lost when using fixed chunk sizes.
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

### Experiments and findings *(Work in progress)*
One of the interesting challenges with works in long book form, is how they can be quite implicit and wordy, making such
sections i.e. chunks, harder to use for more explicit who-what-where questions. Or when the answer to a question requires *multi-hopping* combining multiple chunks located at different places in the work and jointly reasoning over them all.

For example in *Sherlock Holmes*, from the eval golden set `gb_gold_med.csv` the question *"Which character hires Holmes to investigate the strange advertisement seeking red-headed men?"* is non-trivial to answer for the system. 

*How semantic chunking fixes this* \
...
citations fuzzy.

It created better results than when using *fixed sized* chunking as seen here:
TODO! *ADD 95p breakpoint eval results*


However the distribution of the chunk lengths were very unenven, as seen in "Alice's Adventure in Wonderland" and "Frankenstein":\
<img src="../stats/index_stats/charts/28-12-2025_2016/Alice&apos;s_Adventures_in_Wonderland.png" alt="Diagram" height="305" > <img src="../stats/index_stats/charts/28-12-2025_2016/Frankenstein;_Or,_The_Modern_Prometheus.png" alt="Diagram" height="305" >


Using 75% percentile dissimilarity yielded more balanced plots:\


**Summary of the semantic vector collection**\
To better understand how the semantic splitting is applied, \
I've made a summary of the collection that after building the vector index. 
It helps show how the chunk sizes are distributed. Here `std` is the standard deviation and `p` is the percentile, so `p90` is "90% percentile". 




Stat summary of 70 percentile threshold:
| Key                          |              Value |
| ---------------------------- | -----------------: |
| config_id_used               |                  3 |
| book_count                   |                  2 |
| total_chunks                 |                632 |
| book_chunk_count_median      |              316.0 |
| book_chunk_count_p90         |              332.8 |
| book_token_mean_median       | 114.97513956646381 |
| book_token_mean_p90          | 124.02824825227582 |
| book_token_std_median        | 153.82843581032046 |
| book_token_std_p90           | 156.35860631611098 |
| book_token_max_median        |             1125.5 |
| book_token_max_p90           |             1151.5 |
| chunk_token_p10              |               13.0 |
| chunk_token_p50              |               60.0 |
| chunk_token_p90              | 310.39999999999986 |
| chunk_token_p99              |  737.1099999999989 |
| pct_books_token_std_gt_p90   |               50.0 |
| pct_books_token_max_gt_2xp90 |                0.0 |
| pct_books_chunk_count_gt_p99 |               50.0 |


Initially, I experimented with using the semantic splitter with its default parameters of 95 percentile dissimilarity as the break point threshold for splitting.  

The disadvantages of having few but very long chunks are:
- Bias: longer chunks can dominate, since they are more "matchable" due their length.
- Cost/latency: With the reranker + generation over large contexts, it gets slower and more expensive.
- Answer quality drift: long chunks can make topics/meaning too "bland", increasing hallucination risk or making 