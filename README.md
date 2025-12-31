<img src="./imgs/GBRAGLogo.png" alt="Diagram" height="125" >

___

An end-to-end Retrieval-Augmented Generation system that embeds, indexes, and queries books from [Project Gutenberg](https://www.gutenberg.org/). \
See the OpenAPI specification and try out the API itself [here](https://gbragfastapi-accyhah2evcnfxev.westeurope-01.azurewebsites.net/docs) *(NB - Initial load can be slow due to cold booting of the container)*

It’s designed to be production-ready and showcase modern vector search, advanced chunking strategies, proper evaluation, structured experiments and monitoring techniques.

- The `/books/` route query the metadata Postgres DB, fetch metadata of books, either individually, in chunks through paginations or all books at a time. Addtionally `/books/gutenberg/` allows for exploring books in Project Gutenberg without leaving the API.
- The `/index/` route interacts with the vector index/collection. It allows for searching chunks from text, Delete and upload new books by just giving its Gutenberg ID, in turn using the **ingestion** pipeline.  
- The `/query/` route which uses the **retrieval** workflow to answer any text based query, with citations.  

## Features
### 🔍 RAG
* Semantic search with embeddings using Qdrant
* Semantic chunking, splitting chunks based on their meaning
* Automatic book ingestion pipeline with text preprocessing + chunking to either populate or upload books from Gutenberg
* Fast embeddings via Azure OpenAI (text-embedding-3-small) with rate limiting
* RAG response generation using Azure OpenAI GPT models
* LLM-based reranking 
* Evaluation with DeepEval using RAG relevant metrics:
     * Answer generation metrics: *Answer relevancy*, *Faithfulness*
     * Retrieval metrics: *Context relevance*, *Context precision*
     * Golden answer datasets with varying complexity
 * Structured outputs with Pydantic classes

### 🧱 Architecture
- Backend: FastAPI in Azure Web App service
- Storage:
    - Async PostgreSQL DB hosted on Supabase (book metadata such as title, authors, language, etc.)
    - SQLAlchemy as ORM 
- Vector DB interface and implementation: **Qdrant** / Azure AI Search
- LLM: GPT-5-mini with Azure OpenAI, using a custom lenient guardrail filter on Azure Foundry 

### 🛜 API / Software 
* Interfaces for easily swapping vector databases, currently supporting Qdrant and Azure AI Search
* API uses paging for either book metadata or vector store, allowing for memory safe browsing
* Settings, secrets and hyperparameters are handled securely and neatly organised via a Pydantinc Settings singleton
* Built-in ratelimiter for ingestion pipeline, ensuring safely running large uploads 
* Pydantic data classes for strong typing and intellisense

### 📦 Production and deployment
* CI automated:
    * Integration and unit testing with PyTest (FastAPI + async DB)
    * Evaluation of the system using DeepEval + golden dataset
    * All steps must succeed in order to deploy 
* CD pushing and deploying to Azure Container Registry and Docker 

### 📈 Monitoring (Soon)
* Prometheus + Grafana

### 🔒 Simple Authentication (Soon)
* API key auth


___
### ℹ️ Links to more detailed documentation
- [Chunking and chunking experiments](docs/chunking.md)
- [Details on vector store](docs/vector_store.md)
- [Testing approach](docs/testing.md)
- [CI/CD pipelines](docs/deploying.md)


## Ingestion 
The ingestion pipeline is triggered when the API is booting up for the first time, to populate the vector collection *or* it's triggered when a user is uploading a single book. 

A default list of 10 different types of books/works are used for populating the vector DB, and be found under the attribute `default_ids_used` in the [configuration file](config/hp-sem-ch.json).

Populating the vector collection is a long running operation and requires a lot of calls to both the embedding model and the Qdrant client. In order to upload large and high volume of books smoothly without errors, **rate limiting** with *pyrate limiter* and **batching** has been implemented. 

In this latest version the rate limiter is used in the [semantic chunking](docs/chunking.md) step and the embedding of the chunks. Since the embedding model only accepts string (chunks) up to a certain size, all chunks are split into batches based on this "max token threshold" which all is specified in the [config files.](/config/hp-sem70p-ch.json)\
The entire ingestion pipeline is seen here:\
<img src="./imgs/GBRAG-Ingestion.png" alt="Diagram" height="625" >

After populating the vector collection a statistics summary is made.\
You can find the latest the summmary produced and a brief explanation in the chunking docs [here](docs/chunking.md)

#### Database tables
The simple Entity-Relationship diagram depicts the tables used for storing both the \
**metadata of books** in the vector collection and their **chunking stats**:\
<img src="./imgs/mermaid-diagram-GBRAG.png" alt="Entity Relationship Diagram" height="425" >

The `book_metadata` is being populated by the meta data from Gutenberg.\
The `book_chunk_stats` shows relevant stats related to how the individual book was chunked by the semantic chunker, and all these chunk stats gives insight into how the vector collection is distributed overall.




## Retrieval 
<img src="./imgs/GBRAG-Retrieval.png" alt="Diagram" height="525" >


### Automatic deployment with Docker + CI/CD pipelines
For further details, the entire see `Dockerfile` and the CI/CD pipelines specified in `azure-pipelines.yml`

<img src="./imgs/screenshots/GB-cicd-steps.png" alt="Diagram" style="max-width: 750px; width: 100%;">
<img src="./imgs/screenshots/GB-cicd.png" alt="Diagram" style="max-width: 750px; width: 100%;">


### Experiments and findings *(Work in progress)*
One of the interesting challenges with works in long book form, is how they can be quite implicit and wordy, making such
sections i.e. chunks, harder to use for more explicit who-what-where questions. Or when the answer to a question requires *multi-hopping* combining multiple chunks located at different places in the work and jointly reasoning over them all.

For example in *Sherlock Holmes*, from the eval golden set `gb_gold_med.csv` the question *"Which character hires Holmes to investigate the strange advertisement seeking red-headed men?"* is non-trivial to answer for the system. 

*How semantic chunking fixes this* \
...

## 📋 Planned features
#### RAG: Improving quality in retrieval
  
  * Hybrid search integrating with BM25 sparse vector algorithms.
  * Improved semantic context in the chunks:
    * More context in chunks: Add "who-what-where" sentence summary or similar to each chunk header with cheap LLM. 
    * Experiment with other semantic chunkers such as *Statistical chunking* or *Cumulative Semantic chunking*
  * Experiment with better embedding models: Based on the [Hugging Face embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard) many better models are available.  
  * Add halucination metric to evaluation suite
#### Production and increased safety
* Monitoring via [LangFuse](https://langfuse.com/), allowing for tracing the intermediate steps in the answer generation, prompt version control, metrics and even better evaluation.
* Guardrails to ensure that e.g. underage users wouldn't get inappropiate responses. Can be done directly in Azure Foundry, or custom made by adding input and output filters.
#### Other
* Adding interface for embedding models to also make them easily swapable
* Further API integration tests + test coverage on Azure Devops


### Misc
- Hyperparameter json files used are prefixed by 'hp' and are found in the folder `config`.
- Golden datasets for evaluation are located in folder `evals/datasets`

____

**Contact** \
Alekxander Baxwill - alekx.baxwill@hotmail.com
