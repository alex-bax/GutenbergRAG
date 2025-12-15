## 📖 Gutenberg RAG 

An end-to-end Retrieval-Augmented Generation system that embeds, indexes, and queries books from [Project Gutenberg](https://www.gutenberg.org/). \
[Try out the API here](https://gbragfastapi-accyhah2evcnfxev.westeurope-01.azurewebsites.net/docs) \ *NB - Work in progress*

It’s designed to be production-ready and showcase modern vector search, advanced chunking strategies, proper evaluation, structured experiments and monitoring techniques.

### Features
**RAG**
* Semantic search using Qdrant
* Automatic book ingestion pipeline with text cleaning + chunking
* Fast embeddings via Azure OpenAI (text-embedding-3-small)
* RAG response generation using Azure OpenAI GPT models
* LLM-based reranking 
* Evaluation with DeepEval using RAG relevant metrics:
     * Answer generation metrics: * *Answer relevancy*, *Faithfulness*
     * Retrieval metrics: *Context relevance*, *Context precision*
 * Structured outputs with Pydantic classes


**Production and deployment**
* CI automated integration and unit testing with PyTest (FastAPI + async DB)
* CI automated evaluation of the system using DeepEval + gold dataset in CI pipeline
* Deployment with docker 


**Software design**
* Visualisation of experiments
* Interfaces for swapping vector databases, currently supporting Qdrant and Azure AI Search
* Pydantic data classes for strong typing and intellisense

#### Soon
* **Improved quality in retrieval**:
  * Experiment to find better embedding model: Based on [Hugging Face embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard) many better models are available. 
  * Semantic chunking:
    * Dynamic chunk lengths depending the semantic context, ensuring that each chunk is as close as possible to having a single meaning, instead of many
    * Adding "who-what-where" sentence summary or similar to each chunk header with cheap/fast LLM. 
  * Hybrid search integrating with BM25 sparse vector algorithms. 
* **Production / increased safety**
* Monitoring via LangFuse, allowing for
* Guardrails to ensure that e.g. under age users would get inappropiate responses
* **Misc**
* Parallelization of evaluation by using multiple threads 

### Approach
- Ensuring that the answer only uses the retrieved context and not relying on its own training --> Strict designed prompt 
 

### Architecture
- Backend: FastAPI in Azure Web App service
- Storage:
    - Async PostgreSQL DB in Supabase (book metadata such as title, authors, language, etc.)
    - SQLAlchemy as ORM
- Vector DB interface and implementation: Qdrant / Azure AI Search
- LLM: GPT 5 mini with Azure OpenAI

### Automatic deployment with Docker + CI/CD pipelines

### Experiments and findings
From latest experiments it's clear that retrieval needs a 


#### Misc
- Hyperparameter files used are denoted by 'hp' and are found in: `config`
- The CI/CD pipelines are located in `azure-pipelines.yml`

