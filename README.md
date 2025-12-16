## 📖 Gutenberg RAG 

An end-to-end Retrieval-Augmented Generation system that embeds, indexes, and queries books from [Project Gutenberg](https://www.gutenberg.org/). \
Try out the [API here](https://gbragfastapi-accyhah2evcnfxev.westeurope-01.azurewebsites.net/docs)  *NB - Work in progress*

It’s designed to be production-ready and showcase modern vector search, advanced chunking strategies, proper evaluation, structured experiments and monitoring techniques.

## Features
#### RAG
* Semantic search with embeddings using Qdrant
* Automatic book ingestion pipeline with text cleaning + chunking
* Fast embeddings via Azure OpenAI (text-embedding-3-small)
* RAG response generation using Azure OpenAI GPT models
* LLM-based reranking 
* Evaluation with DeepEval using RAG relevant metrics:
     * Answer generation metrics: *Answer relevancy*, *Faithfulness*
     * Retrieval metrics: *Context relevance*, *Context precision*
     * Golden answer datasets with varying complexity and to measure hallucinations with trick questions
 * Structured outputs with Pydantic classes

### Architecture
- Backend: FastAPI in Azure Web App service
- Storage:
    - Async PostgreSQL DB hosted on Supabase (book metadata such as title, authors, language, etc.)
    - SQLAlchemy as ORM 
- Vector DB interface and implementation: Qdrant / Azure AI Search
- LLM: GPT 5 mini with Azure OpenAI

#### API / Software 
* Interfaces for easily swapping vector databases, currently supporting Qdrant and Azure AI Search
* API allows for paging either book metadata or vector store 
* Settings, secrets and hyperparameters are handled securely and neatly organised via a Pydantinc singleton
* Built-in ratelimiter for ingestion pipeline, when running large uploads on the embedding model
* Pydantic data classes for strong typing and intellisense

#### Production and deployment
* CI automated:
    * Integration and unit testing with PyTest (FastAPI + async DB)
    * Evaluation of the system using DeepEval + golden dataset
    * All steps must succeed in order to deploy ensuring quality
* CD pushing and deploying to Azure Container Registry and Docker 

#### Expeeriments
* All experiments are timed with a minimal custom timer
* Logs of the results and the hyperparameters are saved after each run
* Visualisation of experiment results
  
### Planned 
#### RAGE / Improving quality in retrieval
  * Experiment with better embedding model: Based on the [Hugging Face embedding leaderboard](https://huggingface.co/spaces/mteb/leaderboard) many better models are available. 
  * Semantic chunking:
    * Use dynamic chunk lengths depending the semantic context, ensuring that each chunk is as close as possible to having a single meaning, instead of many.
    * Add "who-what-where" sentence summary or similar to each chunk header with cheap/fast LLM. 
  * Hybrid search integrating with BM25 sparse vector algorithms.
  * Add halucination metric to evaluation suite
#### Production / increased safety
* Monitoring via [LangFuse](https://langfuse.com/), allowing for tracing the intermediate steps in the answer generation, prompt version control, metrics and even better evaluation.
* Guardrails to ensure that e.g. underage users wouldn't get inappropiate responses. Can be done directly in Azure Foundry, or custom made by adding input and output filters.
#### Other
* Parallelization of evaluation by using multiple threads to speed it up 
* Adding interface for embedding models
* Further API integration tests + test coverage on Azure Devops

### Approach (Work in progress)
- Ensuring that the answer only uses the retrieved context and not relying on its own training --> Strict designed prompt 
 



### Automatic deployment with Docker + CI/CD pipelines


### Experiments and findings
From latest experiments it's clear that retrieval needs a 


#### Misc
- Hyperparameter files used are denoted by 'hp' and are found in: `config`
- The CI/CD pipelines are located in `azure-pipelines.yml`


**Contact**
Alekxander Baxwill - alekx.baxwill@hotmail.com
