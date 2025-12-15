## 📖 Gutenberg RAG 

A Retrieval-Augmented Generation system built on Project Gutenberg books.
[Access it here](https://gbragfastapi-accyhah2evcnfxev.westeurope-01.azurewebsites.net/docs)
*NB - Work in progress*

**Overview**
Gutenberg RAG is an end-to-end RAG pipeline that ingests, embeds, indexes, and queries public-domain books from [Project Gutenberg](https://www.gutenberg.org/). 
It’s designed to be production-ready and showcase modern vector search, text chunking, evaluation, and monitoring techniques.


**Features**
* Semantic & hybrid search using Qdrant
* Book ingestion pipeline with text cleaning + chunking
* Fast embeddings via Azure OpenAI (text-embedding-3-small)
* RAG response generation using Azure OpenAI GPT models
* Integration testing with PyTest (FastAPI + DB)
* (Soon) Automated evaluation using DeepEval + gold data in CI pipeline
* (Future) Monitoring planned via LangFuse


**Architecture**
- Backend: FastAPI in Azure Web App service
- Storage: PostgreSQL DB in Supabase (book metadata such as title, authors, language, etc.)
- Vector DB interface and implementation: Qdrant / Azure AI Search
- LLM: GPT 5 mini with Azure OpenAI

