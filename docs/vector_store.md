### Vector Store

I've built an interface that allows for switching between vectore stores, currently Qdrant and Azure AI Search are supported, and Qdrant is used in the  deployed version.

All vector collections use *HNSW* (graph-based approximate nearest neighbor) as the search algorithm.\
Cosine distance is used to calculate the similarity, and uses a vector dimension of 1536 (requried by the embedding model `text-embedding-3-small`).\
Remaining configuration parameters can be found in the [Qdrant configuration file](../config/qdrant_collection_config.json)

Additional metadata fields for each chunk are:
- content (actual text content)
- uuid_str
- book_id (the Gutenberg ID that the chunk is from)
- chunk_nr (chunk index, i.e. if chunk_nr is 3 it's the 3rd chunk in the book)
- book_name


