from vector_store_abstract import VectorStore
from settings import Settings
from models.vector_db_model import ContentUploadChunk
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential


class AzSearchVectorStore(VectorStore):
    settings:Settings
    chunk_size:int

    def model_post_init(self, __context):
        # Run after Pydantic validates input and creates the object
        self.name = self.name.strip().title()

        self.search_client = SearchClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                        index_name=self.settings.INDEX_NAME,
                                        credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY)
                                    )

        self.index_client = SearchIndexClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                            credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY))
    
        

    def upsert(*, self, chunks: list[ContentUploadChunk]):
        docs = [chunk.to_dict() for chunk in chunks]
        
        
        for docs in docs:
            self.search_client.upload_documents(docs)


    
