from models.api_response_model import GBBookMeta
from pydantic import Field
from pathlib import Path

class GBBookMetaLocal(GBBookMeta):
    path_to_content:Path = Field(..., description="Path to the local file with the text content")

