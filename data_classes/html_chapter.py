from pydantic import BaseModel
from bs4.element import AttributeValueList

class HtmlChapter(BaseModel):
    content:str
    tag:str
    attrs:dict[str, list[str]]