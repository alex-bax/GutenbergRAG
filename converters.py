from models.api_response_model import GBBookMeta, BookMetaDataResponse
from models.schema import DBBookMetaData

def gbbookmeta_to_db_obj(gbm: GBBookMeta) -> DBBookMetaData:
    return DBBookMetaData(
        gb_id=gbm.id,
        title=gbm.title,
        authors=gbm.authors_as_str(),
        lang=gbm.languages[0],
    )

def db_obj_to_response(row: DBBookMetaData) -> BookMetaDataResponse:
    return BookMetaDataResponse(
        id=row.id,
        gb_id=row.gb_id,          # important: use row.gb_id, not row.id
        title=row.title,
        authors=row.authors,
        lang=row.lang,
    )