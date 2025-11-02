from fastapi import status, APIRouter
from fastapi.testclient import TestClient
from database import Base
from main import app, get_db
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker


# prefix_router = APIRouter(prefix="/v1")
# app.include_router(prefix_router)
client = TestClient(app)

TEST_DB_URL = "sqlite:///:memory:"      # create the db directly in-memory
engine = create_engine(TEST_DB_URL,     # the extra params ensure that all sessions use the same in-mem db
                       connect_args={
                           "check_same_thread": False
                       },
                       poolclass=StaticPool
                    )
TestingSessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


def test_get_book():
    resp = client.get("/v1/books/4")
    assert resp.status_code == status.HTTP_200_OK, resp.text
    #TODO: add assert for json body of Book
    book = resp.json()

    assert book["authors"] == "Frank Herman", resp.text
    assert "id" in book

# Create all tables in our test db
def setup():
    Base.metadata.create_all(bind=engine)

def teardown():
    Base.metadata.drop_all(bind=engine)


