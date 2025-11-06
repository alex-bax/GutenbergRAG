from typing import Iterator, Callable, Any
from fastapi import status, APIRouter
from fastapi.testclient import TestClient
from db.schema import DBBookMetaData
from db.database import Base
from main import app, get_db
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker, Session
import pytest
# prefix_router = APIRouter(prefix="/v1")
# app.include_router(prefix_router)
# client = TestClient(app)

VER_PREFIX = "v1"
TEST_DB_URL = "sqlite:///:memory:"      # create the db directly in-memory
engine = create_engine(TEST_DB_URL,     # the extra params ensure that all sessions use the same in-mem db
                       connect_args={
                           "check_same_thread": False
                       },
                       poolclass=StaticPool
                    )
TestingSessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

@pytest.fixture
def book_factory(db_session: Session) -> Iterator[Callable[..., DBBookMetaData]]: 
    """
    Create and persist Book rows with having defaults.
    Returns the callable (i.e. function that creates the Book)
    Usage:
        book = book_factory(title="Custom")
    """
    created: list[DBBookMetaData] = []

    def _create(**overrides: Any) -> DBBookMetaData:
        defaults: dict[str, Any] = {
            "title": "Test book",
            "gb_id":42,
            "authors": "Frank Herman",
            "lang": "en",
        }
        merged = defaults | overrides
        obj = DBBookMetaData(**merged)
        
        db_session.add(obj)
        db_session.commit()   # commit so the app (same engine) can read it
        db_session.refresh(obj)
        created.append(obj)
        return obj
    
    # yield the factory back to the calling test 
    yield _create

    # teardown: delete everything this factory created in the test
    for obj in created:
        try:
            db_session.delete(obj)
        except Exception:
            pass        # ignore if already gone or constrained differently
    db_session.commit()


@pytest.fixture(scope="session", autouse=True)
def create_test_schema():
    # Create all tables once for the session
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    # Fresh session per test
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session:Session):
    # Override BEFORE creating TestClient
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


def test_get_book(client: TestClient, book_factory):
    # Must insert the expected book before getting it
    book = book_factory(id=4)

    resp = client.get(f"/{VER_PREFIX}/books/{book.id}")
    assert resp.status_code == status.HTTP_200_OK, resp.text
    data = resp.json()["data"]
    assert data["authors"] == "Frank Herman"
    assert "id" in data

 
moby = {"title":"Moby Dick", "authors":"Herman Melville"}
@pytest.mark.parametrize("queries, expected", [(f"authors=Herman Melville&title=Moby Dick", [moby]),
                                                (f"title=moby", [moby]),
                                                (f"authors=melvil", [moby]),
                                                (f"title=Unknown book", []),
                                                (f"title=Frankenstein&authors=HC Andersen", []),
                                            ])
def test_search_book(queries, expected, client: TestClient, book_factory):
    title = "Moby Dick"
    book = book_factory(id=3, title=title, authors="Herman Melville")

    url = f"/{VER_PREFIX}/books/search?{queries}"
    resp = client.get(url)

    assert resp.status_code == status.HTTP_200_OK
    books_found = resp.json()["data"]
    assert len(books_found) == len(expected), len(expected)
    if len(expected) > 0:
        assert books_found[0]["title"] == expected[0]["title"], title

 
@pytest.mark.parametrize("queries, expected", [("title=Unknown book", []),
                                                ("title=Frankenstein&authors=HC Andersen", []),
                                                ("authors=No", status.HTTP_422_UNPROCESSABLE_CONTENT )])
def test_negative_search_book(queries, expected, client:TestClient, book_factory):
    title = "Moby Dick"
    book = book_factory(id=3, title=title, authors="Herman Melville")

    url = f"/{VER_PREFIX}/books/search?{queries}"

    resp = client.get(url)

    if isinstance(expected, list):
        books_found:list = resp.json()["data"]
        assert len(books_found) == len(expected)
    elif isinstance(expected, int):
        assert resp.status_code == expected
    

# TODO test show show_gutenberg_book
# TODO test upload_book




