from typing import Iterator, Callable, Any
from fastapi import status, APIRouter
from fastapi.testclient import TestClient
from db.database import Base
from main import app, get_db
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker, Session
import pytest
from db.schema import DBBook

# prefix_router = APIRouter(prefix="/v1")
# app.include_router(prefix_router)
# client = TestClient(app)

TEST_DB_URL = "sqlite:///:memory:"      # create the db directly in-memory
engine = create_engine(TEST_DB_URL,     # the extra params ensure that all sessions use the same in-mem db
                       connect_args={
                           "check_same_thread": False
                       },
                       poolclass=StaticPool
                    )
TestingSessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

@pytest.fixture
def book_factory(db_session: Session) -> Callable[..., DBBook]: 
    """
    Create and persist Book rows with having defaults.
    Returns the callable (i.e. function that creates the Book)
    Usage:
        book = book_factory(title="Custom")
    """
    created: list[DBBook] = []

    def _create(**overrides: Any) -> DBBook:
        defaults: dict[str, Any] = {
            "title": "Test book",
            "authors": "Frank Herman",
            "lang": "en",
            "slug_key": "x",
        }
        merged = defaults | overrides
        obj = DBBook(**merged)
        
        db_session.add(obj)
        db_session.commit()   # commit so the app (same engine) can read it
        db_session.refresh(obj)
        created.append(obj)
        return obj
    
    return _create


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

    resp = client.get(f"/v1/books/{book.id}")
    assert resp.status_code == status.HTTP_200_OK, resp.text
    data = resp.json()
    assert data["authors"] == "Frank Herman"
    assert "id" in data




