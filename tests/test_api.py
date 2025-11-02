from typing import Iterator
from fastapi import status, APIRouter
from fastapi.testclient import TestClient
from database import Base
from main import app, get_db
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker, Session
import pytest
from schema import Book

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
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_get_book(client: TestClient):
    # Must insert the expected book before retrieving it
    session: Session = TestingSessionLocal()
    session.add(Book(id=4, authors="Frank Herman", title="Test book", lang="en", slug_key="x"))
    session.commit()
    session.close()

    resp = client.get("/v1/books/4")
    assert resp.status_code == status.HTTP_200_OK, resp.text
    data = resp.json()
    assert data["authors"] == "Frank Herman"
    assert "id" in data


