"""Shared test data and factory helpers."""

from unittest.mock import MagicMock
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic models (mirrors app.py — self-contained for tests)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_ANSWER = "This course covers Python fundamentals."
SAMPLE_SOURCES = ["course1_script.txt", "course2_script.txt"]
SAMPLE_SESSION_ID = "session_1"
SAMPLE_COURSES = ["Python Basics", "Advanced FastAPI"]


# ---------------------------------------------------------------------------
# Mock RAG system factory
# ---------------------------------------------------------------------------


def make_mock_rag(
    answer: str = SAMPLE_ANSWER,
    sources: List[str] = None,
    session_id: str = SAMPLE_SESSION_ID,
    course_titles: List[str] = None,
):
    mock = MagicMock()
    mock.session_manager.create_session.return_value = session_id
    _sources = SAMPLE_SOURCES if sources is None else sources
    _titles = SAMPLE_COURSES if course_titles is None else course_titles
    mock.query.return_value = (answer, _sources)
    mock.get_course_analytics.return_value = {
        "total_courses": len(_titles),
        "course_titles": _titles,
    }
    return mock


# ---------------------------------------------------------------------------
# Test FastAPI app factory (no static file mount)
# ---------------------------------------------------------------------------


def build_test_app(mock_rag=None) -> FastAPI:
    if mock_rag is None:
        mock_rag = make_mock_rag()

    test_app = FastAPI(title="Test RAG App")
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @test_app.get("/")
    async def root():
        return {"status": "ok"}

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app
