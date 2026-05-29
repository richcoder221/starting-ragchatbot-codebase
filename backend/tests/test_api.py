import pytest
from httpx import AsyncClient, ASGITransport

from test_helpers import (
    SAMPLE_ANSWER,
    SAMPLE_COURSES,
    SAMPLE_SESSION_ID,
    SAMPLE_SOURCES,
    build_test_app,
    make_mock_rag,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    async def test_returns_200(self, client: AsyncClient):
        response = await client.get("/")
        assert response.status_code == 200

    async def test_returns_ok_status(self, client: AsyncClient):
        response = await client.get("/")
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    async def test_returns_200_with_valid_payload(self, client: AsyncClient):
        response = await client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200

    async def test_response_contains_answer(self, client: AsyncClient):
        response = await client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()
        assert data["answer"] == SAMPLE_ANSWER

    async def test_response_contains_sources(self, client: AsyncClient):
        response = await client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()
        assert data["sources"] == SAMPLE_SOURCES

    async def test_session_id_created_when_absent(self, client: AsyncClient):
        response = await client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()
        assert data["session_id"] == SAMPLE_SESSION_ID

    async def test_provided_session_id_is_preserved(self, client: AsyncClient):
        response = await client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "my-session"},
        )
        data = response.json()
        assert data["session_id"] == "my-session"

    async def test_missing_query_returns_422(self, client: AsyncClient):
        response = await client.post("/api/query", json={})
        assert response.status_code == 422

    async def test_rag_query_called_with_correct_args(self, mock_rag, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.post(
                "/api/query",
                json={"query": "What is Python?", "session_id": "s1"},
            )
        mock_rag.query.assert_called_once_with("What is Python?", "s1")

    async def test_rag_error_returns_500(self):
        failing_rag = make_mock_rag()
        failing_rag.query.side_effect = RuntimeError("DB unavailable")
        failing_app = build_test_app(failing_rag)
        async with AsyncClient(
            transport=ASGITransport(app=failing_app), base_url="http://test"
        ) as ac:
            response = await ac.post("/api/query", json={"query": "fail"})
        assert response.status_code == 500
        assert "DB unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    async def test_returns_200(self, client: AsyncClient):
        response = await client.get("/api/courses")
        assert response.status_code == 200

    async def test_total_courses_count(self, client: AsyncClient):
        response = await client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == len(SAMPLE_COURSES)

    async def test_course_titles_list(self, client: AsyncClient):
        response = await client.get("/api/courses")
        data = response.json()
        assert data["course_titles"] == SAMPLE_COURSES

    async def test_empty_catalog(self):
        empty_rag = make_mock_rag(course_titles=[])
        empty_app = build_test_app(empty_rag)
        async with AsyncClient(
            transport=ASGITransport(app=empty_app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    async def test_analytics_error_returns_500(self):
        failing_rag = make_mock_rag()
        failing_rag.get_course_analytics.side_effect = RuntimeError("Store down")
        failing_app = build_test_app(failing_rag)
        async with AsyncClient(
            transport=ASGITransport(app=failing_app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/courses")
        assert response.status_code == 500
        assert "Store down" in response.json()["detail"]
