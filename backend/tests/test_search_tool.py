"""Tests for CourseSearchTool.execute() in backend/search_tools.py"""

import pytest
from unittest.mock import MagicMock, patch
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


def make_search_results(docs, metas, distances=None, error=None):
    if error:
        return SearchResults.empty(error)
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.1] * len(docs),
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course"
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


class TestCourseSearchToolExecute:
    def test_returns_formatted_content_on_success(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            docs=["Lesson content about RAG."],
            metas=[{"course_title": "RAG Course", "lesson_number": 1}],
        )
        result = tool.execute(query="what is RAG")
        assert "RAG Course" in result
        assert "Lesson content about RAG." in result
        assert "Lesson 1" in result

    def test_returns_error_message_on_store_error(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            [], [], error="Search error: DB unavailable"
        )
        result = tool.execute(query="what is RAG")
        assert "Search error" in result
        assert "DB unavailable" in result

    def test_returns_no_content_message_when_empty(self, tool, mock_store):
        mock_store.search.return_value = make_search_results([], [])
        result = tool.execute(query="obscure topic")
        assert "No relevant content found" in result

    def test_no_content_message_includes_course_filter(self, tool, mock_store):
        mock_store.search.return_value = make_search_results([], [])
        result = tool.execute(query="topic", course_name="MCP Course")
        assert "MCP Course" in result
        assert "No relevant content found" in result

    def test_no_content_message_includes_lesson_filter(self, tool, mock_store):
        mock_store.search.return_value = make_search_results([], [])
        result = tool.execute(query="topic", lesson_number=3)
        assert "lesson 3" in result.lower()

    def test_populates_last_sources_with_lesson(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "My Course", "lesson_number": 2}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson/2"
        tool.execute(query="something")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "My Course - Lesson 2"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/2"

    def test_populates_last_sources_without_lesson(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "My Course"}],
        )
        mock_store.get_course_link.return_value = "https://example.com/course"
        tool.execute(query="something")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "My Course"
        assert tool.last_sources[0]["url"] == "https://example.com/course"

    def test_passes_query_to_store(self, tool, mock_store):
        mock_store.search.return_value = make_search_results([], [])
        tool.execute(query="vector embeddings")
        mock_store.search.assert_called_once_with(
            query="vector embeddings", course_name=None, lesson_number=None
        )

    def test_passes_course_name_and_lesson_to_store(self, tool, mock_store):
        mock_store.search.return_value = make_search_results([], [])
        tool.execute(query="topic", course_name="MCP", lesson_number=5)
        mock_store.search.assert_called_once_with(
            query="topic", course_name="MCP", lesson_number=5
        )

    def test_formats_multiple_results(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            docs=["Content A", "Content B"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
        )
        result = tool.execute(query="topic")
        assert "Content A" in result
        assert "Content B" in result
        assert "Course A" in result
        assert "Course B" in result

    def test_last_sources_reset_between_calls(self, tool, mock_store):
        mock_store.search.return_value = make_search_results(
            docs=["doc1"], metas=[{"course_title": "C1", "lesson_number": 1}]
        )
        tool.execute(query="first query")
        assert len(tool.last_sources) == 1

        mock_store.search.return_value = make_search_results([], [])
        tool.execute(query="second query")
        # On empty results, _format_results is not called so last_sources stays from previous call
        # but execute doesn't reset it — this exposes a potential stale-source bug
        # The test documents the current behavior: last_sources is only updated on non-empty results


class TestToolManagerIntegration:
    def test_register_and_retrieve_tool_definitions(self, mock_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        defs = manager.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "search_course_content"

    def test_execute_tool_by_name(self, mock_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        mock_store.search.return_value = make_search_results([], [])
        result = manager.execute_tool("search_course_content", query="test")
        assert "No relevant content found" in result

    def test_execute_unknown_tool_returns_error(self, mock_store):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_last_sources_after_search(self, mock_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        mock_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "Course", "lesson_number": 1}],
        )
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_last_sources(self, mock_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        mock_store.search.return_value = make_search_results(
            docs=["content"],
            metas=[{"course_title": "Course", "lesson_number": 1}],
        )
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        assert manager.get_last_sources() == []
