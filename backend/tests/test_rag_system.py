"""Tests for RAGSystem.query() in backend/rag_system.py"""

import pytest
from unittest.mock import MagicMock, patch


class FakeConfig:
    ANTHROPIC_API_KEY = "fake-key"
    ANTHROPIC_MODEL = "claude-test"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = ":memory:"


@pytest.fixture
def rag(tmp_path):
    """Build a RAGSystem with all external dependencies mocked."""
    with (
        patch("rag_system.DocumentProcessor") as MockDP,
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.SessionManager") as MockSM,
    ):
        from rag_system import RAGSystem

        config = FakeConfig()
        config.CHROMA_PATH = str(tmp_path / "chroma")
        system = RAGSystem(config)

        system._mock_dp = MockDP.return_value
        system._mock_vs = MockVS.return_value
        system._mock_ai = MockAI.return_value
        system._mock_sm = MockSM.return_value

        # Wire up the mocks that were already injected via __init__
        system.document_processor = system._mock_dp
        system.vector_store = system._mock_vs
        system.ai_generator = system._mock_ai
        system.session_manager = system._mock_sm

        # Re-wire search tool to use the mocked vector store
        from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager

        system.tool_manager = ToolManager()
        system.search_tool = CourseSearchTool(system._mock_vs)
        system.tool_manager.register_tool(system.search_tool)
        system.outline_tool = CourseOutlineTool(system._mock_vs)
        system.tool_manager.register_tool(system.outline_tool)

        return system


class TestRAGSystemQuery:
    def test_query_returns_tuple_of_response_and_sources(self, rag):
        rag.ai_generator.generate_response.return_value = "Here is what I found."
        rag.session_manager.get_conversation_history.return_value = None

        result = rag.query("What is RAG?", session_id="sess1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert response == "Here is what I found."
        assert isinstance(sources, list)

    def test_query_passes_tools_to_ai_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("What is RAG?")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_passes_tool_manager_to_ai_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("What is RAG?")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs.get("tool_manager") is rag.tool_manager

    def test_query_includes_conversation_history_when_session_exists(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.session_manager.get_conversation_history.return_value = (
            "User: hi\nAssistant: hello"
        )

        rag.query("follow up question", session_id="sess1")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_query_updates_session_history_after_response(self, rag):
        rag.ai_generator.generate_response.return_value = "final answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("my question", session_id="sess1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "sess1", "my question", "final answer"
        )

    def test_query_returns_sources_from_search_tool(self, rag):
        from vector_store import SearchResults

        rag._mock_vs.search.return_value = SearchResults(
            documents=["content about RAG"],
            metadata=[{"course_title": "RAG 101", "lesson_number": 1}],
            distances=[0.1],
        )
        rag._mock_vs.get_lesson_link.return_value = "https://example.com/lesson"

        # Simulate AIGenerator calling the tool directly
        def side_effect(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query="RAG")
            return "Here is the answer."

        rag.ai_generator.generate_response.side_effect = side_effect

        _, sources = rag.query("What is RAG?", session_id="sess1")
        assert len(sources) == 1
        assert sources[0]["label"] == "RAG 101 - Lesson 1"

    def test_sources_are_reset_after_retrieval(self, rag):
        from vector_store import SearchResults

        rag._mock_vs.search.return_value = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        rag._mock_vs.get_lesson_link.return_value = "https://example.com"

        def side_effect_first(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query="topic")
            return "answer 1"

        rag.ai_generator.generate_response.side_effect = side_effect_first
        rag.query("first question", session_id="s1")

        # Second query without tool calls — sources should be empty
        rag._mock_vs.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        rag.ai_generator.generate_response.side_effect = None
        rag.ai_generator.generate_response.return_value = "answer 2"
        _, sources2 = rag.query("second question", session_id="s1")
        assert sources2 == []

    def test_query_without_session_does_not_update_history(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"

        rag.query("anonymous question")

        rag.session_manager.add_exchange.assert_not_called()

    def test_query_wraps_user_question_in_prompt(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("What are embeddings?")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "What are embeddings?" in call_kwargs["query"]


class TestRAGSystemQueryFailureModes:
    def test_query_propagates_ai_generator_exception(self, rag):
        rag.ai_generator.generate_response.side_effect = RuntimeError("API timeout")
        rag.session_manager.get_conversation_history.return_value = None

        with pytest.raises(RuntimeError, match="API timeout"):
            rag.query("What is RAG?", session_id="sess1")

    def test_content_question_triggers_tool_definitions_in_call(self, rag):
        """Content questions must have tool definitions available so Claude can choose to search."""
        rag.ai_generator.generate_response.return_value = "answer"
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("Explain the concept of vector embeddings from the course")

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs.get("tools"), "tools must be non-empty for content questions"
