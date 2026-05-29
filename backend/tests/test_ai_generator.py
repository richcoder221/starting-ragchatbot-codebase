"""Tests for AIGenerator in backend/ai_generator.py"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator


def make_mock_response(content=None, finish_reason="stop", tool_calls=None):
    """Build a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_tool_call(tool_id, name, args_dict):
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args_dict)
    return tc


@pytest.fixture
def generator():
    with patch("ai_generator.OpenAI") as MockOpenAI:
        gen = AIGenerator(api_key="fake-key", model="deepseek-chat")
        gen.client = MockOpenAI.return_value
        return gen


class TestGenerateResponseDirectAnswer:
    def test_returns_content_on_stop(self, generator):
        generator.client.chat.completions.create.return_value = make_mock_response(
            content="Here is your answer.", finish_reason="stop"
        )
        result = generator.generate_response("What is RAG?")
        assert result == "Here is your answer."

    def test_no_tools_not_passed_when_none(self, generator):
        generator.client.chat.completions.create.return_value = make_mock_response(
            content="answer", finish_reason="stop"
        )
        generator.generate_response("hello")
        call_kwargs = generator.client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_tools_and_tool_choice_passed_when_provided(self, generator):
        generator.client.chat.completions.create.return_value = make_mock_response(
            content="answer", finish_reason="stop"
        )
        tool_defs = [
            {"type": "function", "function": {"name": "search_course_content"}}
        ]
        generator.generate_response(
            "question", tools=tool_defs, tool_manager=MagicMock()
        )
        call_kwargs = generator.client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    def test_conversation_history_added_to_system_prompt(self, generator):
        generator.client.chat.completions.create.return_value = make_mock_response(
            content="answer", finish_reason="stop"
        )
        generator.generate_response(
            "question", conversation_history="User: hi\nAssistant: hello"
        )
        call_kwargs = generator.client.chat.completions.create.call_args[1]
        system_content = call_kwargs["messages"][0]["content"]
        assert "Previous conversation:" in system_content
        assert "User: hi" in system_content


class TestHandleToolExecution:
    def test_tool_call_triggers_handle_tool_execution(self, generator):
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "RAG"})
        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_response = make_mock_response(
            content="Final answer after tool.", finish_reason="stop"
        )
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Lesson content about RAG."

        result = generator.generate_response(
            "Tell me about RAG",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer after tool."
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG"
        )

    def test_tool_result_included_in_follow_up_messages(self, generator):
        tool_call = make_tool_call(
            "call_abc", "search_course_content", {"query": "embeddings"}
        )
        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_response = make_mock_response(
            content="Embeddings are...", finish_reason="stop"
        )
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Embeddings map text to vectors."

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]
        generator.generate_response(
            "Explain embeddings", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = generator.client.chat.completions.create.call_args_list[1][
            1
        ]
        messages = second_call_kwargs["messages"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "Embeddings map text to vectors."
        assert tool_messages[0]["tool_call_id"] == "call_abc"

    def test_assistant_message_with_none_content_does_not_raise(self, generator):
        """
        When finish_reason is 'tool_calls', assistant_message.content is typically None.
        The follow-up API call must not fail because of None content in messages.
        """
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "RAG"})
        # Simulate content=None (typical when finish_reason=="tool_calls")
        first_response = make_mock_response(
            content=None, finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_response = make_mock_response(
            content="Final answer.", finish_reason="stop"
        )
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some content"

        # This should NOT raise even when content is None
        result = generator.generate_response(
            "question",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer."

        # Verify the assistant message sent to the second API call
        second_call_kwargs = generator.client.chat.completions.create.call_args_list[1][
            1
        ]
        messages = second_call_kwargs["messages"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) == 1
        # The content field should not be None — it causes API rejections
        assert assistant_messages[0]["content"] is not None, (
            "content=None in assistant message causes DeepSeek/OpenAI API to reject the request; "
            "it should be converted to an empty string"
        )

    def test_tool_arguments_are_parsed_from_json(self, generator):
        """Tool call arguments arrive as a JSON string and must be parsed before calling execute_tool."""
        tool_call = make_tool_call(
            "call_x",
            "search_course_content",
            {"query": "MCP", "course_name": "MCP Course", "lesson_number": 2},
        )
        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_response = make_mock_response(content="Answer.", finish_reason="stop")
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "content"

        generator.generate_response("question", tool_manager=mock_tool_manager)

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP",
            course_name="MCP Course",
            lesson_number=2,
        )

    def test_tool_call_without_tool_manager_returns_direct_content(self, generator):
        """If finish_reason==tool_calls but no tool_manager, return the message content directly."""
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "RAG"})
        response = make_mock_response(
            content="(thinking...)", finish_reason="tool_calls", tool_calls=[tool_call]
        )
        generator.client.chat.completions.create.return_value = response
        result = generator.generate_response("question")
        assert result == "(thinking...)"


class TestMultipleToolCalls:
    def test_multiple_sequential_tool_calls(self, generator):
        """Claude can make a second tool call after seeing the first result."""
        tool_call_1 = make_tool_call(
            "call_1", "get_course_outline", {"course_name": "Course X"}
        )
        tool_call_2 = make_tool_call(
            "call_2", "search_course_content", {"query": "lesson 4 topic"}
        )

        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call_1]
        )
        second_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call_2]
        )
        third_response = make_mock_response(
            content="Final answer after two tools.", finish_reason="stop"
        )
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline content.",
            "Related course found.",
        ]

        tools = [{"type": "function", "function": {"name": "get_course_outline"}}]
        result = generator.generate_response(
            "Find a course covering the same topic as lesson 4 of Course X",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Final answer after two tools."
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="Course X"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="lesson 4 topic"
        )

    def test_tools_included_in_follow_up_requests(self, generator):
        """Tools are passed to each follow-up API call so Claude can make additional tool calls."""
        tool_call = make_tool_call("call_1", "search_course_content", {"query": "RAG"})
        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_response = make_mock_response(
            content="Final answer.", finish_reason="stop"
        )
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some content"

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]
        generator.generate_response(
            "question", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = generator.client.chat.completions.create.call_args_list[1][
            1
        ]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == "auto"

    def test_all_tool_results_in_final_request_messages(self, generator):
        """After two tool calls, both tool results appear in the messages for the final API call."""
        tool_call_1 = make_tool_call(
            "call_1", "get_course_outline", {"course_name": "Course X"}
        )
        tool_call_2 = make_tool_call(
            "call_2", "search_course_content", {"query": "topic"}
        )

        first_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call_1]
        )
        second_response = make_mock_response(
            finish_reason="tool_calls", tool_calls=[tool_call_2]
        )
        third_response = make_mock_response(content="Done.", finish_reason="stop")
        generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        tools = [{"type": "function", "function": {"name": "get_course_outline"}}]
        generator.generate_response(
            "complex question", tools=tools, tool_manager=mock_tool_manager
        )

        third_call_kwargs = generator.client.chat.completions.create.call_args_list[2][
            1
        ]
        messages = third_call_kwargs["messages"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        tool_contents = {m["tool_call_id"]: m["content"] for m in tool_messages}
        assert tool_contents["call_1"] == "outline result"
        assert tool_contents["call_2"] == "search result"
