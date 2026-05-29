"""Tests for AIGenerator in backend/ai_generator.py"""

import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_id, name, input_dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def make_mock_response(text=None, stop_reason="end_turn", tool_use_blocks=None):
    """Build a mock Anthropic messages response."""
    response = MagicMock()
    response.stop_reason = stop_reason
    content = []
    if tool_use_blocks:
        content.extend(tool_use_blocks)
    if text is not None:
        content.append(make_text_block(text))
    response.content = content
    return response


@pytest.fixture
def generator():
    with patch("anthropic.Anthropic"):
        gen = AIGenerator(api_key="fake-key", model="claude-test")
    gen.client = MagicMock()
    return gen


class TestGenerateResponseDirectAnswer:
    def test_returns_content_on_stop(self, generator):
        generator.client.messages.create.return_value = make_mock_response(
            text="Here is your answer.", stop_reason="end_turn"
        )
        result = generator.generate_response("What is RAG?")
        assert result == "Here is your answer."

    def test_no_tools_not_passed_when_none(self, generator):
        generator.client.messages.create.return_value = make_mock_response(
            text="answer", stop_reason="end_turn"
        )
        generator.generate_response("hello")
        call_kwargs = generator.client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_tools_and_tool_choice_passed_when_provided(self, generator):
        generator.client.messages.create.return_value = make_mock_response(
            text="answer", stop_reason="end_turn"
        )
        tool_defs = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]
        generator.generate_response(
            "question", tools=tool_defs, tool_manager=MagicMock()
        )
        call_kwargs = generator.client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_added_to_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_mock_response(
            text="answer", stop_reason="end_turn"
        )
        generator.generate_response(
            "question", conversation_history="User: hi\nAssistant: hello"
        )
        call_kwargs = generator.client.messages.create.call_args[1]
        system_content = call_kwargs["system"]
        assert "Previous conversation:" in system_content
        assert "User: hi" in system_content


class TestHandleToolExecution:
    def test_tool_call_triggers_handle_tool_execution(self, generator):
        tool_block = make_tool_use_block(
            "call_1", "search_course_content", {"query": "RAG"}
        )
        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )
        second_response = make_mock_response(
            text="Final answer after tool.", stop_reason="end_turn"
        )
        generator.client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Lesson content about RAG."

        result = generator.generate_response(
            "Tell me about RAG",
            tools=[
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer after tool."
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG"
        )

    def test_tool_result_included_in_follow_up_messages(self, generator):
        tool_block = make_tool_use_block(
            "call_abc", "search_course_content", {"query": "embeddings"}
        )
        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )
        second_response = make_mock_response(
            text="Embeddings are...", stop_reason="end_turn"
        )
        generator.client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Embeddings map text to vectors."

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]
        generator.generate_response(
            "Explain embeddings", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Anthropic format: tool results are user messages with type "tool_result" content
        tool_result_messages = [
            m
            for m in messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_messages) == 1
        tool_result = tool_result_messages[0]["content"][0]
        assert tool_result["content"] == "Embeddings map text to vectors."
        assert tool_result["tool_use_id"] == "call_abc"

    def test_assistant_message_content_is_response_content_blocks(self, generator):
        """Assistant message in follow-up contains the raw content blocks from the response."""
        tool_block = make_tool_use_block(
            "call_1", "search_course_content", {"query": "RAG"}
        )
        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )
        second_response = make_mock_response(
            text="Final answer.", stop_reason="end_turn"
        )
        generator.client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some content"

        result = generator.generate_response(
            "question",
            tools=[
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer."

        second_call_kwargs = generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == first_response.content

    def test_tool_arguments_are_passed_from_input(self, generator):
        """Tool input dict is unpacked and passed directly to execute_tool."""
        tool_block = make_tool_use_block(
            "call_x",
            "search_course_content",
            {"query": "MCP", "course_name": "MCP Course", "lesson_number": 2},
        )
        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )
        second_response = make_mock_response(text="Answer.", stop_reason="end_turn")
        generator.client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "content"

        generator.generate_response("question", tool_manager=mock_tool_manager)

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP",
            course_name="MCP Course",
            lesson_number=2,
        )

    def test_tool_call_without_tool_manager_returns_text_content(self, generator):
        """If stop_reason==tool_use but no tool_manager, return text from content[0]."""
        text_block = make_text_block("(thinking...)")
        tool_block = make_tool_use_block(
            "call_1", "search_course_content", {"query": "RAG"}
        )
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [text_block, tool_block]
        generator.client.messages.create.return_value = response
        result = generator.generate_response("question")
        assert result == "(thinking...)"


class TestMultipleToolCalls:
    def test_multiple_sequential_tool_calls(self, generator):
        """Claude can make a second tool call after seeing the first result."""
        tool_block_1 = make_tool_use_block(
            "call_1", "get_course_outline", {"course_name": "Course X"}
        )
        tool_block_2 = make_tool_use_block(
            "call_2", "search_course_content", {"query": "lesson 4 topic"}
        )

        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block_1]
        )
        second_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block_2]
        )
        third_response = make_mock_response(
            text="Final answer after two tools.", stop_reason="end_turn"
        )
        generator.client.messages.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline content.",
            "Related course found.",
        ]

        tools = [
            {"name": "get_course_outline", "description": "Outline", "input_schema": {}}
        ]
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
        tool_block = make_tool_use_block(
            "call_1", "search_course_content", {"query": "RAG"}
        )
        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )
        second_response = make_mock_response(
            text="Final answer.", stop_reason="end_turn"
        )
        generator.client.messages.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some content"

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]
        generator.generate_response(
            "question", tools=tools, tool_manager=mock_tool_manager
        )

        second_call_kwargs = generator.client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_all_tool_results_in_final_request_messages(self, generator):
        """After two tool calls, both tool results appear in messages for the final API call."""
        tool_block_1 = make_tool_use_block(
            "call_1", "get_course_outline", {"course_name": "Course X"}
        )
        tool_block_2 = make_tool_use_block(
            "call_2", "search_course_content", {"query": "topic"}
        )

        first_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block_1]
        )
        second_response = make_mock_response(
            stop_reason="tool_use", tool_use_blocks=[tool_block_2]
        )
        third_response = make_mock_response(text="Done.", stop_reason="end_turn")
        generator.client.messages.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        tools = [
            {"name": "get_course_outline", "description": "Outline", "input_schema": {}}
        ]
        generator.generate_response(
            "complex question", tools=tools, tool_manager=mock_tool_manager
        )

        third_call_kwargs = generator.client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        # Each tool result is a separate user message with type "tool_result" content
        tool_result_messages = [
            m
            for m in messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_messages) == 2
        tool_contents = {}
        for msg in tool_result_messages:
            for content in msg["content"]:
                if content.get("type") == "tool_result":
                    tool_contents[content["tool_use_id"]] = content["content"]
        assert tool_contents["call_1"] == "outline result"
        assert tool_contents["call_2"] == "search result"
