import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **get_course_outline**: Use when the user asks for a course outline, syllabus, table of contents, lesson list, or wants to know what lessons/topics a course covers. It returns the course title, course link, and all lesson numbers with their titles.
- **search_course_content**: Use for questions about specific course content or detailed educational materials.
- You may make multiple sequential tool calls when a query requires it — for example, first get a course outline to find a lesson title, then search for that topic in another course.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline/structure questions**: Use get_course_outline, then present the course title, link, and numbered lesson list
- **Course-specific content questions**: Use search_course_content, then answer
- **Multi-step queries**: Chain tool calls as needed — each call can use results from prior calls
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model

        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params = {
            **self.base_params,
            "messages": messages
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)

        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager, tools=tools)

        return response.choices[0].message.content

    def _handle_tool_execution(self, initial_response, messages: List, tool_manager, tools: Optional[List] = None) -> str:
        """
        Handle tool calls in a loop, allowing Claude to chain multiple tool calls.
        Each iteration appends tool results and re-calls the API with tools still available.
        """
        messages = messages.copy()
        response = initial_response

        while response.choices[0].finish_reason == "tool_calls":
            assistant_message = response.choices[0].message

            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            for tool_call in assistant_message.tool_calls:
                tool_args = json.loads(tool_call.function.arguments)
                tool_result = tool_manager.execute_tool(tool_call.function.name, **tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            api_params = {**self.base_params, "messages": messages}
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**api_params)

        return response.choices[0].message.content
