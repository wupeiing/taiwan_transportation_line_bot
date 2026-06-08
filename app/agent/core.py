"""Transport AI Agent with tool-calling loop."""

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import search_next_bus, search_next_metro, search_next_train
from app.config import settings

logger = logging.getLogger(__name__)

TOOLS = [search_next_metro, search_next_bus, search_next_train]
TOOL_MAP = {t.name: t for t in TOOLS}

MAX_TOOL_ROUNDS = 5

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=settings.groq_api_key,
    temperature=0.3,
    max_tokens=1024,
)

llm_with_tools = llm.bind_tools(TOOLS)


async def get_transport_response(user_message: str) -> str:
    """Run the agent loop: LLM decides tools -> execute -> LLM summarizes."""

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    for _round in range(MAX_TOOL_ROUNDS):
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content or "抱歉，我無法處理這個請求。"

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info("Tool call: %s(%s)", tool_name, tool_args)

            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result = f"未知的工具：{tool_name}"
            else:
                try:
                    result = await tool_fn.ainvoke(tool_args)
                except Exception as e:
                    logger.exception("Tool %s failed", tool_name)
                    result = f"工具執行失敗：{e}"

            messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    return "查詢步驟過多，請嘗試簡化您的問題。"
