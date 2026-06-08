"""Transport AI Agent — intent parsing + direct tool dispatch."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agent.tools import search_next_bus, search_next_hsr, search_next_metro, search_next_train
from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

PARSE_PROMPT = """你是台灣交通查詢意圖解析器。根據使用者的訊息，判斷要呼叫哪個查詢功能，並提取參數。

可用的功能：
1. search_next_metro — 查捷運/輕軌時刻
   參數：station_name (站名), system (TRTC=台北捷運, KRTC=高雄捷運, TYMC=桃園機場捷運, NTDLRT=淡海輕軌, KLRT=高雄輕軌)

2. search_next_bus — 查公車到站時間
   參數：city (Taipei/NewTaipei/Taoyuan/Taichung/Tainan/Kaohsiung), route_name (路線名), stop_name (站牌名，可選)

3. search_next_train — 查台鐵時刻
   參數：from_station (出發站), to_station (到達站), train_date (日期YYYY-MM-DD，可選)

4. search_next_hsr — 查高鐵時刻
   參數：from_station (出發站), to_station (到達站)

判斷規則：
- 提到「捷運」或台北/高雄捷運站名 → search_next_metro, system=TRTC 或 KRTC
- 提到「輕軌」或淡海輕軌站名（淡金北新、濱海義山、濱海沙崙、淡海新市鎮等）→ search_next_metro, system=NTDLRT
- 提到「公車」→ search_next_bus
- 提到「台鐵」「火車」「自強號」「莒光號」「區間車」→ search_next_train
- 提到「高鐵」→ search_next_hsr

注意：
- 紅樹林站同時是台北捷運和淡海輕軌的站。如果目的地或出發站是淡海輕軌站（如濱海義山、淡金北新等），用 NTDLRT。否則預設 TRTC。
- station_name 請去掉「站」字，例如「紅樹林」而非「紅樹林站」。
- **站名請完全保留使用者原文**，不要自行修正或猜測站名。例如使用者說「汶談」就填「汶談」，不要改成「汶水」。系統會自動做模糊比對。

你必須只回傳一個 JSON，不要有任何其他文字：
{"function": "功能名稱", "params": {"參數名": "值"}}

如果無法判斷，回傳：
{"function": "unknown", "message": "需要更多資訊的問題"}"""

SUMMARY_PROMPT = """你是台灣交通查詢助手。以下是查詢結果，請整理成簡潔、適合手機閱讀的繁體中文回覆。

重要規則：
- 不要編造查詢結果中沒有的時間或資訊
- 捷運/輕軌時刻表顯示的「往XX」是終點站方向，中間站都會經過。例如使用者問「往濱海義山」，而結果顯示「往崁頂」，因為濱海義山在崁頂方向的路線上，所以應該回覆該方向的時刻
- 如果查詢結果包含多個方向，選擇使用者問的方向回覆

使用者問題：{question}

查詢結果：
{result}"""


def _build_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=512,
    )


async def _parse_intent(llm: ChatGroq, user_message: str) -> dict:
    """Use LLM to parse user intent into function + params."""
    response = await llm.ainvoke([
        SystemMessage(content=PARSE_PROMPT),
        HumanMessage(content=user_message),
    ])
    text = response.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


async def _summarize(llm: ChatGroq, question: str, result: str) -> str:
    """Use LLM to format the raw tool result into a friendly reply."""
    response = await llm.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT.format(question=question, result=result)),
    ])
    return response.content


TOOL_DISPATCH = {
    "search_next_metro": search_next_metro,
    "search_next_bus": search_next_bus,
    "search_next_train": search_next_train,
    "search_next_hsr": search_next_hsr,
}


async def get_transport_response(user_message: str) -> str:
    """Parse intent → call tool → summarize result."""
    llm = _build_llm()

    try:
        # Step 1: Parse intent
        intent = await _parse_intent(llm, user_message)
        logger.info("Parsed intent: %s", intent)

        func_name = intent.get("function", "unknown")
        params = intent.get("params", {})

        if func_name == "unknown":
            return intent.get("message", "請問您想查詢什麼交通資訊？（捷運、公車、台鐵、高鐵）")

        # Step 2: Execute tool
        tool_fn = TOOL_DISPATCH.get(func_name)
        if not tool_fn:
            return "無法辨識您的查詢類型，請試著說明要查捷運、公車、台鐵或高鐵。"
        raw_result = await tool_fn.ainvoke(params)

        logger.info("Tool result length: %d", len(raw_result))

        # Step 3: Return tool result directly — it's already well-formatted.
        # Skip LLM summary to avoid direction/geographic mistakes.
        return raw_result

    except json.JSONDecodeError:
        logger.exception("Failed to parse intent JSON")
        return "無法理解您的問題，請試著用以下格式詢問：\n• 捷運紅樹林站往象山\n• 307公車到台北車站\n• 台鐵臺北到花蓮"

    except Exception as e:
        logger.exception("Agent failed")
        return f"查詢時發生錯誤，請稍後再試。"
