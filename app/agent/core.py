"""Transport AI Agent — intent parsing + direct tool dispatch."""

import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.tools import TW_TZ, search_next_hsr, search_next_metro, search_next_train
from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

PARSE_PROMPT = """你是台灣交通查詢意圖解析器。根據使用者的訊息，判斷要呼叫哪個查詢功能，並提取參數。

可用的功能：
1. search_next_metro — 查捷運/輕軌時刻
   必填：station_name (出發站名), transport_type ("捷運" 或 "輕軌")
   選填：to_station (到達站), start_time (HH:MM), end_time (HH:MM)

2. search_next_train — 查台鐵時刻
   必填：from_station (出發站), to_station (到達站)
   選填：train_type (車種), train_date (YYYY-MM-DD), start_time (HH:MM), end_time (HH:MM)

   train_type 對照（只在使用者明確提到車種時填入）：
   - 「自強」「自強號」→ "自強"（含太魯閣、普悠瑪、自強、自強3000）
   - 「新自強」「EMU3000」「城際列車」「自強3000」→ "新自強"
   - 「太魯閣」→ "太魯閣"
   - 「普悠瑪」→ "普悠瑪"
   - 「莒光」「莒光號」→ "莒光"
   - 「復興」→ "復興"
   - 「區間」「區間車」→ "區間"
   - 「區間快」→ "區間快"
   - 沒提到車種 → 不填 train_type
   - 車種不屬於上述 → function="unknown", message="請問您要搭乘哪種車種？"

3. search_next_hsr — 查高鐵時刻
   必填：from_station (出發站), to_station (到達站)
   選填：start_time (HH:MM), end_time (HH:MM)

時間參數提取規則（只在使用者明確提到時間時才填入）：
- 「X點後」「X點以後」「從X點開始」→ start_time="HH:MM"
- 「X點前」「X點以前」「X點之前」→ end_time="HH:MM"
- 「X點到Y點」「X點～Y點」→ start_time="HH:MM", end_time="HH:MM"
- 時間格式統一用 24 小時制，例如下午2點 → "14:00"，早上9點半 → "09:30"
- 使用者沒有提到任何時間 → 不填 start_time 和 end_time
- 使用者沒有指定上午/下午時，根據「現在台灣時間」找下一個符合的時間點：
  先當作 AM（00:00–11:59），若該時間已過，改用 PM（+12小時）
  例如現在 10:00，「兩點」→ 02:00 已過 → 14:00；「十一點」→ 11:00 未過 → 11:00

判斷規則：
- 提到「捷運」→ search_next_metro，transport_type="捷運"
- 提到「輕軌」→ search_next_metro，transport_type="輕軌"
- 提到「台鐵」「火車」「自強」「新自強」「太魯閣」「普悠瑪」「莒光」「復興」「區間車」「區間快」→ search_next_train
- 提到「高鐵」→ search_next_hsr

注意：
- 使用者說「從A到B」時，A 是出發站、B 是目的地。捷運/輕軌填 station_name=A, to_station=B；台鐵/高鐵填 from_station=A, to_station=B。
- station_name / from_station 請去掉「站」字，例如「紅樹林」而非「紅樹林站」。
- **站名必須逐字複製使用者原文，絕對不能更動任何一個字**。即使你認為站名有錯，也不可以修正。例如使用者說「淡金北新」就填「淡金北新」，不可以改成其他字。系統會自動做模糊比對。
- **不可依站名判斷交通工具是否合理**。即使站名聽起來像是其他交通工具的站名，也必須完全依照使用者說的交通工具（捷運/輕軌/火車/高鐵）來決定 function，不得拒絕或改變。例如使用者說「淡金北新出發的火車」，即使淡金北新是輕軌站，仍應回傳 search_next_train，站名填「淡金北新」。
- 輕軌/捷運系統名稱（如「淡海輕軌」「安坑輕軌」「高雄輕軌」）不是站名，transport_type 一律填 "輕軌"，不可填系統名稱。
- 若訊息中無法確定出發站（station_name / from_station），回傳 {"function": "unknown", "message": "請問您的出發站是哪裡？"}。「到X的捷運/輕軌」「往X怎麼搭」中，X 是目的地不是出發站，出發站未知，應回傳 unknown。例：「到崁頂的輕軌」「到象山的捷運怎麼搭」皆應回傳 unknown。
- 台鐵/高鐵的 to_station 為必填，若使用者沒提供目的地，回傳 {"function": "unknown", "message": "請問您要前往哪個站？"}。例如「從台北出發的火車」「從高雄搭高鐵」皆因缺少目的地而應回傳 unknown。

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


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=settings.google_api_key,
        temperature=0,
        max_tokens=512,
    )


async def _parse_intent(llm: ChatGoogleGenerativeAI, user_message: str) -> dict:
    """Use LLM to parse user intent into function + params."""
    current_time = datetime.now(TW_TZ).strftime("%H:%M")
    prompt = f"{PARSE_PROMPT}\n\n現在台灣時間：{current_time}"
    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    intent = json.loads(text)
    # Normalize transport_type — guard against LLM hallucinations
    if intent.get("function") == "search_next_metro":
        t = intent.get("params", {}).get("transport_type", "")
        if t not in ("捷運", "輕軌"):
            intent.setdefault("params", {})["transport_type"] = "捷運"
    return intent


async def _summarize(llm: ChatGoogleGenerativeAI, question: str, result: str) -> str:
    """Use LLM to format the raw tool result into a friendly reply."""
    response = await llm.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT.format(question=question, result=result)),
    ])
    return response.content


TOOL_DISPATCH = {
    "search_next_metro": search_next_metro,
    "search_next_train": search_next_train,
    "search_next_hsr": search_next_hsr,
}


async def get_transport_response(user_message: str) -> str:
    """Parse intent → call tool → summarize result."""
    llm = _build_llm()

    try:
        # Step 1: Parse intent
        intent = await _parse_intent(llm, user_message)
        # logger.info("Parsed intent: %s", intent)

        func_name = intent.get("function", "unknown")
        params = intent.get("params", {})

        if func_name == "unknown":
            return intent.get("message", "請問您想查詢什麼交通資訊？（捷運、公車、台鐵、高鐵）")

        # Step 2: Execute tool
        tool_fn = TOOL_DISPATCH.get(func_name)
        if not tool_fn:
            return "無法辨識您的查詢類型，請試著說明要查捷運、公車、台鐵或高鐵。"
        raw_result = await tool_fn.ainvoke(params)

        # logger.info("Tool result length: %d", len(raw_result))

        # Step 3: Return tool result directly — it's already well-formatted.
        # Skip LLM summary to avoid direction/geographic mistakes.
        return raw_result

    except json.JSONDecodeError:
        logger.exception("Failed to parse intent JSON")
        return "無法理解您的問題，請試著用以下格式詢問：\n• 捷運紅樹林站往象山\n• 307公車到台北車站\n• 台鐵臺北到花蓮"

    except Exception as e:
        logger.exception("Agent failed")
        return f"查詢時發生錯誤，請稍後再試。"
