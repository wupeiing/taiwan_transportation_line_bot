"""Tests for LLM intent parsing — verifies the prompt correctly extracts function + params."""
from datetime import datetime
from unittest.mock import patch

import pytest
import logging

from app.agent.core import _parse_intent
from app.agent.tools import TW_TZ

logger = logging.getLogger(__name__)


# Metro related (LRT, MRT)
@pytest.mark.asyncio
async def test_light_rail_intent_time_range(llm):
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "兩點到五點從紅樹林到崁頂的輕軌")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "紅樹林" in intent["params"]["station_name"]
    assert "崁頂" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"
    assert intent["params"]["end_time"] == "17:00"


@pytest.mark.asyncio
async def test_light_rail_intent(llm):
    frozen = datetime(2025, 6, 15, 9, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "10點從紅樹林到崁頂的輕軌")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "紅樹林" in intent["params"]["station_name"]
    assert "崁頂" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "10:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_light_rail_intent_no_destination(llm):
    intent = await _parse_intent(llm, "早上七點從漁人碼頭出發的輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "漁人碼頭" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert intent["params"]["start_time"] == "07:00"


@pytest.mark.asyncio
async def test_lrt_no_destination_no_start_time(llm):
    intent = await _parse_intent(llm, "淡金北新 下一班輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "淡金北新" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]


@pytest.mark.asyncio
async def test_lrt_incorrect_station_name(llm):
    intent = await _parse_intent(llm, "下一個淡金北新出發的火車")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert "淡金北新" in intent["params"]["from_station"]


@pytest.mark.asyncio
async def test_mrt_intent(llm):
    intent = await _parse_intent(llm, "從忠孝復興出發的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "忠孝復興" in intent["params"]["station_name"]


@pytest.mark.asyncio
async def test_unknown_intent(llm):
    intent = await _parse_intent(llm, "今天天氣如何")
    assert intent["function"] == "unknown"


# ── 淡海輕軌 (NTDLRT) ─────────────────────────────────

@pytest.mark.asyncio
async def test_ntdlrt_from_to_with_time(llm):
    """下午明確，不需 mock"""
    intent = await _parse_intent(llm, "下午兩點從紅樹林到崁頂的輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "紅樹林" in intent["params"]["station_name"]
    assert "崁頂" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ntdlrt_from_to_no_time(llm):
    intent = await _parse_intent(llm, "從紅樹林到崁頂的淡海輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "紅樹林" in intent["params"]["station_name"]
    assert "崁頂" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ntdlrt_from_only_with_time(llm):
    """「三點」在 10:00 context → 03:00 已過 → PM 15:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "三點從淡金北新出發的輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "淡金北新" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert intent["params"]["start_time"] == "15:00"


@pytest.mark.asyncio
async def test_ntdlrt_from_only_no_time(llm):
    intent = await _parse_intent(llm, "淡水漁人碼頭 下一班淡海輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "漁人碼頭" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert "start_time" not in intent["params"]


# ── 安坑輕軌 (NTALRT) ─────────────────────────────────

@pytest.mark.asyncio
async def test_ntalrt_from_to_with_time(llm):
    """早上明確，不需 mock"""
    intent = await _parse_intent(llm, "早上九點從十四張搭安坑輕軌到安康")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "十四張" in intent["params"]["station_name"]
    assert "安康" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "09:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ntalrt_from_to_no_time(llm):
    # 避免「安坑」被 LLM 誤解為到達站，改用明確不同的目的站
    intent = await _parse_intent(llm, "從十四張搭輕軌到景文科大")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "十四張" in intent["params"]["station_name"]
    assert "景文科大" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ntalrt_from_only(llm):
    intent = await _parse_intent(llm, "景文科大 下一班安坑輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "輕軌"
    assert "景文科大" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]


# ── 邊界情況 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_lrt_only_destination(llm):
    """只給目的站，沒給出發站 — 應回傳 unknown 並提示使用者提供出發站"""
    intent = await _parse_intent(llm, "到崁頂的輕軌怎麼搭")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"
    assert "出發" in intent.get("message", "")


@pytest.mark.asyncio
async def test_lrt_no_stations(llm):
    """完全沒給站名 — 應回傳 unknown"""
    intent = await _parse_intent(llm, "下一班輕軌")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"


# ── 捷運測試 ─────────────────────────────────────────

# 淡水信義線
@pytest.mark.asyncio
async def test_xinyi_from_to_with_time(llm):
    """下午時間明確，不需 mock"""
    intent = await _parse_intent(llm, "下午三點從淡水到象山的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "淡水" in intent["params"]["station_name"]
    assert "象山" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "15:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_xinyi_from_to_no_time(llm):
    intent = await _parse_intent(llm, "從象山到淡水的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "象山" in intent["params"]["station_name"]
    assert "淡水" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


@pytest.mark.asyncio
async def test_xinyi_from_only_with_time(llm):
    """「四點」在 10:00 context → 04:00 已過 → PM 16:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "四點從中山出發的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "中山" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert intent["params"]["start_time"] == "16:00"


@pytest.mark.asyncio
async def test_xinyi_from_only_no_time(llm):
    intent = await _parse_intent(llm, "淡水站下一班捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "淡水" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert "start_time" not in intent["params"]


# 板南線
@pytest.mark.asyncio
async def test_bannan_from_to_with_time(llm):
    """早上時間明確，不需 mock"""
    intent = await _parse_intent(llm, "早上九點從板橋搭捷運到忠孝敦化")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "板橋" in intent["params"]["station_name"]
    assert "忠孝敦化" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "09:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_bannan_from_to_no_time(llm):
    intent = await _parse_intent(llm, "從土城到南港展覽館的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "土城" in intent["params"]["station_name"]
    assert "南港展覽館" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


@pytest.mark.asyncio
async def test_bannan_from_only_with_time(llm):
    """「十一點」在 10:00 context → 11:00 未過 → 11:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "十一點從西門出發的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "西門" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]
    assert intent["params"]["start_time"] == "11:00"


# 環狀線 (NTMC)
@pytest.mark.asyncio
async def test_ring_from_to_with_time(llm):
    """下午時間明確，不需 mock"""
    intent = await _parse_intent(llm, "下午兩點從新埔民生到景安的捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "新埔民生" in intent["params"]["station_name"]
    assert "景安" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ring_from_to_no_time(llm):
    intent = await _parse_intent(llm, "從板新搭捷運到中和")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "板新" in intent["params"]["station_name"]
    assert "中和" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


@pytest.mark.asyncio
async def test_ring_from_only(llm):
    intent = await _parse_intent(llm, "新埔民生 下一班捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["transport_type"] == "捷運"
    assert "新埔民生" in intent["params"]["station_name"]
    assert "to_station" not in intent["params"]


# 捷運邊界情況
@pytest.mark.asyncio
async def test_mrt_only_destination(llm):
    """只給目的站，沒給出發站 — 應回傳 unknown 並提示提供出發站"""
    intent = await _parse_intent(llm, "到象山的捷運怎麼搭")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"
    assert "出發" in intent.get("message", "")


@pytest.mark.asyncio
async def test_mrt_no_stations(llm):
    """完全沒給站名 — 應回傳 unknown"""
    intent = await _parse_intent(llm, "下一班捷運")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"


# ── 台鐵測試 ─────────────────────────────────────────

# 台北 → 花蓮
@pytest.mark.asyncio
async def test_tra_taipei_hualien_with_time(llm):
    """早上時間明確，不需 mock"""
    intent = await _parse_intent(llm, "早上九點從台北搭台鐵到花蓮")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert "台北" in intent["params"]["from_station"]
    assert "花蓮" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "09:00"
    assert "train_type" not in intent["params"]


@pytest.mark.asyncio
async def test_tra_taipei_hualien_no_time(llm):
    intent = await _parse_intent(llm, "從台北到花蓮的火車")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert "台北" in intent["params"]["from_station"]
    assert "花蓮" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


# 屏東出發（屏東有台鐵站，與高鐵不同）
@pytest.mark.asyncio
async def test_tra_pingtung_with_time(llm):
    """「兩點」在 10:00 context → 02:00 已過 → PM 14:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "兩點從屏東搭台鐵到台北")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert "屏東" in intent["params"]["from_station"]
    assert "台北" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"


@pytest.mark.asyncio
async def test_tra_pingtung_no_time(llm):
    intent = await _parse_intent(llm, "從屏東搭火車到台北")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert "屏東" in intent["params"]["from_station"]
    assert "台北" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


# 無目的地
@pytest.mark.asyncio
async def test_tra_no_destination(llm):
    """只給出發站，沒有目的地 — 應回傳 unknown"""
    intent = await _parse_intent(llm, "從台北出發的火車")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"
    assert "前往" in intent.get("message", "") or "目的" in intent.get("message", "")


# 車種判斷
@pytest.mark.asyncio
async def test_tra_type_jujian(llm):
    intent = await _parse_intent(llm, "台北到桃園的區間車")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert intent["params"].get("train_type") == "區間"
    assert "台北" in intent["params"]["from_station"]
    assert "桃園" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_tra_type_ziqiang(llm):
    intent = await _parse_intent(llm, "台北到花蓮的自強號")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert intent["params"].get("train_type") == "自強"
    assert "台北" in intent["params"]["from_station"]
    assert "花蓮" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_tra_type_puyuma(llm):
    intent = await _parse_intent(llm, "台北到花蓮的普悠瑪")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert intent["params"].get("train_type") == "普悠瑪"
    assert "台北" in intent["params"]["from_station"]
    assert "花蓮" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_tra_type_new_ziqiang(llm):
    intent = await _parse_intent(llm, "台北到高雄的新自強")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_train"
    assert intent["params"].get("train_type") == "新自強"
    assert "台北" in intent["params"]["from_station"]
    assert "高雄" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_tra_type_future_ziqiang(llm):
    """未來號 unknown，因為不是正式車種"""
    intent = await _parse_intent(llm, "台北到台南的未來號")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"
    assert "車種" in intent.get("message", "")


# ── 高鐵測試 ─────────────────────────────────────────

# 南港 → 高雄
@pytest.mark.asyncio
async def test_hsr_nangang_kaohsiung_with_time(llm):
    """下午時間明確，不需 mock"""
    intent = await _parse_intent(llm, "下午兩點從南港高鐵到高雄")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "南港" in intent["params"]["from_station"]
    assert "高雄" in intent["params"]["to_station"] or "左營" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"
    assert "end_time" not in intent["params"]


@pytest.mark.asyncio
async def test_hsr_nangang_kaohsiung_no_time(llm):
    intent = await _parse_intent(llm, "從南港到高雄的高鐵")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "南港" in intent["params"]["from_station"]
    assert "高雄" in intent["params"]["to_station"] or "左營" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


# 台北出發
@pytest.mark.asyncio
async def test_hsr_taipei_with_time(llm):
    """早上時間明確，不需 mock"""
    intent = await _parse_intent(llm, "早上十點從台北高鐵到台南")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "台北" in intent["params"]["from_station"]
    assert "台南" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "10:00"


@pytest.mark.asyncio
async def test_hsr_taipei_ambiguous_time(llm):
    """「三點」在 10:00 context → 03:00 已過 → PM 15:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "三點從台北搭高鐵到台中")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "台北" in intent["params"]["from_station"]
    assert "台中" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "15:00"


# 屏東出發（無高鐵站，錯誤由 tool 處理）
@pytest.mark.asyncio
async def test_hsr_pingtung_no_time(llm):
    intent = await _parse_intent(llm, "從屏東搭高鐵到台北")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "屏東" in intent["params"]["from_station"]
    assert "台北" in intent["params"]["to_station"]
    assert "start_time" not in intent["params"]


@pytest.mark.asyncio
async def test_hsr_pingtung_with_time(llm):
    """「兩點」在 10:00 context → 02:00 已過 → PM 14:00"""
    frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
    with patch("app.agent.core.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        intent = await _parse_intent(llm, "兩點從屏東出發的高鐵到台北")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "search_next_hsr"
    assert "屏東" in intent["params"]["from_station"]
    assert "台北" in intent["params"]["to_station"]
    assert intent["params"]["start_time"] == "14:00"


# 無目的地
@pytest.mark.asyncio
async def test_hsr_no_destination(llm):
    """只給出發站，沒有目的地 — 應回傳 unknown 並提示目的地"""
    intent = await _parse_intent(llm, "從台北出發的高鐵")
    logger.info(f"Intent: {intent}")
    assert intent["function"] == "unknown"
    assert "前往" in intent.get("message", "") or "目的" in intent.get("message", "")
