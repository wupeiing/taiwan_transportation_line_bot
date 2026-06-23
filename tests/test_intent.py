"""Tests for LLM intent parsing — verifies the prompt correctly extracts function + params."""
import pytest

from app.agent.core import _build_llm, _parse_intent


@pytest.mark.asyncio
async def test_light_rail_intent():
    llm = _build_llm()
    intent = await _parse_intent(llm, "下一班從紅樹林到崁頂的輕軌")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["system"] == "NTDLRT"
    assert "紅樹林" in intent["params"]["station_name"]


@pytest.mark.asyncio
async def test_taipei_metro_intent():
    llm = _build_llm()
    intent = await _parse_intent(llm, "台北捷運忠孝復興站")
    assert intent["function"] == "search_next_metro"
    assert intent["params"]["system"] == "TRTC"
    assert "忠孝復興" in intent["params"]["station_name"]


@pytest.mark.asyncio
async def test_hsr_intent():
    llm = _build_llm()
    intent = await _parse_intent(llm, "高鐵台北到左營")
    assert intent["function"] == "search_next_hsr"
    assert "台北" in intent["params"]["from_station"]
    assert "左營" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_train_intent():
    llm = _build_llm()
    intent = await _parse_intent(llm, "台鐵臺北到花蓮")
    assert intent["function"] == "search_next_train"
    assert "臺北" in intent["params"]["from_station"] or "台北" in intent["params"]["from_station"]
    assert "花蓮" in intent["params"]["to_station"]


@pytest.mark.asyncio
async def test_unknown_intent():
    llm = _build_llm()
    intent = await _parse_intent(llm, "今天天氣如何")
    assert intent["function"] == "unknown"
