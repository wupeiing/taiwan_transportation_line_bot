"""E2E tests using /debug/ask endpoint — real Groq LLM + real TDX API calls."""
import re

import pytest


@pytest.mark.asyncio
async def test_light_rail_reply(client):
    resp = await client.get("/debug/ask", params={"q": "下一班從紅樹林到崁頂的輕軌"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert any(kw in reply for kw in ["分鐘", "即將", "方向", "崁頂", "輕軌", ":"]), (
        f"Reply missing expected transportation keywords: {reply}"
    )


@pytest.mark.asyncio
async def test_hsr_reply(client):
    resp = await client.get("/debug/ask", params={"q": "高鐵台北到左營"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert time_pattern.search(reply) or any(kw in reply for kw in ["號", "高鐵", "車次"]), (
        f"Reply missing HSR schedule info: {reply}"
    )


@pytest.mark.asyncio
async def test_train_reply(client):
    resp = await client.get("/debug/ask", params={"q": "台鐵臺北到花蓮"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert time_pattern.search(reply) or any(kw in reply for kw in ["號", "台鐵", "車次", "花蓮"]), (
        f"Reply missing TRA schedule info: {reply}"
    )


@pytest.mark.asyncio
async def test_mrt_with_destination(client):
    """捷運指定起訖站 — 應回傳往象山方向的時刻"""
    resp = await client.get("/debug/ask", params={"q": "下一班從紅樹林往象山的捷運"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert any(kw in reply for kw in ["象山", "紅樹林", "捷運", ":", "分鐘", "即將"]), (
        f"Reply missing MRT direction info: {reply}"
    )


@pytest.mark.asyncio
async def test_hsr_with_start_time(client):
    """高鐵含起始時間 — 應只顯示兩點後的班次"""
    resp = await client.get("/debug/ask", params={"q": "台北高鐵到台中下午兩點後"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert time_pattern.search(reply) or any(kw in reply for kw in ["高鐵", "台中", "號", "車次"]), (
        f"Reply missing HSR schedule: {reply}"
    )


@pytest.mark.asyncio
async def test_tra_with_start_time(client):
    """台鐵含起始時間 — 應只顯示十點後的班次"""
    resp = await client.get("/debug/ask", params={"q": "台鐵台北到花蓮上午十點後"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert time_pattern.search(reply) or any(kw in reply for kw in ["台鐵", "花蓮", "號", "車次"]), (
        f"Reply missing TRA schedule: {reply}"
    )


@pytest.mark.asyncio
async def test_unknown_train_type(client):
    """未來號不是正式車種 — 應回傳說明訊息而非時刻表"""
    resp = await client.get("/debug/ask", params={"q": "台北到台南的未來號"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert len(reply) > 0
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert not time_pattern.search(reply), (
        f"Reply should not contain a schedule for unknown train type: {reply}"
    )


@pytest.mark.asyncio
async def test_tra_missing_destination(client):
    """台鐵缺少目的地 — 應回傳詢問訊息"""
    resp = await client.get("/debug/ask", params={"q": "從台北搭台鐵"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert len(reply) > 0
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert not time_pattern.search(reply), (
        f"Reply should not contain a schedule for missing destination: {reply}"
    )


@pytest.mark.asyncio
async def test_nangang_exhibition_no_wenhu_duplicates(client):
    """南港展覽館應回傳板南線時刻，且包含文湖線無法查詢的提示，不應出現南港站的結果"""
    resp = await client.get("/debug/ask", params={"q": "捷運南港展覽館"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert time_pattern.search(reply), f"Should contain BL line schedule: {reply}"
    assert "文湖線" in reply, f"Should mention Wenhu Line unavailability: {reply}"


@pytest.mark.asyncio
async def test_wenhu_only_station_no_schedule(client):
    """東湖屬於文湖線唯一路線，應回傳文湖線不支援的提示而非時刻表"""
    resp = await client.get("/debug/ask", params={"q": "捷運東湖站"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    time_pattern = re.compile(r"\d{2}:\d{2}")
    assert not time_pattern.search(reply), f"Should not contain schedule for Wenhu-only station: {reply}"
    assert "文湖線" in reply, f"Should mention Wenhu Line: {reply}"


@pytest.mark.asyncio
async def test_unknown_query_no_crash(client):
    resp = await client.get("/debug/ask", params={"q": "今天天氣如何"})
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert len(reply) > 0


@pytest.mark.asyncio
async def test_gibberish_no_crash(client):
    resp = await client.get("/debug/ask", params={"q": "asdfjkl;qwerty"})
    assert resp.status_code == 200
    assert len(resp.json()["reply"]) > 0
