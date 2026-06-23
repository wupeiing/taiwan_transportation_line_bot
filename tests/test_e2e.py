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
