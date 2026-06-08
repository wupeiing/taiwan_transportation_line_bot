from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent

from app.bot.handler import handle_message
from app.config import settings
from app.services.tdx import tdx_client

parser = WebhookParser(settings.line_channel_secret)
messaging_api: AsyncMessagingApi | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global messaging_api
    config = Configuration(access_token=settings.line_channel_access_token)
    async_api_client = AsyncApiClient(config)
    messaging_api = AsyncMessagingApi(async_api_client)
    yield
    await async_api_client.close()
    await tdx_client.close()


app = FastAPI(title="Taiwan Transport Bot", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/debug/metro/{station_name}")
async def debug_metro(station_name: str, system: str = "TRTC"):
    """Debug endpoint to test TDX metro StationTimeTable API."""
    try:
        data = await tdx_client.get_metro_station_timetable(station_name, system)
        return {"station": station_name, "system": system, "count": len(data), "results": data[:2]}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/debug/metro-all/{system}")
async def debug_metro_all(system: str = "TRTC", top: int = 5):
    """Debug: dump first N records from LiveBoard to see actual field names."""
    try:
        from app.services.tdx import TDX_BASE_URL
        await tdx_client._ensure_token()
        resp = await tdx_client._client.get(
            f"{TDX_BASE_URL}/v2/Rail/Metro/LiveBoard/{system}",
            headers={"Authorization": f"Bearer {tdx_client._token}"},
            params={"$top": str(top), "$format": "JSON"},
        )
        resp.raise_for_status()
        return {"system": system, "results": resp.json()}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/debug/hsr-timetable/{from_station}/{to_station}")
async def debug_hsr_timetable(from_station: str, to_station: str):
    """Debug: test THSR DailyTimetable OD API."""
    from datetime import datetime, timezone, timedelta
    from app.services.tdx import TDX_BASE_URL
    TW_TZ = timezone(timedelta(hours=8))
    today = datetime.now(TW_TZ).strftime("%Y-%m-%d")

    await tdx_client._ensure_token()
    headers = {"Authorization": f"Bearer {tdx_client._token}"}

    # Try different endpoint patterns for HSR
    endpoints = {
        "v2_od": f"/v2/Rail/THSR/DailyTimetable/OD/{from_station}/to/{to_station}/{today}",
        "v2_daily_top3": f"/v2/Rail/THSR/DailyTimetable/Today",
    }
    results = {}
    for name, path in endpoints.items():
        params = {"$format": "JSON"}
        if "top3" in name:
            params["$top"] = "3"
        resp = await tdx_client._client.get(
            f"{TDX_BASE_URL}{path}",
            headers=headers,
            params=params,
        )
        if resp.status_code == 200:
            data = resp.json()
            results[name] = {"status": 200, "count": len(data) if isinstance(data, list) else "dict", "sample": (data[:2] if isinstance(data, list) else data)[:2] if isinstance(data, list) else str(data)[:1000]}
        else:
            results[name] = {"status": resp.status_code, "body": resp.text[:500]}
    return results


@app.get("/debug/ask")
async def debug_ask(q: str):
    """Debug: test the full agent pipeline without LINE."""
    from app.agent.core import get_transport_response
    reply = await get_transport_response(q)
    return {"question": q, "reply": reply}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(...),
):
    body = (await request.body()).decode("utf-8")

    try:
        events = parser.parse(body, x_line_signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent):
            await handle_message(event, messaging_api)

    return "OK"
