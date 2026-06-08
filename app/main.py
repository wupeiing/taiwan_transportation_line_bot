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
