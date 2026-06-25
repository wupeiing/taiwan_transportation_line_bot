import asyncio

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.agent.core import _build_llm


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def rate_limit_delay():
    yield
    await asyncio.sleep(10)


@pytest.fixture(scope="session")
async def llm():
    _llm = _build_llm()
    yield _llm
    async_client = getattr(_llm, "async_client", None)
    if async_client is not None and hasattr(async_client, "aclose"):
        await async_client.aclose()
