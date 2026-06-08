import time

import httpx

from app.config import settings

TDX_AUTH_URL = (
    "https://tdx.transportdata.tw/auth/realms/TDXConnect"
    "/protocol/openid-connect/token"
)
TDX_BASE_URL = "https://tdx.transportdata.tw/api/basic"


class TDXClient:
    """Client for Taiwan TDX (Transport Data eXchange) API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=10.0)
        self._token: str | None = None
        self._token_expires_at: float = 0

    async def _ensure_token(self) -> None:
        """Refresh access token if expired (tokens last 24 hours)."""
        if self._token and time.time() < self._token_expires_at:
            return

        resp = await self._client.post(
            TDX_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": settings.tdx_client_id,
                "client_secret": settings.tdx_client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 86400) - 3600

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        """Make authenticated GET request to TDX API."""
        await self._ensure_token()
        resp = await self._client.get(
            f"{TDX_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {self._token}"},
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Metro (MRT) ──────────────────────────────

    async def get_metro_station_live(
        self, station_name: str, system: str = "TRTC"
    ) -> list:
        """Get real-time arrivals for a specific metro station."""
        return await self._get(
            f"/v2/Rail/Metro/LiveBoard/{system}",
            params={
                "$filter": f"contains(StationName/Zh_tw,'{station_name}')",
                "$format": "JSON",
            },
        )

    # ── Bus ───────────────────────────────────────

    async def get_bus_eta(self, city: str, route_name: str) -> list:
        """Get estimated time of arrival for all stops on a bus route."""
        return await self._get(
            f"/v2/Bus/EstimatedTimeOfArrival/City/{city}/{route_name}",
            params={"$format": "JSON"},
        )

    async def get_bus_stop_eta(
        self, city: str, route_name: str, stop_name: str
    ) -> list:
        """Get ETA for a specific stop on a bus route."""
        return await self._get(
            f"/v2/Bus/EstimatedTimeOfArrival/City/{city}/{route_name}",
            params={
                "$filter": f"contains(StopName/Zh_tw,'{stop_name}')",
                "$format": "JSON",
            },
        )

    # ── TRA (Taiwan Railways) ─────────────────────

    async def get_tra_timetable(
        self, from_station_id: str, to_station_id: str, train_date: str
    ) -> list:
        """Get TRA daily timetable between two stations."""
        return await self._get(
            f"/v3/Rail/TRA/DailyTrainTimetable/OD"
            f"/{from_station_id}/to/{to_station_id}/{train_date}",
            params={"$format": "JSON"},
        )

    async def get_tra_stations(self) -> list:
        """Get list of all TRA stations with IDs."""
        return await self._get(
            "/v3/Rail/TRA/Station",
            params={"$format": "JSON"},
        )

    # ── Lifecycle ─────────────────────────────────

    async def close(self) -> None:
        await self._client.aclose()


tdx_client = TDXClient()
