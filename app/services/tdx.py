import time
from difflib import SequenceMatcher

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
        self._token: str | None = None
        self._token_expires_at: float = 0
        self._tra_stations_cache: list | None = None

    async def _ensure_token(self) -> None:
        if self._token and time.time() < self._token_expires_at:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
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
        await self._ensure_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{TDX_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                params=params,
            )
        resp.raise_for_status()
        return resp.json()

    # ── Metro (MRT) ──────────────────────────────

    async def get_metro_station_timetable(
        self, station_name: str, system: str = "TRTC"
    ) -> list:
        all_data = await self._get(
            f"/v2/Rail/Metro/StationTimeTable/{system}",
            params={"$format": "JSON"},
        )
        name = station_name.rstrip("站")
        # Exact match first to avoid "南港" bleeding into "南港展覽館" results
        exact = [
            item for item in all_data
            if item.get("StationName", {}).get("Zh_tw", "") == name
        ]
        if exact:
            return exact
        # Partial match fallback for slight name variations
        return [
            item for item in all_data
            if name in item.get("StationName", {}).get("Zh_tw", "")
            or item.get("StationName", {}).get("Zh_tw", "") in name
        ]

    # ── Bus ───────────────────────────────────────

    async def get_bus_eta(self, city: str, route_name: str) -> list:
        return await self._get(
            f"/v2/Bus/EstimatedTimeOfArrival/City/{city}/{route_name}",
            params={"$format": "JSON"},
        )

    async def get_bus_stop_eta(
        self, city: str, route_name: str, stop_name: str
    ) -> list:
        all_data = await self._get(
            f"/v2/Bus/EstimatedTimeOfArrival/City/{city}/{route_name}",
            params={"$format": "JSON"},
        )
        return [
            item for item in all_data
            if stop_name in item.get("StopName", {}).get("Zh_tw", "")
        ]

    # ── TRA (Taiwan Railways) ─────────────────────

    # 台→臺 and other common aliases
    TRA_NAME_ALIASES = {
        "台北": "臺北", "台中": "臺中", "台南": "臺南",
        "台東": "臺東", "台北車站": "臺北",
        "竹南車站": "竹南",
    }

    @staticmethod
    def _normalize_tra_name(name: str) -> str:
        name = name.rstrip("車站").rstrip("站")
        return TDXClient.TRA_NAME_ALIASES.get(name, name)

    async def get_tra_stations(self) -> list:
        if self._tra_stations_cache:
            return self._tra_stations_cache
        data = await self._get("/v3/Rail/TRA/Station", params={"$format": "JSON"})
        self._tra_stations_cache = data
        return data

    def _get_all_tra_names(self, station_list: list) -> list[tuple[str, str]]:
        """Extract (station_name_zh, station_id) pairs from station list."""
        results = []
        for s in station_list:
            name_obj = s.get("StationName") or s.get("stationName", {})
            name_zh = name_obj.get("Zh_tw") or name_obj.get("zh_tw", "")
            station_id = s.get("StationID") or s.get("stationID")
            if name_zh and station_id:
                results.append((name_zh, station_id))
        return results

    def resolve_tra_station_id(self, station_list: list, name: str) -> str | None:
        normalized = self._normalize_tra_name(name)
        all_names = self._get_all_tra_names(station_list)
        # Exact match first to avoid "臺中" matching "臺中港" before "臺中"
        for name_zh, station_id in all_names:
            if normalized == name_zh:
                return station_id
        # Partial match fallback
        for name_zh, station_id in all_names:
            if normalized in name_zh or name_zh in normalized:
                return station_id
        return None

    def find_similar_tra_stations(self, station_list: list, name: str, top_n: int = 3) -> list[str]:
        """Find TRA stations with similar names using fuzzy matching."""
        normalized = self._normalize_tra_name(name)
        scored: list[tuple[float, str]] = []
        for name_zh, _ in self._get_all_tra_names(station_list):
            ratio = SequenceMatcher(None, normalized, name_zh).ratio()
            # Boost score if any character matches
            char_overlap = len(set(normalized) & set(name_zh))
            boosted = ratio + char_overlap * 0.15
            scored.append((boosted, name_zh))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:top_n]]

    async def get_tra_timetable(
        self, from_station_id: str, to_station_id: str, train_date: str
    ) -> list:
        return await self._get(
            f"/v3/Rail/TRA/DailyTrainTimetable/OD"
            f"/{from_station_id}/to/{to_station_id}/{train_date}",
            params={"$format": "JSON"},
        )

    # ── THSR (High Speed Rail) ─────────────────────

    HSR_STATIONS = {
        "南港": "0990", "台北": "1000", "臺北": "1000",
        "板橋": "1010", "桃園": "1020", "新竹": "1030",
        "苗栗": "1035", "台中": "1040", "臺中": "1040",
        "彰化": "1043", "雲林": "1047", "嘉義": "1050",
        "台南": "1060", "臺南": "1060", "左營": "1070",
        "高雄": "1070", "北車": "1000",
    }

    def resolve_hsr_station_id(self, name: str) -> str | None:
        name = name.rstrip("站")
        for key, sid in self.HSR_STATIONS.items():
            if key in name or name in key:
                return sid
        return None

    async def get_hsr_timetable(
        self, from_station_id: str, to_station_id: str, train_date: str
    ) -> list:
        return await self._get(
            f"/v2/Rail/THSR/DailyTimetable/OD"
            f"/{from_station_id}/to/{to_station_id}/{train_date}",
            params={"$format": "JSON"},
        )

    # ── Lifecycle ─────────────────────────────────

    async def close(self) -> None:
        pass  # no persistent client to close


tdx_client = TDXClient()
