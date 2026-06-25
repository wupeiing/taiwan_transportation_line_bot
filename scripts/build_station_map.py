"""One-time script to build metro/LRT station → system mapping JSON.

Usage:
    python scripts/build_station_map.py

Output:
    app/data/metro_station_map.json
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

TDX_AUTH_URL = (
    "https://tdx.transportdata.tw/auth/realms/TDXConnect"
    "/protocol/openid-connect/token"
)
TDX_BASE_URL = "https://tdx.transportdata.tw/api/basic"

MRT_SYSTEMS = {
    "TRTC": "臺北捷運",
    "KRTC": "高雄捷運",
    "TYMC": "桃園捷運",
    "TMRT": "臺中捷運",
    "NTMC": "新北捷運",
}

LRT_SYSTEMS = {
    "NTDLRT": "淡海輕軌",
    "NTALRT": "安坑輕軌",
    "KLRT":   "高雄輕軌",
}

OUTPUT_PATH = Path(__file__).parent.parent / "app" / "data" / "metro_station_map.json"


async def fetch_token(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        TDX_AUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["TDX_CLIENT_ID"],
            "client_secret": os.environ["TDX_CLIENT_SECRET"],
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def fetch_stations(client: httpx.AsyncClient, token: str, system: str) -> list[str]:
    resp = await client.get(
        f"{TDX_BASE_URL}/v2/Rail/Metro/StationOfLine/{system}",
        headers={"Authorization": f"Bearer {token}"},
        params={"$format": "JSON", "$top": 9999},
    )
    resp.raise_for_status()
    lines = resp.json()

    seen: set[str] = set()
    for line in lines:
        for station in line.get("Stations", []):
            name = station.get("StationName", {}).get("Zh_tw", "")
            if name:
                seen.add(name)
    return list(seen)


async def main() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        token = await fetch_token(client)
        print(f"Token acquired.")

        station_map: dict[str, str] = {}

        for system_code, system_name in {**MRT_SYSTEMS, **LRT_SYSTEMS}.items():
            await asyncio.sleep(60)
            try:
                stations = await fetch_stations(client, token, system_code)
                for name in stations:
                    # If a station exists in multiple systems, keep both (comma-separated)
                    if name in station_map:
                        existing = station_map[name]
                        if system_code not in existing:
                            station_map[name] = f"{existing},{system_code}"
                    else:
                        station_map[name] = system_code
                print(f"  {system_name} ({system_code}): {len(stations)} unique stations")
            except Exception as e:
                print(f"  ERROR fetching {system_code}: {e}")

    output = {
        "mrt_systems": list(MRT_SYSTEMS.keys()),
        "lrt_systems": list(LRT_SYSTEMS.keys()),
        "stations": station_map,
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(station_map)} stations to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
