"""LangChain tools that wrap TDX API calls for the transport agent."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from langchain_core.tools import tool

from app.services.tdx import tdx_client

_STATION_MAP_PATH = Path(__file__).parent.parent / "data" / "metro_station_map.json"
_station_map_cache: dict | None = None


def _load_station_map() -> dict:
    global _station_map_cache
    if _station_map_cache is None:
        _station_map_cache = json.loads(_STATION_MAP_PATH.read_text())
    return _station_map_cache


def _resolve_metro_systems(station_name: str, transport_type: str) -> list[str]:
    """根據站名與交通類型（捷運/輕軌）回傳對應系統代碼列表。"""
    station_map = _load_station_map()
    mrt_systems = set(station_map["mrt_systems"])
    lrt_systems = set(station_map["lrt_systems"])

    systems_str = station_map["stations"].get(station_name, "")
    if not systems_str:
        return []

    all_systems = [s.strip() for s in systems_str.split(",")]

    if transport_type == "輕軌":
        filtered = [s for s in all_systems if s in lrt_systems]
    else:
        filtered = [s for s in all_systems if s in mrt_systems]

    return filtered if filtered else all_systems


TW_TZ = timezone(timedelta(hours=8))

TRAIN_TYPE_NAMES: dict[str, str] = {
    "1":  "太魯閣",
    "2":  "普悠瑪",
    "3":  "自強",
    "4":  "莒光",
    "5":  "復興",
    "6":  "區間",
    "7":  "普快",
    "10": "區間快",
    "11": "自強3000",
}

# 使用者說的車種 → 允許的 TrainTypeCode 集合
TRAIN_TYPE_FILTER: dict[str, set[str]] = {
    "自強":   {"1", "2", "3", "11"},
    "新自強": {"11"},
    "太魯閣": {"1"},
    "普悠瑪": {"2"},
    "莒光":   {"4"},
    "復興":   {"5"},
    "區間":   {"6"},
    "區間車": {"6"},
    "區間快": {"10"},
    "普快":   {"7"},
}


def _format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "無資料"
    if seconds < 60:
        return "即將到站"
    return f"約 {seconds // 60} 分鐘"


def _is_service_today(service_day: dict) -> bool:
    """Check if a timetable entry is valid for today."""
    now_tw = datetime.now(TW_TZ)
    day_map = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
    }
    return service_day.get(day_map.get(now_tw.weekday(), ""), False)


def _compute_time_bounds(start_time: str | None, end_time: str | None) -> tuple[str, str | None]:
    """Return (lower, upper) HH:MM bounds for timetable filtering.

    lower = start_time if given, else current Taiwan time.
    upper = end_time if given, else None (no upper bound).
    """
    now_tw = datetime.now(TW_TZ)
    lower = start_time if start_time else now_tw.strftime("%H:%M")
    return lower, end_time


@tool
async def search_next_metro(
    station_name: str,
    transport_type: str = "捷運",
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """查詢指定捷運或輕軌站的下一班列車時刻。

    Args:
        station_name: 站名（中文），例如「台北車站」「忠孝復興」「紅樹林」「淡金北新」
        transport_type: 「捷運」或「輕軌」
        start_time: 查詢起始時間 HH:MM，不填則從現在開始
        end_time: 查詢結束時間 HH:MM，不填則不限制結束時間
    """
    try:
        systems = _resolve_metro_systems(station_name, transport_type)

        if not systems:
            type_label = transport_type if transport_type in ("捷運", "輕軌") else "捷運/輕軌"
            return f"找不到{type_label}站「{station_name}」，請確認站名是否正確。"

        station_map = _load_station_map()
        wenhu_only = set(station_map.get("wenhu_line_only_stations", []))
        wenhu_transfer = set(station_map.get("wenhu_line_transfer_stations", []))

        lower, upper = _compute_time_bounds(start_time, end_time)
        icon = "🚈" if transport_type == "輕軌" else "🚇"
        lines: list[str] = [f"{icon} {station_name}站 時刻表：\n"]

        for system in systems:
            data = await tdx_client.get_metro_station_timetable(station_name, system)
            if not data:
                if system == "TRTC" and station_name in wenhu_only:
                    return (
                        f"🚇 {station_name}站屬於文湖線（BR），"
                        f"目前 TDX 未提供文湖線即時時刻資料，"
                        f"建議使用台北捷運官方 App 查詢。"
                    )
                continue

            for entry in data:
                dest = entry.get("DestinationStationName", {}).get("Zh_tw", "未知")
                service_day = entry.get("ServiceDay", {})

                if not _is_service_today(service_day):
                    continue

                timetables = entry.get("Timetables", [])
                upcoming = [
                    t["DepartureTime"] for t in timetables
                    if t.get("DepartureTime", "") >= lower
                    and (upper is None or t.get("DepartureTime", "") < upper)
                ]

                if not upcoming:
                    lines.append(f"• 往{dest}：今日已無班次")
                    continue

                times_str = "、".join(upcoming[:3])
                lines.append(f"• 往{dest}：接下來 {times_str}")

        if len(lines) == 1:
            return f"「{station_name}」站目前無可查詢的班次。"

        if station_name in wenhu_transfer:
            lines.append("\n（文湖線方向時刻目前無法查詢，請使用台北捷運官方 App）")

        return "\n".join(lines)
    except Exception as e:
        return f"查詢捷運/輕軌資料時發生錯誤：{e}"


@tool
async def search_next_bus(
    city: str,
    route_name: str,
    stop_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """查詢公車路線的預估到站時間。

    Args:
        city: 城市英文名。Taipei=台北市, NewTaipei=新北市, Taoyuan=桃園市,
              Taichung=台中市, Tainan=台南市, Kaohsiung=高雄市
        route_name: 公車路線名稱，例如「307」「紅5」「藍28」
        stop_name: 特定站牌名稱（中文），不填則回傳整條路線。例如「台北車站」
        start_time: 篩選預估到站時間的起始 HH:MM，不填則從現在開始
        end_time: 篩選預估到站時間的結束 HH:MM，不填則不限制結束時間
    """
    try:
        if stop_name:
            data = await tdx_client.get_bus_stop_eta(city, route_name, stop_name)
        else:
            data = await tdx_client.get_bus_eta(city, route_name)

        if not data:
            return f"找不到 {city} 的「{route_name}」路線資訊，請確認路線名稱。"

        now_tw = datetime.now(TW_TZ)
        lower, upper = _compute_time_bounds(start_time, end_time)
        apply_filter = start_time is not None or end_time is not None

        status_map = {
            0: None,
            1: "尚未發車",
            2: "交管不停靠",
            3: "末班車已過",
            4: "今日未營運",
        }

        lines: list[str] = [f"🚌 {route_name} 路公車到站預估：\n"]
        shown = 0
        for item in data:
            if shown >= 15:
                lines.append(f"\n（還有更多站牌，僅顯示前 15 筆）")
                break

            name = item.get("StopName", {}).get("Zh_tw", "未知站牌")
            status_code = item.get("StopStatus", 0)
            estimate = item.get("EstimateTime")
            direction = item.get("Direction", 0)
            dir_label = "去程" if direction == 0 else "返程"

            if apply_filter:
                if status_code != 0 or estimate is None:
                    continue
                arrival = (now_tw + timedelta(seconds=estimate)).strftime("%H:%M")
                if arrival < lower or (upper is not None and arrival >= upper):
                    continue

            status_text = status_map.get(status_code)
            if status_text:
                lines.append(f"• [{dir_label}] {name}：{status_text}")
            else:
                lines.append(f"• [{dir_label}] {name}：{_format_seconds(estimate)}")
            shown += 1

        return "\n".join(lines)
    except Exception as e:
        return f"查詢公車資料時發生錯誤：{e}"


@tool
async def search_next_train(
    from_station: str,
    to_station: str,
    train_type: str | None = None,
    train_date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """查詢台鐵從某站到某站最近的班次時刻。

    Args:
        from_station: 出發站名（中文），例如「臺北」「松山」「花蓮」
        to_station: 到達站名（中文），例如「花蓮」「臺中」「高雄」
        train_type: 車種篩選，例如「自強」「新自強」「太魯閣」「莒光」「區間」「區間快」，不填則全部
        train_date: 查詢日期 YYYY-MM-DD 格式，不填則查今天
        start_time: 查詢起始時間 HH:MM，不填則從現在開始
        end_time: 查詢結束時間 HH:MM，不填則不限制結束時間
    """
    try:
        stations = await tdx_client.get_tra_stations()

        station_list = stations
        if isinstance(stations, dict):
            station_list = stations.get("Stations", stations.get("stations", []))

        from_id = tdx_client.resolve_tra_station_id(station_list, from_station)
        to_id = tdx_client.resolve_tra_station_id(station_list, to_station)

        if not from_id:
            similar = tdx_client.find_similar_tra_stations(station_list, from_station)
            suggestions = "、".join(f"「{s}」" for s in similar)
            return f"找不到出發站「{from_station}」。\n你是不是要找：{suggestions}？"
        if not to_id:
            similar = tdx_client.find_similar_tra_stations(station_list, to_station)
            suggestions = "、".join(f"「{s}」" for s in similar)
            return f"找不到到達站「{to_station}」。\n你是不是要找：{suggestions}？"

        if not train_date:
            now_tw = datetime.now(TW_TZ)
            train_date = now_tw.strftime("%Y-%m-%d")

        timetable = await tdx_client.get_tra_timetable(from_id, to_id, train_date)

        train_list = timetable
        if isinstance(timetable, dict):
            train_list = timetable.get("TrainTimetables", timetable.get("trainTimetables", []))

        if not train_list:
            return f"查無 {train_date} 從「{from_station}」到「{to_station}」的班次。"

        lower, upper = _compute_time_bounds(start_time, end_time)
        allowed_codes = TRAIN_TYPE_FILTER.get(train_type) if train_type else None

        upcoming: list[dict] = []
        for train in train_list:
            info = train.get("TrainInfo", train.get("trainInfo", {}))
            stops = train.get("StopTimes", train.get("stopTimes", []))

            train_no = info.get("TrainNo") or info.get("trainNo", "?")
            type_code = str(info.get("TrainTypeCode", ""))
            type_name = TRAIN_TYPE_NAMES.get(type_code, info.get("TrainTypeName", {}).get("Zh_tw", ""))

            if allowed_codes and type_code not in allowed_codes:
                continue

            dep_time = None
            arr_time = None
            for stop in stops:
                sid = stop.get("StationID") or stop.get("stationID")
                if sid == from_id:
                    dep_time = stop.get("DepartureTime") or stop.get("departureTime", "")
                if sid == to_id:
                    arr_time = stop.get("ArrivalTime") or stop.get("arrivalTime", "")

            if dep_time and dep_time >= lower and (upper is None or dep_time < upper):
                upcoming.append({
                    "no": train_no,
                    "type": type_name,
                    "dep": dep_time,
                    "arr": arr_time or "?",
                })

        if not upcoming:
            type_hint = f"（車種：{train_type}）" if train_type else ""
            return (
                f"今日從「{from_station}」到「{to_station}」已無更多班次{type_hint}。\n"
                f"可嘗試查詢明天的班次。"
            )

        type_hint = f"（{train_type}）" if train_type else ""
        lines: list[str] = [
            f"🚂 {from_station} → {to_station}{type_hint}（{train_date}）\n"
            f"接下來的班次：\n"
        ]
        for t in upcoming[:5]:
            lines.append(
                f"• {t['type']} {t['no']}號　"
                f"{t['dep']} 出發 → {t['arr']} 到達"
            )

        if len(upcoming) > 5:
            lines.append(f"\n（還有 {len(upcoming) - 5} 班，僅顯示最近 5 班）")

        return "\n".join(lines)
    except Exception as e:
        return f"查詢台鐵時刻時發生錯誤：{e}"


@tool
async def search_next_hsr(
    from_station: str,
    to_station: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """查詢高鐵從某站到某站最近的班次時刻。

    Args:
        from_station: 出發站名（中文），例如「台北」「台中」「左營」「高雄」
        to_station: 到達站名（中文），例如「台南」「台北」「板橋」
        start_time: 查詢起始時間 HH:MM，不填則從現在開始
        end_time: 查詢結束時間 HH:MM，不填則不限制結束時間
    """
    try:
        from_id = tdx_client.resolve_hsr_station_id(from_station)
        to_id = tdx_client.resolve_hsr_station_id(to_station)

        if not from_id:
            return f"找不到高鐵站「{from_station}」。高鐵站有：南港、台北、板橋、桃園、新竹、苗栗、台中、彰化、雲林、嘉義、台南、左營(高雄)。"
        if not to_id:
            return f"找不到高鐵站「{to_station}」。高鐵站有：南港、台北、板橋、桃園、新竹、苗栗、台中、彰化、雲林、嘉義、台南、左營(高雄)。"

        now_tw = datetime.now(TW_TZ)
        train_date = now_tw.strftime("%Y-%m-%d")
        lower, upper = _compute_time_bounds(start_time, end_time)

        data = await tdx_client.get_hsr_timetable(from_id, to_id, train_date)

        if not data:
            return f"查無今日從「{from_station}」到「{to_station}」的高鐵班次。"

        upcoming: list[dict] = []
        for train in data:
            info = train.get("DailyTrainInfo", {})
            origin = train.get("OriginStopTime", {})
            dest = train.get("DestinationStopTime", {})

            dep_time = origin.get("DepartureTime", "")
            arr_time = dest.get("ArrivalTime", "")

            if dep_time >= lower and (upper is None or dep_time < upper):
                upcoming.append({
                    "no": info.get("TrainNo", "?"),
                    "dep": dep_time,
                    "arr": arr_time,
                })

        if not upcoming:
            return (
                f"今日從「{from_station}」到「{to_station}」已無更多高鐵班次。\n"
                f"可嘗試查詢明天的班次。"
            )

        lines: list[str] = [
            f"🚄 高鐵 {from_station} → {to_station}（{train_date}）\n"
        ]
        for t in upcoming[:5]:
            lines.append(
                f"• 車次 {t['no']}　{t['dep']} 出發 → {t['arr']} 到達"
            )

        if len(upcoming) > 5:
            lines.append(f"\n（還有 {len(upcoming) - 5} 班，僅顯示最近 5 班）")

        return "\n".join(lines)
    except Exception as e:
        return f"查詢高鐵時刻時發生錯誤：{e}"
