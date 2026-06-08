"""LangChain tools that wrap TDX API calls for the transport agent."""

from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from app.services.tdx import tdx_client

TW_TZ = timezone(timedelta(hours=8))


def _format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "無資料"
    if seconds < 60:
        return "即將到站"
    return f"約 {seconds // 60} 分鐘"


@tool
async def search_next_metro(station_name: str, system: str = "TRTC") -> str:
    """查詢指定捷運站的下一班列車即時到站時間。

    Args:
        station_name: 捷運站名（中文），例如「台北車站」「忠孝復興」「板橋」「淡水」「紅樹林」
        system: 捷運/輕軌系統代碼。TRTC=台北捷運, KRTC=高雄捷運, TYMC=桃園機場捷運,
                NTDLRT=淡海輕軌, KLRT=高雄輕軌
    """
    try:
        data = await tdx_client.get_metro_station_live(station_name, system)

        if not data:
            return f"找不到「{station_name}」的即時資訊，請確認站名是否正確。"

        lines: list[str] = [f"🚇 {station_name}站 即時到站資訊：\n"]
        for item in data:
            dest = item.get("DestinationStationName", {}).get("Zh_tw", "未知")
            estimate = item.get("EstimateTime")
            line_name = item.get("LineName", {}).get("Zh_tw", "")
            time_str = _format_seconds(estimate)
            direction_str = f"往{dest}"
            if line_name:
                direction_str = f"[{line_name}] {direction_str}"
            lines.append(f"• {direction_str}：{time_str}")

        return "\n".join(lines)
    except Exception as e:
        return f"查詢捷運資料時發生錯誤：{e}"


@tool
async def search_next_bus(
    city: str, route_name: str, stop_name: str | None = None
) -> str:
    """查詢公車路線的預估到站時間。

    Args:
        city: 城市英文名。Taipei=台北市, NewTaipei=新北市, Taoyuan=桃園市,
              Taichung=台中市, Tainan=台南市, Kaohsiung=高雄市
        route_name: 公車路線名稱，例如「307」「紅5」「藍28」
        stop_name: 特定站牌名稱（中文），不填則回傳整條路線。例如「台北車站」
    """
    try:
        if stop_name:
            data = await tdx_client.get_bus_stop_eta(city, route_name, stop_name)
        else:
            data = await tdx_client.get_bus_eta(city, route_name)

        if not data:
            return f"找不到 {city} 的「{route_name}」路線資訊，請確認路線名稱。"

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
    from_station: str, to_station: str, train_date: str | None = None
) -> str:
    """查詢台鐵從某站到某站最近的班次時刻。

    Args:
        from_station: 出發站名（中文），例如「臺北」「松山」「花蓮」
        to_station: 到達站名（中文），例如「花蓮」「臺中」「高雄」
        train_date: 查詢日期 YYYY-MM-DD 格式，不填則查今天
    """
    try:
        stations = await tdx_client.get_tra_stations()

        station_list = stations
        if isinstance(stations, dict):
            station_list = stations.get("Stations", stations.get("stations", []))

        from_id = None
        to_id = None
        for s in station_list:
            name_zh = None
            station_id = None
            if "StationName" in s:
                name_obj = s["StationName"]
                name_zh = name_obj.get("Zh_tw") or name_obj.get("zh_tw")
                station_id = s.get("StationID") or s.get("stationID")
            elif "stationName" in s:
                name_obj = s["stationName"]
                name_zh = name_obj.get("zh_tw") or name_obj.get("Zh_tw")
                station_id = s.get("stationID") or s.get("StationID")

            if name_zh and station_id:
                if from_station in name_zh or name_zh in from_station:
                    from_id = station_id
                if to_station in name_zh or name_zh in to_station:
                    to_id = station_id

        if not from_id:
            return f"找不到出發站「{from_station}」，請用正式站名（如「臺北」而非「台北」）。"
        if not to_id:
            return f"找不到到達站「{to_station}」，請用正式站名（如「臺北」而非「台北」）。"

        if not train_date:
            now_tw = datetime.now(TW_TZ)
            train_date = now_tw.strftime("%Y-%m-%d")

        timetable = await tdx_client.get_tra_timetable(from_id, to_id, train_date)

        train_list = timetable
        if isinstance(timetable, dict):
            train_list = timetable.get("TrainTimetables", timetable.get("trainTimetables", []))

        if not train_list:
            return f"查無 {train_date} 從「{from_station}」到「{to_station}」的班次。"

        now_tw = datetime.now(TW_TZ)
        current_time = now_tw.strftime("%H:%M")

        upcoming: list[dict] = []
        for train in train_list:
            info = train.get("TrainInfo", train.get("trainInfo", {}))
            stops = train.get("StopTimes", train.get("stopTimes", []))

            train_no = info.get("TrainNo") or info.get("trainNo", "?")
            train_type = (
                info.get("TrainTypeName", {}).get("Zh_tw")
                or info.get("trainTypeName", {}).get("zh_tw", "")
            )

            dep_time = None
            arr_time = None
            for stop in stops:
                sid = stop.get("StationID") or stop.get("stationID")
                if sid == from_id:
                    dep_time = stop.get("DepartureTime") or stop.get("departureTime", "")
                if sid == to_id:
                    arr_time = stop.get("ArrivalTime") or stop.get("arrivalTime", "")

            if dep_time and dep_time >= current_time:
                upcoming.append({
                    "no": train_no,
                    "type": train_type,
                    "dep": dep_time,
                    "arr": arr_time or "?",
                })

        if not upcoming:
            return (
                f"今日從「{from_station}」到「{to_station}」已無更多班次。\n"
                f"可嘗試查詢明天的班次。"
            )

        lines: list[str] = [
            f"🚂 {from_station} → {to_station}（{train_date}）\n"
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
