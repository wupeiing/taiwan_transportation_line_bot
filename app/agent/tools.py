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


def _is_service_today(service_day: dict) -> bool:
    """Check if a timetable entry is valid for today."""
    now_tw = datetime.now(TW_TZ)
    day_map = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
    }
    return service_day.get(day_map.get(now_tw.weekday(), ""), False)


@tool
async def search_next_metro(station_name: str, system: str = "TRTC") -> str:
    """查詢指定捷運站的下一班列車時刻。

    Args:
        station_name: 捷運站名（中文），例如「台北車站」「忠孝復興」「板橋」「淡水」「紅樹林」
        system: 捷運/輕軌系統代碼。TRTC=台北捷運, KRTC=高雄捷運, TYMC=桃園機場捷運,
                NTDLRT=淡海輕軌, KLRT=高雄輕軌
    """
    try:
        data = await tdx_client.get_metro_station_timetable(station_name, system)

        if not data:
            return f"找不到「{station_name}」的時刻資訊，請確認站名是否正確。"

        now_tw = datetime.now(TW_TZ)
        current_time = now_tw.strftime("%H:%M")

        lines: list[str] = [f"🚇 {station_name}站 時刻表：\n"]

        for entry in data:
            dest = entry.get("DestinationStationName", {}).get("Zh_tw", "未知")
            service_day = entry.get("ServiceDay", {})

            if not _is_service_today(service_day):
                continue

            timetables = entry.get("Timetables", [])
            upcoming = [
                t["DepartureTime"] for t in timetables
                if t.get("DepartureTime", "") >= current_time
            ]

            if not upcoming:
                lines.append(f"• 往{dest}：今日已無班次")
                continue

            next_3 = upcoming[:3]
            times_str = "、".join(next_3)
            lines.append(f"• 往{dest}：接下來 {times_str}")

        if len(lines) == 1:
            return f"「{station_name}」站目前無可查詢的班次。"

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


@tool
async def search_next_hsr(
    from_station: str, to_station: str,
) -> str:
    """查詢高鐵從某站到某站最近的班次時刻。

    Args:
        from_station: 出發站名（中文），例如「台北」「台中」「左營」「高雄」
        to_station: 到達站名（中文），例如「台南」「台北」「板橋」
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
        current_time = now_tw.strftime("%H:%M")

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

            if dep_time >= current_time:
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
