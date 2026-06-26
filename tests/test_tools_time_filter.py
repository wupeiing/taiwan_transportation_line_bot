"""Tests for start_time / end_time filtering on search_next_* tools."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.tools import (
    _compute_time_bounds,
    search_next_bus,
    search_next_hsr,
    search_next_metro,
    search_next_train,
)

TW_TZ = timezone(timedelta(hours=8))

# Service day covering all weekdays so _is_service_today returns True on any test run day
_ALL_DAYS = {
    "Monday": True, "Tuesday": True, "Wednesday": True,
    "Thursday": True, "Friday": True, "Saturday": True, "Sunday": True,
}


# ── _compute_time_bounds ──────────────────────────────────────────────────────

class TestComputeTimeBounds:
    def test_neither_uses_current_taiwan_time(self):
        frozen = datetime(2025, 6, 15, 10, 30, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            lower, upper = _compute_time_bounds(None, None)
        assert lower == "10:30"
        assert upper is None

    def test_start_only(self):
        lower, upper = _compute_time_bounds("12:00", None)
        assert lower == "12:00"
        assert upper is None

    def test_end_only_lower_falls_back_to_current_time(self):
        frozen = datetime(2025, 6, 15, 9, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            lower, upper = _compute_time_bounds(None, "14:00")
        assert lower == "09:00"
        assert upper == "14:00"

    def test_both_passed_through_unchanged(self):
        lower, upper = _compute_time_bounds("08:00", "20:00")
        assert lower == "08:00"
        assert upper == "20:00"


# ── search_next_metro ─────────────────────────────────────────────────────────

_METRO_DATA = [{
    "DestinationStationName": {"Zh_tw": "南港展覽館"},
    "ServiceDay": _ALL_DAYS,
    "Timetables": [
        {"DepartureTime": "08:00"},
        {"DepartureTime": "10:00"},
        {"DepartureTime": "12:00"},
        {"DepartureTime": "14:00"},
        {"DepartureTime": "16:00"},
    ],
}]


class TestSearchNextMetroTimeFilter:
    async def test_start_time_only_excludes_earlier_trains(self):
        with patch("app.agent.tools.tdx_client") as m:
            m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DATA)
            result = await search_next_metro.ainvoke({
                "station_name": "板橋",
                "start_time": "12:00",
            })
        assert "12:00" in result
        assert "14:00" in result
        assert "08:00" not in result
        assert "10:00" not in result

    async def test_end_time_only_upper_bound_is_exclusive(self):
        frozen = datetime(2025, 6, 15, 10, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DATA)
                result = await search_next_metro.ainvoke({
                    "station_name": "板橋",
                    "end_time": "12:00",
                })
        assert "10:00" in result       # >= current (10:00) and < 12:00
        assert "08:00" not in result   # before current time
        assert "12:00" not in result   # upper bound is exclusive (<)
        assert "14:00" not in result

    async def test_both_bounds_narrow_window(self):
        with patch("app.agent.tools.tdx_client") as m:
            m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DATA)
            result = await search_next_metro.ainvoke({
                "station_name": "板橋",
                "start_time": "10:00",
                "end_time": "14:00",
            })
        assert "10:00" in result
        assert "12:00" in result
        assert "08:00" not in result
        assert "14:00" not in result   # upper bound exclusive
        assert "16:00" not in result

    async def test_no_filter_defaults_to_current_time_as_lower_bound(self):
        frozen = datetime(2025, 6, 15, 13, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DATA)
                result = await search_next_metro.ainvoke({"station_name": "板橋"})
        assert "14:00" in result
        assert "16:00" in result
        assert "08:00" not in result
        assert "10:00" not in result
        assert "12:00" not in result


# ── search_next_metro deduplication ──────────────────────────────────────────

# Simulates 台北車站 on BL line: TDX returns two entries for 往象山 (two service patterns)
_METRO_DUPLICATE_DATA = [
    {
        "DestinationStationName": {"Zh_tw": "象山"},
        "ServiceDay": _ALL_DAYS,
        "Timetables": [
            {"DepartureTime": "08:17"},
            {"DepartureTime": "08:27"},
            {"DepartureTime": "08:37"},
        ],
    },
    {
        "DestinationStationName": {"Zh_tw": "象山"},
        "ServiceDay": _ALL_DAYS,
        "Timetables": [
            {"DepartureTime": "08:20"},
            {"DepartureTime": "08:30"},
            {"DepartureTime": "08:40"},
        ],
    },
    {
        "DestinationStationName": {"Zh_tw": "頂埔"},
        "ServiceDay": _ALL_DAYS,
        "Timetables": [
            {"DepartureTime": "08:19"},
            {"DepartureTime": "08:29"},
        ],
    },
]


class TestSearchNextMetroDeduplicate:
    async def test_duplicate_destinations_merged_into_one_row(self):
        """同方向有多筆 TDX entry 時，應合併成一行（不重複顯示）。"""
        frozen = datetime(2025, 6, 15, 8, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DUPLICATE_DATA)
                result = await search_next_metro.ainvoke({"station_name": "台北車站"})
        assert result.count("往象山") == 1, f"Expected exactly one 往象山 row:\n{result}"
        assert result.count("往頂埔") == 1, f"Expected exactly one 往頂埔 row:\n{result}"

    async def test_merged_times_are_sorted_and_earliest_three_shown(self):
        """合併後取最早的三班，且時間應排序。"""
        frozen = datetime(2025, 6, 15, 8, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DUPLICATE_DATA)
                result = await search_next_metro.ainvoke({"station_name": "台北車站"})
        # Merged 往象山: 08:17, 08:20, 08:27, 08:30, 08:37, 08:40 → first 3: 08:17, 08:20, 08:27
        assert "08:17" in result
        assert "08:20" in result
        assert "08:27" in result
        assert "08:30" not in result
        assert "08:40" not in result

    async def test_time_filter_applied_before_merge(self):
        """start_time 篩選在合併前套用：08:25 後，每個 entry 只留符合的時間。"""
        with patch("app.agent.tools.tdx_client") as m:
            m.get_metro_station_timetable = AsyncMock(return_value=_METRO_DUPLICATE_DATA)
            result = await search_next_metro.ainvoke({
                "station_name": "台北車站",
                "start_time": "08:25",
            })
        assert "08:17" not in result
        assert "08:20" not in result
        assert "08:27" in result
        assert "08:30" in result


# ── search_next_bus ───────────────────────────────────────────────────────────

# With now=10:00:00, estimated arrivals:
#   A站: 10:05 (+300 s)   B站: 11:00 (+3600 s)
#   C站: 12:00 (+7200 s)  D站: 13:00 (+10800 s)
#   E站: no ETA (StopStatus=1, not yet departed)
_FROZEN_BUS = datetime(2025, 6, 15, 10, 0, 0, tzinfo=TW_TZ)
_BUS_DATA = [
    {"StopName": {"Zh_tw": "A站"}, "StopStatus": 0, "EstimateTime": 300,   "Direction": 0},
    {"StopName": {"Zh_tw": "B站"}, "StopStatus": 0, "EstimateTime": 3600,  "Direction": 0},
    {"StopName": {"Zh_tw": "C站"}, "StopStatus": 0, "EstimateTime": 7200,  "Direction": 0},
    {"StopName": {"Zh_tw": "D站"}, "StopStatus": 0, "EstimateTime": 10800, "Direction": 0},
    {"StopName": {"Zh_tw": "E站"}, "StopStatus": 1, "EstimateTime": None,  "Direction": 0},
]


class TestSearchNextBusTimeFilter:
    async def test_no_filter_shows_all_stops_including_status_messages(self):
        with patch("app.agent.tools.tdx_client") as m:
            m.get_bus_eta = AsyncMock(return_value=_BUS_DATA)
            result = await search_next_bus.ainvoke({"city": "Taipei", "route_name": "307"})
        for stop in ("A站", "B站", "C站", "D站", "E站"):
            assert stop in result
        assert "尚未發車" in result

    async def test_filter_by_arrival_window(self):
        # start=11:00, end=13:00 → B站(11:00) and C站(12:00) only
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = _FROZEN_BUS
            with patch("app.agent.tools.tdx_client") as m:
                m.get_bus_eta = AsyncMock(return_value=_BUS_DATA)
                result = await search_next_bus.ainvoke({
                    "city": "Taipei", "route_name": "307",
                    "start_time": "11:00", "end_time": "13:00",
                })
        assert "B站" in result     # arrives 11:00, in [11:00, 13:00)
        assert "C站" in result     # arrives 12:00, in [11:00, 13:00)
        assert "A站" not in result  # arrives 10:05, before start
        assert "D站" not in result  # arrives 13:00, equals upper (exclusive)
        assert "E站" not in result  # no ETA; skipped when filter is active

    async def test_start_time_only_no_upper_cap(self):
        # start=11:00, no end → B, C, D all included
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = _FROZEN_BUS
            with patch("app.agent.tools.tdx_client") as m:
                m.get_bus_eta = AsyncMock(return_value=_BUS_DATA)
                result = await search_next_bus.ainvoke({
                    "city": "Taipei", "route_name": "307",
                    "start_time": "11:00",
                })
        assert "B站" in result
        assert "C站" in result
        assert "D站" in result     # no upper bound
        assert "A站" not in result  # 10:05 < 11:00
        assert "E站" not in result  # no ETA

    async def test_filter_active_drops_non_eta_stops(self):
        """Any time bound being set causes stops with no valid ETA to be skipped."""
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = _FROZEN_BUS
            with patch("app.agent.tools.tdx_client") as m:
                m.get_bus_eta = AsyncMock(return_value=_BUS_DATA)
                result = await search_next_bus.ainvoke({
                    "city": "Taipei", "route_name": "307",
                    "end_time": "23:59",
                })
        assert "E站" not in result
        assert "尚未發車" not in result


# ── search_next_train ─────────────────────────────────────────────────────────

_TRAIN_STATIONS = [
    {"StationID": "1000", "StationName": {"Zh_tw": "臺北"}},
    {"StationID": "2000", "StationName": {"Zh_tw": "花蓮"}},
]
_TRAIN_DATA = {
    "TrainTimetables": [
        {
            "TrainInfo": {"TrainNo": "101", "TrainTypeName": {"Zh_tw": "自強號"}},
            "StopTimes": [
                {"StationID": "1000", "DepartureTime": "08:00", "ArrivalTime": ""},
                {"StationID": "2000", "ArrivalTime": "11:00", "DepartureTime": ""},
            ],
        },
        {
            "TrainInfo": {"TrainNo": "103", "TrainTypeName": {"Zh_tw": "自強號"}},
            "StopTimes": [
                {"StationID": "1000", "DepartureTime": "10:00", "ArrivalTime": ""},
                {"StationID": "2000", "ArrivalTime": "13:00", "DepartureTime": ""},
            ],
        },
        {
            "TrainInfo": {"TrainNo": "105", "TrainTypeName": {"Zh_tw": "自強號"}},
            "StopTimes": [
                {"StationID": "1000", "DepartureTime": "12:00", "ArrivalTime": ""},
                {"StationID": "2000", "ArrivalTime": "15:00", "DepartureTime": ""},
            ],
        },
        {
            "TrainInfo": {"TrainNo": "107", "TrainTypeName": {"Zh_tw": "自強號"}},
            "StopTimes": [
                {"StationID": "1000", "DepartureTime": "14:00", "ArrivalTime": ""},
                {"StationID": "2000", "ArrivalTime": "17:00", "DepartureTime": ""},
            ],
        },
    ]
}

_STATION_ID_MAP = {"臺北": "1000", "花蓮": "2000"}


def _wire_train_mock(m):
    m.get_tra_stations = AsyncMock(return_value=_TRAIN_STATIONS)
    m.resolve_tra_station_id = MagicMock(
        side_effect=lambda lst, name: _STATION_ID_MAP.get(name)
    )
    m.get_tra_timetable = AsyncMock(return_value=_TRAIN_DATA)


class TestSearchNextTrainTimeFilter:
    async def test_start_time_only_excludes_earlier_trains(self):
        with patch("app.agent.tools.tdx_client") as m:
            _wire_train_mock(m)
            result = await search_next_train.ainvoke({
                "from_station": "臺北", "to_station": "花蓮",
                "train_date": "2025-06-15", "start_time": "10:00",
            })
        assert "10:00" in result
        assert "12:00" in result
        assert "08:00" not in result

    async def test_end_time_only_upper_bound_is_exclusive(self):
        frozen = datetime(2025, 6, 15, 9, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_train_mock(m)
                result = await search_next_train.ainvoke({
                    "from_station": "臺北", "to_station": "花蓮",
                    "train_date": "2025-06-15", "end_time": "12:00",
                })
        assert "10:00" in result       # >= 09:00 (current) and < 12:00
        assert "08:00" not in result   # 08:00 < 09:00 (current time)
        assert "12:00" not in result   # upper bound exclusive
        assert "14:00" not in result

    async def test_both_bounds_narrow_window(self):
        with patch("app.agent.tools.tdx_client") as m:
            _wire_train_mock(m)
            result = await search_next_train.ainvoke({
                "from_station": "臺北", "to_station": "花蓮",
                "train_date": "2025-06-15",
                "start_time": "10:00", "end_time": "14:00",
            })
        assert "10:00" in result
        assert "12:00" in result
        assert "08:00" not in result
        assert "14:00" not in result   # upper bound exclusive

    async def test_no_filter_defaults_to_current_time(self):
        frozen = datetime(2025, 6, 15, 11, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_train_mock(m)
                result = await search_next_train.ainvoke({
                    "from_station": "臺北", "to_station": "花蓮",
                    "train_date": "2025-06-15",
                })
        assert "12:00" in result
        assert "14:00" in result
        assert "08:00" not in result
        assert "10:00" not in result


# ── search_next_hsr ───────────────────────────────────────────────────────────

_HSR_DATA = [
    {
        "DailyTrainInfo": {"TrainNo": "0601"},
        "OriginStopTime": {"DepartureTime": "08:00"},
        "DestinationStopTime": {"ArrivalTime": "09:30"},
    },
    {
        "DailyTrainInfo": {"TrainNo": "0603"},
        "OriginStopTime": {"DepartureTime": "10:00"},
        "DestinationStopTime": {"ArrivalTime": "11:30"},
    },
    {
        "DailyTrainInfo": {"TrainNo": "0605"},
        "OriginStopTime": {"DepartureTime": "12:00"},
        "DestinationStopTime": {"ArrivalTime": "13:30"},
    },
    {
        "DailyTrainInfo": {"TrainNo": "0607"},
        "OriginStopTime": {"DepartureTime": "14:00"},
        "DestinationStopTime": {"ArrivalTime": "15:30"},
    },
]


def _wire_hsr_mock(m):
    m.resolve_hsr_station_id = MagicMock(return_value="THSR-ID")
    m.get_hsr_timetable = AsyncMock(return_value=_HSR_DATA)


class TestSearchNextHsrTimeFilter:
    async def test_start_time_only_excludes_earlier_trains(self):
        frozen = datetime(2025, 6, 15, 8, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_hsr_mock(m)
                result = await search_next_hsr.ainvoke({
                    "from_station": "台北", "to_station": "左營",
                    "start_time": "10:00",
                })
        assert "0603" in result     # 10:00 >= start
        assert "0605" in result     # 12:00 >= start
        assert "0607" in result     # 14:00 >= start
        assert "0601" not in result  # 08:00 before start

    async def test_end_time_only_upper_bound_is_exclusive(self):
        frozen = datetime(2025, 6, 15, 9, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_hsr_mock(m)
                result = await search_next_hsr.ainvoke({
                    "from_station": "台北", "to_station": "左營",
                    "end_time": "12:00",
                })
        assert "0603" in result      # 10:00 in [09:00, 12:00)
        assert "0601" not in result   # 08:00 < 09:00 (current)
        assert "0605" not in result   # 12:00 not < 12:00
        assert "0607" not in result

    async def test_both_bounds_narrow_window(self):
        frozen = datetime(2025, 6, 15, 8, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_hsr_mock(m)
                result = await search_next_hsr.ainvoke({
                    "from_station": "台北", "to_station": "左營",
                    "start_time": "10:00", "end_time": "14:00",
                })
        assert "0603" in result      # 10:00 in [10:00, 14:00)
        assert "0605" in result      # 12:00 in [10:00, 14:00)
        assert "0601" not in result   # before start
        assert "0607" not in result   # 14:00 not < 14:00

    async def test_no_filter_defaults_to_current_time(self):
        frozen = datetime(2025, 6, 15, 11, 0, tzinfo=TW_TZ)
        with patch("app.agent.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            with patch("app.agent.tools.tdx_client") as m:
                _wire_hsr_mock(m)
                result = await search_next_hsr.ainvoke({
                    "from_station": "台北", "to_station": "左營",
                })
        assert "0605" in result     # 12:00 >= 11:00
        assert "0607" in result     # 14:00 >= 11:00
        assert "0601" not in result  # 08:00 < 11:00
        assert "0603" not in result  # 10:00 < 11:00
