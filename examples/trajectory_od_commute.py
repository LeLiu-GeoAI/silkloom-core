"""OD and commute example for SilkLoom Core.

This example demonstrates:
1) origin-destination extraction from text notes
2) distance and time segmentation
3) flowline-friendly output format (GeoJSON-like LineString feature)

Run:
    python examples/trajectory_od_commute.py

Requires:
    OPENAI_API_KEY in environment.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Optional

from pydantic import BaseModel, Field

from silkloom_core import FunctionNode, LLMNode, Pipeline


class CommuteOD(BaseModel):
    city: str
    origin_name: str
    destination_name: str
    origin_latitude: float
    origin_longitude: float
    destination_latitude: float
    destination_longitude: float
    departure_time: str = Field(description="24-hour format HH:MM")
    travel_minutes: int = Field(ge=1, description="Estimated one-way travel time in minutes")
    mode: str = Field(description="Main commute mode, e.g., metro, bus, bike, walk, car")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def _safe_hour(departure_time: str) -> Optional[int]:
    try:
        return int(departure_time.split(":", maxsplit=1)[0])
    except Exception:
        return None


def time_segment(departure_time: str) -> str:
    hour = _safe_hour(departure_time)
    if hour is None:
        return "unknown"
    if 6 <= hour < 10:
        return "morning_peak"
    if 10 <= hour < 16:
        return "daytime"
    if 16 <= hour < 20:
        return "evening_peak"
    return "off_peak"


def distance_segment(distance_km: float) -> str:
    if distance_km < 3:
        return "short_0_3km"
    if distance_km < 8:
        return "medium_3_8km"
    if distance_km < 15:
        return "long_8_15km"
    return "very_long_15km_plus"


def duration_segment(travel_minutes: int) -> str:
    if travel_minutes < 20:
        return "fast_under_20min"
    if travel_minutes < 40:
        return "mid_20_40min"
    if travel_minutes < 60:
        return "slow_40_60min"
    return "very_slow_60min_plus"


def build_flow_feature(
    city: str,
    origin_name: str,
    destination_name: str,
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    departure_time: str,
    travel_minutes: int,
    mode: str,
) -> dict:
    distance_km = round(
        haversine_km(origin_latitude, origin_longitude, destination_latitude, destination_longitude),
        3,
    )

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [origin_longitude, origin_latitude],
                [destination_longitude, destination_latitude],
            ],
        },
        "properties": {
            "city": city,
            "origin_name": origin_name,
            "destination_name": destination_name,
            "mode": mode,
            "departure_time": departure_time,
            "travel_minutes": travel_minutes,
            "distance_km": distance_km,
            "distance_segment": distance_segment(distance_km),
            "time_segment": time_segment(departure_time),
            "duration_segment": duration_segment(travel_minutes),
            "flow_weight": 1,
        },
    }

    return {"flow_feature": feature}


def main() -> None:
    pipeline = Pipeline(db_path="gis_od_commute.db", execution_mode="breadth_first", default_workers=6)

    pipeline.add_node(
        LLMNode(
            name="extract_od",
            prompt_template=(
                "Extract one structured commute OD record from the following note. "
                "Return city, origin_name, destination_name, origin_latitude, origin_longitude, "
                "destination_latitude, destination_longitude, departure_time (HH:MM), "
                "travel_minutes (int), mode. If coordinates are missing, infer plausible coordinates "
                "from place names. Note: {input.note}"
            ),
            response_model=CommuteOD,
            model="gpt-4o-mini",
            max_retries=3,
        )
    )

    pipeline.add_node(
        FunctionNode(
            name="build_flow",
            func=build_flow_feature,
            kwargs_mapping={
                "city": "{extract_od.city}",
                "origin_name": "{extract_od.origin_name}",
                "destination_name": "{extract_od.destination_name}",
                "origin_latitude": "{extract_od.origin_latitude}",
                "origin_longitude": "{extract_od.origin_longitude}",
                "destination_latitude": "{extract_od.destination_latitude}",
                "destination_longitude": "{extract_od.destination_longitude}",
                "departure_time": "{extract_od.departure_time}",
                "travel_minutes": "{extract_od.travel_minutes}",
                "mode": "{extract_od.mode}",
            },
        )
    )

    inputs = [
        {
            "note": (
                "Shanghai commuter: leaves from Minhang Xinzhuang at 07:40, takes metro to "
                "Lujiazui office area in Pudong, around 55 minutes door-to-door."
            )
        },
        {
            "note": (
                "Beijing commuter: starts near Huilongguan at 08:15, rides bus to Zhongguancun "
                "software park, typical travel time about 45 minutes."
            )
        },
        {
            "note": (
                "Shenzhen commuter: departs from Bao'an center at 18:05, drives to Nanshan "
                "Hi-tech Park, around 35 minutes in normal traffic."
            )
        },
    ]

    run_id = pipeline.run(inputs)
    results = pipeline.export_results(run_id)

    flow_features = []
    for item in results:
        if item.get("status") != "success":
            continue
        flow = item.get("context", {}).get("build_flow", {}).get("flow_feature")
        if flow:
            flow_features.append(flow)

    flow_collection = {"type": "FeatureCollection", "features": flow_features}

    print(f"run_id={run_id}")
    print("flow_feature_count=", len(flow_features))
    print(flow_collection)


if __name__ == "__main__":
    main()
