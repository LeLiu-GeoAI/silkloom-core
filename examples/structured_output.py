"""Structured output example for SilkLoom Core.

Run:
    python examples/structured_output.py

Requires:
    OPENAI_API_KEY in environment.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from pydantic import BaseModel, Field

from silkloom_core import FunctionNode, LLMNode, Pipeline


class GeoObservation(BaseModel):
    city: str
    district: str
    topic: str = Field(description="Research theme such as flooding, heat, mobility")
    latitude: float
    longitude: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def build_geojson_feature(
    city: str,
    district: str,
    topic: str,
    latitude: float,
    longitude: float,
) -> dict:
    # Use Shanghai People's Square as a demo CBD anchor.
    cbd_lat, cbd_lon = 31.2304, 121.4737
    dist_km = round(haversine_km(latitude, longitude, cbd_lat, cbd_lon), 3)
    return {
        "feature": {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                "city": city,
                "district": district,
                "topic": topic,
                "distance_to_cbd_km": dist_km,
            },
        }
    }


def main() -> None:
    pipeline = Pipeline(db_path="gis_structured.db", execution_mode="breadth_first", default_workers=8)

    pipeline.add_node(
        LLMNode(
            name="extract_geo",
            prompt_template=(
                "Extract a structured urban observation from this field note. "
                "Return city, district, topic, latitude, longitude. "
                "If coordinates are not explicitly present, infer a plausible representative "
                "location from place names in the note. Field note: {input.note}"
            ),
            response_model=GeoObservation,
            model="gpt-4o-mini",
        )
    )

    pipeline.add_node(
        FunctionNode(
            name="to_feature",
            func=build_geojson_feature,
            kwargs_mapping={
                "city": "{extract_geo.city}",
                "district": "{extract_geo.district}",
                "topic": "{extract_geo.topic}",
                "latitude": "{extract_geo.latitude}",
                "longitude": "{extract_geo.longitude}",
            },
        )
    )

    inputs = [
        {
            "note": (
                "In Shanghai Jing'an district near People's Park, residents reported repeated "
                "waterlogging after evening storms around a low-lying intersection."
            )
        },
        {
            "note": (
                "In Pudong close to Lujiazui office towers, daytime heat stress is strong on "
                "wide paved streets with limited shade."
            )
        },
    ]

    run_id = pipeline.run(inputs)
    results = pipeline.export_results(run_id)

    print(f"run_id={run_id}")
    for row in results:
        context = row.get("context", {})
        print(row["status"], context.get("to_feature", {}).get("feature", {}))


if __name__ == "__main__":
    main()
