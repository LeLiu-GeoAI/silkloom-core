"""Resume behavior example using a stable run_id.

This example simulates tiled remote-sensing processing with one transient
tile failure on the first run, then resumes from the same run_id.
It uses FunctionNode only, so no OpenAI API key is required.

Run:
    python examples/resume_with_run_id.py
"""

from __future__ import annotations

from silkloom_core import FunctionNode, Pipeline


_FAILURE_ONCE = {1}


def preprocess_tile(tile_id: int, ndvi: float) -> dict:
    # Simulate one transient tile read failure.
    if tile_id in _FAILURE_ONCE:
        _FAILURE_ONCE.remove(tile_id)
        raise RuntimeError("simulated tile IO failure")
    return {"tile_id": tile_id, "ndvi": ndvi}


def classify_vegetation(tile_id: int, ndvi: float) -> dict:
    if ndvi >= 0.5:
        cls = "high_vegetation"
    elif ndvi >= 0.2:
        cls = "moderate_vegetation"
    else:
        cls = "low_vegetation"
    return {"tile_id": tile_id, "ndvi_class": cls}


def print_results(label: str, results: list[dict]) -> None:
    print(f"\n{label}")
    for item in results:
        context = item.get("context", {})
        print(
            {
                "item_index": item.get("item_index"),
                "status": item.get("status"),
                "last_node": item.get("last_node"),
                "preprocess": context.get("preprocess"),
                "classify": context.get("classify"),
                "errors": len(item.get("errors", [])),
            }
        )


def main() -> None:
    pipeline = Pipeline(db_path="gis_resume.db", execution_mode="depth_first", default_workers=2)

    pipeline.add_node(
        FunctionNode(
            name="preprocess",
            func=preprocess_tile,
            kwargs_mapping={"tile_id": "{input.tile_id}", "ndvi": "{input.ndvi}"},
            max_retries=1,
        ),
        depends_on=[],
    )
    pipeline.add_node(
        FunctionNode(
            name="classify",
            func=classify_vegetation,
            kwargs_mapping={"tile_id": "{preprocess.tile_id}", "ndvi": "{preprocess.ndvi}"},
            max_retries=1,
        ),
        depends_on=["preprocess"],
    )

    inputs = [
        {"tile_id": 0, "ndvi": 0.62},
        {"tile_id": 1, "ndvi": 0.43},
        {"tile_id": 2, "ndvi": 0.11},
    ]
    run_id = "gis-resume-demo-run"

    pipeline.run(inputs, run_id=run_id)
    first = pipeline.export_results(run_id)
    print_results("After first run:", first)

    pipeline.run(inputs, run_id=run_id)
    second = pipeline.export_results(run_id)
    print_results("After resume run:", second)


if __name__ == "__main__":
    main()
