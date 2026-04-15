"""Quick start example for SilkLoom Core.

Run:
    python examples/quickstart.py

Requires:
    OPENAI_API_KEY in environment.
"""

from __future__ import annotations

from silkloom_core import FunctionNode, LLMNode, Pipeline


def tag_research_theme(summary: str) -> dict:
    """Assign a simple thematic tag from generated summary text."""
    text = summary.lower()
    if any(word in text for word in ["flood", "rain", "drainage", "inundation"]):
        theme = "urban_flood_risk"
    elif any(word in text for word in ["heat", "temperature", "shade", "cooling"]):
        theme = "urban_heat"
    elif any(word in text for word in ["mobility", "traffic", "transit", "commute"]):
        theme = "urban_mobility"
    else:
        theme = "general_urban_observation"

    return {"theme": theme, "summary_length": len(summary)}


def main() -> None:
    pipeline = Pipeline(
        db_path="gis_quickstart.db",
        execution_mode="depth_first",
        default_workers=4,
    )

    pipeline.add_node(
        LLMNode(
            name="summarize_note",
            prompt_template=(
                "You are assisting urban geography research. "
                "Summarize the following field note into one concise sentence "
                "focusing on spatial phenomenon and possible planning implication: {input.note}"
            ),
            model="gpt-4o-mini",
            max_retries=3,
        )
    )

    pipeline.add_node(
        FunctionNode(
            name="tag_theme",
            func=tag_research_theme,
            kwargs_mapping={"summary": "{summarize_note.text}"},
        )
    )

    inputs = [
        {
            "note": (
                "After 30 minutes of rainfall, water ponding appeared near the underpass "
                "at the east side of Zhongshan Road; pedestrians avoided the crossing."
            )
        },
        {
            "note": (
                "At 14:00 the commercial block had noticeably higher thermal discomfort "
                "than the nearby park corridor with dense tree canopy."
            )
        },
    ]

    run_id = pipeline.run(inputs)
    results = pipeline.export_results(run_id)

    print(f"run_id={run_id}")
    for item in results:
        context = item.get("context", {})
        print(
            {
                "item_index": item.get("item_index"),
                "status": item.get("status"),
                "summary": context.get("summarize_note", {}).get("text"),
                "theme": context.get("tag_theme", {}).get("theme"),
            }
        )


if __name__ == "__main__":
    main()
