# Examples

This folder contains runnable SilkLoom Core examples designed for repeatable workflows.

## Files

- `quickstart.py`: Summarize notes and tag themes (`depth_first`)
- `structured_output.py`: Extract structured attributes and build GeoJSON-like features (`breadth_first`)
- `resume_with_run_id.py`: Simulate repeatable tile processing with resume (`run_id`)
- `trajectory_od_commute.py`: OD extraction + distance/time segmentation + flowline output

## Run

From project root:

```bash
python examples/quickstart.py
python examples/structured_output.py
python examples/resume_with_run_id.py
python examples/trajectory_od_commute.py
```

## Notes

- `quickstart.py` and `structured_output.py` require `OPENAI_API_KEY`.
- `trajectory_od_commute.py` also requires `OPENAI_API_KEY`.
- `resume_with_run_id.py` does not require API key and demonstrates that successful tasks are reused when rerunning with the same `run_id`.
- You can adapt these scripts to process interview notes, POI text, trajectory descriptions, or raster tile metadata in any pipeline.
