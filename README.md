# Player Tracking Data

Exploratory analysis and feature engineering on hockey **player tracking + event data** for the [Big Data Cup 2026](https://github.com/bigdatacup/Big-Data-Cup-2026). The current focus is building a **feature-based Expected Goals (xG) model** that combines shot location, shot angle, and shooter motion (derived from tracking data) to estimate the probability that a shot becomes a goal.

The methods here follow Jonathan Arsenault's thesis, *[Quantitative Analysis of Hockey Using Spatiotemporal Tracking Data](https://mcgill.scholaris.ca/items/da91938f-30c7-4f75-9ca3-2402a9e5422f)* (McGill University), which develops an xG model from features derived from spatiotemporal tracking data.

## Data

The dataset is Stathletes-tracked hockey event data plus player tracking data generated from broadcast video. Data is **not committed to this repo** — download it from the Big Data Cup release and place it under `data/`:

> https://github.com/bigdatacup/Big-Data-Cup-2026/releases/tag/Data

Each game has several files:

| File | Description |
| --- | --- |
| `<date> Team X @ Team Y Events.csv` | Tracked events (shots, goals, plays, takeaways, zone entries, faceoffs, penalties, …) with coordinates and per-event details |
| `<date> Team X @ Team Y Shifts.csv` | Per-player shift start/end clock times by period |
| `<date> Team X @ Team Y Tracking_P{1,2,3}.csv` | Per-period frame-by-frame player and puck positions (~30 fps) |
| `camera_orientations.csv` | Which side each goalie defends in the 1st period (needed to normalize coordinates) |

See [data/README.md](data/README.md) for the full column-level data dictionary and event definitions.

### Coordinate system
- X coordinate: between -100 and 100, Y coordinate: between -42.5 and 42.5, from the camera's perspective.
- Tracking `Image Id` carries a per-period frame counter (suffix ticks ~30×/game-clock second), used to compute velocity from the actual frame-index gap.

## What's here

- **[inspect-data.ipynb](inspect-data.ipynb)** — the main working notebook:
  - Loads events, shifts, and tracking data with [Polars](https://pola.rs/).
  - Summary tables (shot-type and shot-outcome distributions) rendered with [great-tables](https://posit-dev.github.io/great-tables/).
  - Shot-location visualization on a rink diagram (matplotlib).
  - Feature engineering for the xG model: normalized shot coordinates, distance/angle to net, and shooter speed at the moment of release (matching each shot event to its tracking frame on Period + Team + Jersey number).
- **[camera_orientations.csv](camera_orientations.csv)** — goalie-side index used to normalize shot coordinates onto a single attacking side.
- **[main.py](main.py)** — placeholder entry point.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
uv sync          # create the virtualenv and install dependencies
```

Then open the notebook:

```bash
uv run jupyter lab inspect-data.ipynb
# or run the placeholder entry point
uv run main.py
```


## References

- Jonathan Arsenault, *[Quantitative Analysis of Hockey Using Spatiotemporal Tracking Data](https://mcgill.scholaris.ca/items/da91938f-30c7-4f75-9ca3-2402a9e5422f)*, McGill University — the source for the xG modeling approach used here.
