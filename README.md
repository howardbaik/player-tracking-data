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

- **[feature-based-expected-goals.ipynb](feature-based-expected-goals.ipynb)** — the main working notebook, end to end from raw data to a trained xG model. Code behind the *[Feature-Based Expected Goals (xG)](https://passthepuck.dev/posts/feature-based-xg/)* blog post.
  - **Load** — reads every game's Events and Tracking files directly from the Big Data Cup 2026 release (no local download) with [Polars](https://pola.rs/), tagging each row with a `Game` id so shot↔tracking joins don't cross-match between games.
  - **Explore** — shot-type and shot-outcome distributions as [great-tables](https://posit-dev.github.io/great-tables/) summary tables, plus a shot-location rink diagram (matplotlib) colored by shot angle and shaped by outcome.
  - **Feature engineering** — six groups of per-shot features, joined into one table on `shot_id`:
    - *Shot Location* — shot angle, meridian distance (`d_m`), and distance to net, on coordinates normalized to a single attacking side.
    - *Shooter Motion* — shooter speed at release (player closest to the shot, from tracking).
    - *Pressure* — distance to the closest and second-closest defenders.
    - *Pre-Shot Movement* — time since the puck crossed the meridian, where it crossed, and average puck speed in the second before the shot.
    - *Traffic* — total occlusion (density of traffic in the shooting lane).
    - *Goaltender Positioning* — angle discrepancy and goaltender depth.
  - **Model** — an [XGBoost](https://xgboost.readthedocs.io/) classifier evaluated with `StratifiedGroupKFold` (5-fold, grouped by game) out-of-fold predictions, scored on log loss, Brier score, and ROC AUC against a base-rate baseline.
  - **Interpret** — [SHAP](https://shap.readthedocs.io/) feature importance (mean |SHAP value|) and dependence plots.
- **[camera_orientations.csv](camera_orientations.csv)** — goalie-side index used to normalize shot coordinates onto a single attacking side.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
uv sync          # create the virtualenv and install dependencies
```

The notebook fetches the data straight from the Big Data Cup 2026 release, so no local `data/` download is needed to run it.


## References

- Jonathan Arsenault, *[Quantitative Analysis of Hockey Using Spatiotemporal Tracking Data](https://mcgill.scholaris.ca/items/da91938f-30c7-4f75-9ca3-2402a9e5422f)*, McGill University — the source for the xG modeling approach used here.
