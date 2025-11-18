# Thailand 2026 Trip Monitor

This repository monitors flight prices using the Amadeus Flight Offers Search API and produces lightweight artifacts you can track in GitHub. It focuses on a **`thailand_2026`** configuration to build 2–3 city Thailand itineraries around early June 2026.

## Prerequisites
- Python 3.11+
- Amadeus credentials exported as `AMADEUS_CLIENT_ID` and `AMADEUS_CLIENT_SECRET` (set as GitHub Actions secrets in CI)

Install dependencies locally:

```bash
pip install -r requirements.txt
```

## Thailand 2026 configuration
The trip parameters live in `config/thailand_2026.yaml` and can be edited without touching code. Key fields:

| Key | Description |
| --- | --- |
| `departure_window.start` / `.end` | Inclusive outbound date range (YYYY-MM-DD). |
| `return_window.start` / `.end` | Inclusive inbound date range. |
| `origin` | Long-haul origin airport (e.g., ORD). |
| `mandatory_city` | City that must appear (Bangkok/BKK by default). |
| `beach_cities` | Candidate beach stops (HKT/USM/KBV). At least one is always included. |
| `inland_cities` | Optional inland stop candidates (e.g., CNX). |
| `min_cities` / `max_cities` | Total Thai cities to include (2–3 recommended). |
| `cabin` | Travel class passed to Amadeus (`ECONOMY`). |
| `currency` | Pricing currency (USD). |
| `adults` | Traveler count; used for price-per-person math. |
| `max_stopovers_ord_legs` | Max stops allowed on ORD ↔ Thailand legs (1). |
| `domestic_nonstop_only` | Enforce nonstop domestic hops inside Thailand. |
| `price_ceiling_usd` | Discard itineraries priced above this value. |
| `preferred_departure_hours.earliest/latest` | Soft window to penalize very early/late departures/arrivals. |
| `history_csv` | Output path for the best-itinerary tracker CSV. |
| `plot_path` | Output PNG path for the price-trend chart. |

Tweak the windows or city lists to explore different date ranges or destinations. For quick smoke tests, narrow the date windows to a single day to reduce API calls.

## Running the Thailand monitor locally
The Thailand workflow does **not** send email; it just computes and saves artifacts.

```bash
python thailand_monitor.py --config config/thailand_2026.yaml
```

Outputs:
- `data/thailand_tracker.csv`: one row per run with the best itinerary (price, stops, airlines, durations).
- `data/thailand_best.md`: Markdown summary with a leg-by-leg table.
- `plots/thailand_price_trend.png`: run-date vs total-price trend line.

## GitHub Actions
`.github/workflows/daily.yml` installs dependencies and runs only the Thailand monitor (`thailand_monitor.py`) on a daily schedule (13:00 UTC). Updated artifacts (`data/thailand_tracker.csv`, `data/thailand_best.md`, `plots/thailand_price_trend.png`) are committed back to the repo so the time series persist.
