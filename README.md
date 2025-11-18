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
By default the monitor will also send an email if SMTP settings are configured (see **Email notifications** below).

```bash
python thailand_monitor.py --config config/thailand_2026.yaml
```

Outputs:
- `data/thailand_tracker.csv`: one row per run with the best itinerary (price, stops, airlines, durations).
- `data/thailand_best.md`: Markdown summary with a leg-by-leg table.
- `plots/thailand_price_trend.png`: run-date vs total-price trend line.

### Email notifications
Email is configured in `config/thailand_2026.yaml` under the `email` block:

```yaml
email:
  enabled: true
  sender: "Thailand Monitor <notifier@example.com>"
  recipients:
    - "your_email@example.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  use_starttls: true
  username_env: "SMTP_USERNAME"
  password_env: "SMTP_PASSWORD"
  include_plot: true
```

Provide SMTP credentials via environment variables referenced by `username_env` / `password_env`, e.g. `export SMTP_USERNAME=...` and `export SMTP_PASSWORD=...`. Set `enabled: false` to skip emails. When enabled, the Markdown summary is attached to the message and the price-trend plot is attached if available.

## GitHub Actions
`.github/workflows/daily.yml` installs dependencies and runs only the Thailand monitor (`thailand_monitor.py`) on a daily schedule (13:00 UTC). Updated artifacts (`data/thailand_tracker.csv`, `data/thailand_best.md`, `plots/thailand_price_trend.png`) are committed back to the repo so the time series persist. Set the following GitHub Actions secrets for automation:

- `AMADEUS_CLIENT_ID` / `AMADEUS_CLIENT_SECRET` (required)
- `SMTP_USERNAME` / `SMTP_PASSWORD` (required if `email.enabled` is true)
