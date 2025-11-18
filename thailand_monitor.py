#!/usr/bin/env python3
"""
Thailand 2026 multi-city fare monitor
- Generates candidate itineraries that include Bangkok + 1–2 other Thai cities
- Considers open-jaw and round-trip patterns with domestic hops between cities
- Picks the best itinerary based on price → total travel time → better times
- Writes the best-of-day snapshot to data/thailand_tracker.csv
- Plots run_date vs total_price_usd to plots/thailand_price_trend.png

ENV VARS REQUIRED (GitHub Actions Secrets):
  AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
Optional: HTTP_PROXY, HTTPS_PROXY if needed.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from email.message import EmailMessage
import smtplib
import ssl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from amadeus import Client
from amadeus.client.errors import ClientError, ResponseError

BANGKOK_AIRPORTS = ["BKK", "DMK"]
REPO_ROOT = pathlib.Path(".")


def log(msg: str) -> None:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def load_config(path: pathlib.Path) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def iso_date_range(start: str, end: str) -> List[str]:
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    return [(s + dt.timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


def parse_duration_hours(iso_duration: str) -> float:
    # Supports formats like "PT14H20M" or "P1DT2H"
    hours = 0.0
    num = ""
    in_time = False
    for ch in iso_duration:
        if ch == "T":
            in_time = True
            continue
        if ch.isdigit():
            num += ch
            continue
        if ch == "P":
            continue
        if ch == "D":
            hours += float(num or 0) * 24
            num = ""
        elif ch == "H":
            hours += float(num or 0)
            num = ""
        elif ch == "M":
            hours += float(num or 0) / 60
            num = ""
        elif ch == "S" and in_time:
            hours += float(num or 0) / 3600
            num = ""
    return round(hours, 2)


def time_penalty(ts_iso: str, prefer_window: Dict[str, int]) -> float:
    hour = dt.datetime.fromisoformat(ts_iso).hour
    lo = prefer_window.get("earliest", 6)
    hi = prefer_window.get("latest", 22)
    if lo <= hour <= hi:
        return 0.0
    # Penalize distance from nearest edge
    if hour < lo:
        return float(lo - hour)
    return float(hour - hi)


def segments_summary(segments: List[Dict]) -> Tuple[str, List[str], List[str]]:
    codes = [f"{s['carrierCode']}{s['number']}" for s in segments]
    airlines = sorted({s["carrierCode"] for s in segments})
    layovers = []
    if len(segments) > 1:
        for prev, nxt in zip(segments[:-1], segments[1:]):
            airport = prev["arrival"]["iataCode"]
            arr = dt.datetime.fromisoformat(prev["arrival"]["at"])
            dep = dt.datetime.fromisoformat(nxt["departure"]["at"])
            diff = dep - arr
            mins = int(diff.total_seconds() // 60)
            layovers.append(f"{airport} ({mins}m)")
    return ", ".join(codes), airlines, layovers


@dataclass
class LegChoice:
    origin: str
    dest: str
    date: str
    price: float
    currency: str
    duration_hours: float
    stops: int
    departure_iso: str
    arrival_iso: str
    flight_codes: str
    airlines: List[str]
    layovers: List[str]
    penalty: float


class AmadeusClient:
    def __init__(self):
        self.client = Client(
            client_id=os.environ["AMADEUS_CLIENT_ID"],
            client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
            hostname="production",
        )
        self.cache: Dict[Tuple, Optional[LegChoice]] = {}
        self.calls = 0

    def search_best_leg(
        self,
        origin: str,
        dest: str,
        date: str,
        currency: str,
        travel_class: str,
        adults: int,
        max_stops: int,
        prefer_window: Dict[str, int],
        nonstop_only: bool = False,
    ) -> Optional[LegChoice]:
        key = (
            origin,
            dest,
            date,
            currency,
            travel_class,
            adults,
            max_stops,
            nonstop_only,
            prefer_window.get("earliest", 6),
            prefer_window.get("latest", 22),
        )

        if key in self.cache:
            return self.cache[key]

        try:
            self.calls += 1
            resp = self.client.shopping.flight_offers_search.get(
                originLocationCode=origin,
                destinationLocationCode=dest,
                departureDate=date,
                adults=adults,
                currencyCode=currency,
                travelClass=travel_class,
                max=30,
            )
        except (ClientError, ResponseError) as e:
            status = getattr(getattr(e, "response", None), "status_code", "?")
            body = getattr(getattr(e, "response", None), "body", None)
            log(f"[Amadeus {origin}->{dest} {date}] HTTP {status} body={json.dumps(body or {}, ensure_ascii=False)[:400]}")
            self.cache[key] = None
            return None

        offers = resp.data or []
        choices: List[LegChoice] = []
        for off in offers:
            itinerary = off["itineraries"][0]
            segments = itinerary["segments"]
            stops = max(len(segments) - 1, 0)
            if nonstop_only and stops > 0:
                continue
            if stops > max_stops:
                continue
            price = float(off["price"]["grandTotal"])
            duration = parse_duration_hours(itinerary["duration"])
            dep_iso = segments[0]["departure"]["at"]
            arr_iso = segments[-1]["arrival"]["at"]
            codes, airlines, layovers = segments_summary(segments)
            penalty = time_penalty(dep_iso, prefer_window) + time_penalty(arr_iso, prefer_window)
            choices.append(
                LegChoice(
                    origin=origin,
                    dest=dest,
                    date=date,
                    price=price,
                    currency=off["price"]["currency"],
                    duration_hours=duration,
                    stops=stops,
                    departure_iso=dep_iso,
                    arrival_iso=arr_iso,
                    flight_codes=codes,
                    airlines=airlines,
                    layovers=layovers,
                    penalty=penalty,
                )
            )

        if not choices:
            self.cache[key] = None
            return None
        choices.sort(key=lambda c: (c.price, c.duration_hours, c.penalty, c.stops))
        best_choice = choices[0]
        self.cache[key] = best_choice
        return best_choice


def generate_city_sequences(cfg: Dict) -> List[Tuple[str, ...]]:
    mandatory = cfg.get("mandatory_city", "BKK")
    beaches = cfg.get("beach_cities", [])
    inland = cfg.get("inland_cities", [])
    min_cities = cfg.get("min_cities", 2)
    max_cities = cfg.get("max_cities", 3)

    sequences: List[Tuple[str, ...]] = []
    for beach in beaches:
        base = [mandatory, beach]
        if min_cities <= 2 <= max_cities:
            sequences.append(tuple(base))
            sequences.append(tuple(reversed(base)))
        if max_cities >= 3:
            for inland_city in inland:
                trio = [mandatory, beach, inland_city]
                # Include permutations where BKK might be first or last
                permutations = {
                    tuple(trio),
                    (mandatory, inland_city, beach),
                    (beach, mandatory, inland_city),
                    (beach, inland_city, mandatory),
                    (inland_city, mandatory, beach),
                    (inland_city, beach, mandatory),
                }
                for p in permutations:
                    if mandatory in p and beach in p:
                        sequences.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Tuple[str, ...]] = []
    for seq in sequences:
        if seq in seen:
            continue
        seen.add(seq)
        unique.append(seq)
    return unique


def evenly_spaced_dates(start: dt.date, end: dt.date, steps: int) -> List[dt.date]:
    if steps <= 0:
        return []
    total_days = max((end - start).days, steps)
    interval = max(total_days // (steps + 1), 1)
    return [start + dt.timedelta(days=interval * (i + 1)) for i in range(steps)]


def build_itinerary(
    sequence: Tuple[str, ...],
    depart_date: str,
    return_date: str,
    cfg: Dict,
    client: AmadeusClient,
) -> Optional[Dict]:
    origin = cfg.get("origin", "ORD")
    currency = cfg.get("currency", "USD")
    travel_class = cfg.get("cabin", "ECONOMY")
    adults = int(cfg.get("adults", 1))
    prefer_window = cfg.get("preferred_departure_hours", {"earliest": 6, "latest": 22})
    max_stops_ord = int(cfg.get("max_stopovers_ord_legs", 1))
    domestic_nonstop = bool(cfg.get("domestic_nonstop_only", True))
    price_cap = cfg.get("price_ceiling_usd")

    depart_dt = dt.date.fromisoformat(depart_date)
    return_dt = dt.date.fromisoformat(return_date)
    internal_dates = evenly_spaced_dates(depart_dt, return_dt, len(sequence) - 1)

    legs: List[LegChoice] = []
    total_price = 0.0
    total_duration = 0.0

    pairs = [(origin, sequence[0], depart_date)]
    for d_idx, city in enumerate(sequence[:-1]):
        next_city = sequence[d_idx + 1]
        hop_date = internal_dates[d_idx].isoformat()
        pairs.append((city, next_city, hop_date))
    pairs.append((sequence[-1], origin, return_date))

    def resolve_airports(city_code: str, is_domestic: bool) -> List[str]:
        # For now, we treat "BKK" as a logical Bangkok city code and
        # expand to BKK + DMK only for domestic legs.
        if is_domestic and city_code == "BKK":
            return BANGKOK_AIRPORTS
        return [city_code]

    for idx, (o, d, day) in enumerate(pairs):
        is_domestic = o in sequence and d in sequence
        origin_candidates = resolve_airports(o, is_domestic)
        dest_candidates = resolve_airports(d, is_domestic)

        best_leg: Optional[LegChoice] = None
        for o_code in origin_candidates:
            for d_code in dest_candidates:
                leg_candidate = client.search_best_leg(
                    origin=o_code,
                    dest=d_code,
                    date=day,
                    currency=currency,
                    travel_class=travel_class,
                    adults=adults,
                    max_stops=0 if domestic_nonstop and is_domestic else max_stops_ord,
                    prefer_window=prefer_window,
                    nonstop_only=domestic_nonstop and is_domestic,
                )
                if not leg_candidate:
                    continue
                if (
                    best_leg is None
                    or (
                        leg_candidate.price,
                        leg_candidate.duration_hours,
                        leg_candidate.penalty,
                    )
                    < (best_leg.price, best_leg.duration_hours, best_leg.penalty)
                ):
                    best_leg = leg_candidate

        if not best_leg:
            return None

        legs.append(best_leg)
        total_price += best_leg.price
        total_duration += best_leg.duration_hours
        if price_cap and total_price > float(price_cap):
            log(
                f"Pruned itinerary {sequence} {depart_date}->{return_date} due to price cap "
                f"({total_price:.0f} > {float(price_cap):.0f})"
            )
            return None

    airlines = sorted({a for leg in legs for a in leg.airlines})

    itinerary_id = f"{depart_date}_{'-'.join(sequence)}_{return_date}"
    return {
        "itinerary_id": itinerary_id,
        "sequence": sequence,
        "depart_date": depart_date,
        "return_date": return_date,
        "legs": legs,
        "total_price": round(total_price, 2),
        "total_duration_hours": round(total_duration, 2),
        "airlines": ",".join(airlines),
    }


def best_itinerary(cfg: Dict, client: AmadeusClient) -> Optional[Dict]:
    dep_range = iso_date_range(cfg["departure_window"]["start"], cfg["departure_window"]["end"])
    ret_range = iso_date_range(cfg["return_window"]["start"], cfg["return_window"]["end"])
    sequences = generate_city_sequences(cfg)
    if not sequences:
        log("No candidate city sequences generated; check config.")
        return None

    fast_cfg = cfg.get("fast_test", {})
    fast_enabled = bool(fast_cfg.get("enabled", False))

    if fast_enabled:
        max_dep = int(fast_cfg.get("max_departure_dates", 1))
        max_ret = int(fast_cfg.get("max_return_dates", 1))
        max_seq = int(fast_cfg.get("max_sequences", 2))

        dep_range = dep_range[:max_dep]
        ret_range = ret_range[:max_ret]
        sequences = sequences[:max_seq]

        log(
            f"FAST TEST MODE: limiting to {len(dep_range)} departure date(s), "
            f"{len(ret_range)} return date(s), {len(sequences)} sequence(s)"
        )

    num_dep = len(dep_range)
    num_ret = len(ret_range)
    num_seq = len(sequences)
    log(f"Evaluating {num_dep} departure date(s), {num_ret} return date(s), {num_seq} city sequence(s)")

    approx_itins = sum(1 for dep in dep_range for ret in ret_range if dep < ret) * num_seq
    log(f"Approximate maximum itineraries to try: {approx_itins}")

    itins_tried = 0
    itins_success = 0
    start_time = dt.datetime.utcnow()

    candidates: List[Dict] = []
    for dep in dep_range:
        for ret in ret_range:
            if dep >= ret:
                continue
            for seq_idx, seq in enumerate(sequences, start=1):
                itins_tried += 1
                if itins_tried % 10 == 1:
                    elapsed = (dt.datetime.utcnow() - start_time).total_seconds()
                    log(
                        f"[progress] Tried {itins_tried}/{approx_itins} itineraries "
                        f"so far (elapsed ~{elapsed:.1f}s)"
                    )

                itin = build_itinerary(seq, dep, ret, cfg, client)
                if itin:
                    itins_success += 1
                    candidates.append(itin)

    if not candidates:
        elapsed_total = (dt.datetime.utcnow() - start_time).total_seconds()
        log(
            f"Finished itinerary search: {itins_tried} attempted, {itins_success} successful "
            f"({elapsed_total:.1f}s)"
        )
        return None

    prefer_window = cfg.get("preferred_departure_hours", {"earliest": 6, "latest": 22})

    def itinerary_penalty(itin: Dict) -> float:
        return sum(
            time_penalty(leg.departure_iso, prefer_window) + time_penalty(leg.arrival_iso, prefer_window)
            for leg in itin["legs"]
        )

    candidates.sort(
        key=lambda i: (
            i["total_price"],
            i["total_duration_hours"],
            itinerary_penalty(i),
            len(i["sequence"]),
        )
    )
    elapsed_total = (dt.datetime.utcnow() - start_time).total_seconds()
    log(
        f"Finished itinerary search: {itins_tried} attempted, {itins_success} successful "
        f"({elapsed_total:.1f}s)"
    )
    return candidates[0]


def write_tracker(best: Dict, cfg: Dict) -> pathlib.Path:
    history_path = REPO_ROOT / cfg.get("history_csv", "data/thailand_tracker.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    run_date = dt.date.today().isoformat()
    seq_str = " > ".join(best["sequence"])
    outbound = best["legs"][0]
    inbound = best["legs"][-1]
    row = {
        "run_date": run_date,
        "itinerary_id": best["itinerary_id"],
        "destinations_sequence": seq_str,
        "depart_date": best["depart_date"],
        "return_date": best["return_date"],
        "total_price_usd": best["total_price"],
        "price_per_person_usd": round(best["total_price"] / cfg.get("adults", 1), 2),
        "total_travel_hours": best["total_duration_hours"],
        "outbound_travel_hours": outbound.duration_hours,
        "return_travel_hours": inbound.duration_hours,
        "num_stops_outbound": outbound.stops,
        "num_stops_return": inbound.stops,
        "airlines": best["airlines"],
    }

    if history_path.exists():
        hist = pd.read_csv(history_path, dtype=str)
    else:
        hist = pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist = hist.drop_duplicates(subset=["run_date"], keep="last")
    hist.to_csv(history_path, index=False)
    log(f"Updated tracker: {history_path} ({len(hist)} rows)")
    return history_path


def render_markdown_table(best: Dict) -> str:
    lines = ["| Leg | Flight(s) | Depart → Arrive | Duration | Stops / Layovers |", "| --- | --- | --- | --- | --- |"]
    for leg in best["legs"]:
        dep = dt.datetime.fromisoformat(leg.departure_iso).strftime("%Y-%m-%d %H:%M")
        arr = dt.datetime.fromisoformat(leg.arrival_iso).strftime("%Y-%m-%d %H:%M")
        layovers = ", ".join(leg.layovers) if leg.layovers else "—"
        lines.append(
            f"| {leg.origin} → {leg.dest} | {leg.flight_codes} | {dep} → {arr} | "
            f"{leg.duration_hours:.1f}h | {leg.stops} stop(s); {layovers} |"
        )
    lines.append(
        f"| **Total** |  |  | **${best['total_price']:.0f} ({best['total_duration_hours']:.1f}h)** |  |"
    )
    return "\n".join(lines)


def write_best_markdown(best: Dict, cfg: Dict) -> pathlib.Path:
    outpath = REPO_ROOT / "data" / "thailand_best.md"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = render_markdown_table(best)
    meta = (
        f"## Best Thailand itinerary ({cfg.get('trip_name','')})\n"
        f"Sequence: {' > '.join(best['sequence'])}\n\n"
        f"Depart {best['depart_date']} • Return {best['return_date']}\n\n"
        f"Total price: ${best['total_price']:.2f} • Total travel time: {best['total_duration_hours']:.1f}h\n\n"
    )
    outpath.write_text(meta + table + "\n")
    log(f"Wrote Markdown summary to {outpath}")
    return outpath


def plot_price_trend(history_csv: pathlib.Path, out_path: pathlib.Path) -> Optional[pathlib.Path]:
    if not history_csv.exists():
        log(f"History CSV missing ({history_csv}); skipping plot.")
        return None
    df = pd.read_csv(history_csv)
    if df.empty or "run_date" not in df or "total_price_usd" not in df:
        log("History CSV empty or missing required columns; skipping plot.")
        return None

    df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")
    df["total_price_usd"] = pd.to_numeric(df["total_price_usd"], errors="coerce")
    df = df.dropna(subset=["run_date", "total_price_usd"])
    if df.empty:
        log("No valid rows to plot; skipping.")
        return None
    df = df.sort_values("run_date")

    plt.figure(figsize=(7, 3.5))
    plt.plot(df["run_date"], df["total_price_usd"], marker="o")
    plt.title("Thailand 2026 total trip price trend")
    plt.xlabel("Run date")
    plt.ylabel("Total price (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved price trend plot to {out_path}")
    return out_path


def send_email_notification(
    best: Dict,
    cfg: Dict,
    history_path: pathlib.Path,
    best_md_path: pathlib.Path,
    plot_path: Optional[pathlib.Path],
) -> None:
    email_cfg = cfg.get("email", {})
    if not email_cfg.get("enabled"):
        log("Email disabled in config; skipping notification.")
        return

    recipients = email_cfg.get("recipients") or []
    if not recipients:
        log("Email enabled but no recipients configured; skipping notification.")
        return

    smtp_host = email_cfg.get("smtp_host")
    smtp_port = int(email_cfg.get("smtp_port", 587))
    use_starttls = bool(email_cfg.get("use_starttls", True))
    username_env = email_cfg.get("username_env")
    password_env = email_cfg.get("password_env")
    username = os.environ.get(username_env) if username_env else None
    password = os.environ.get(password_env) if password_env else None

    if not smtp_host:
        log("SMTP host not set; skipping email notification.")
        return
    if username_env and not username:
        log(f"SMTP username missing in env {username_env}; skipping email.")
        return
    if password_env and not password:
        log(f"SMTP password missing in env {password_env}; skipping email.")
        return

    sender = email_cfg.get("sender") or username or "thailand-monitor@example.com"
    subject = (
        f"[Thailand 2026] ${best['total_price']:.0f} • {' > '.join(best['sequence'])}"
        f" ({best['depart_date']} → {best['return_date']})"
    )

    body = "\n".join(
        [
            "Thailand 2026 daily fare update",
            f"Depart {best['depart_date']} → Return {best['return_date']}",
            f"Route: {' > '.join(best['sequence'])}",
            f"Total price: ${best['total_price']:.2f}",
            f"Total travel time: {best['total_duration_hours']:.1f}h",
            "",
            "Legs (Markdown table):",
            render_markdown_table(best),
        ]
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if best_md_path.exists():
        msg.add_attachment(
            best_md_path.read_text(),
            subtype="markdown",
            filename=best_md_path.name,
        )

    if history_path.exists():
        msg.add_attachment(
            history_path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=history_path.name,
        )

    if email_cfg.get("include_plot", True) and plot_path and plot_path.exists():
        msg.add_attachment(
            plot_path.read_bytes(),
            maintype="image",
            subtype="png",
            filename=plot_path.name,
        )

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        if use_starttls:
            server.starttls(context=context)
        if username and password:
            server.login(username, password)
        server.send_message(msg)
    log(f"Sent email notification to {', '.join(recipients)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Monitor Thailand 2026 trip fares")
    ap.add_argument("--config", default="config/thailand_2026.yaml", help="Path to YAML config")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    log("=== Thailand 2026 monitor run started ===")
    log(
        f"Trip name: {cfg.get('trip_name', '(none)')}, origin={cfg.get('origin','ORD')}, "
        f"currency={cfg.get('currency','USD')}, adults={cfg.get('adults',1)}, cabin={cfg.get('cabin','ECONOMY')}"
    )
    dw = cfg.get("departure_window", {})
    rw = cfg.get("return_window", {})
    log(
        f"Departure window: {dw.get('start')} → {dw.get('end')}; "
        f"Return window: {rw.get('start')} → {rw.get('end')}"
    )
    fast_cfg = cfg.get("fast_test", {})
    if fast_cfg:
        log(
            f"Fast test enabled={fast_cfg.get('enabled', False)}; "
            f"max_dep={fast_cfg.get('max_departure_dates', 'n/a')}, "
            f"max_ret={fast_cfg.get('max_return_dates', 'n/a')}, "
            f"max_seq={fast_cfg.get('max_sequences', 'n/a')}"
        )
    client = AmadeusClient()

    best = best_itinerary(cfg, client)
    if not best:
        log("No feasible itinerary found for configured windows.")
        sys.exit(3)

    history_path = write_tracker(best, cfg)
    best_md = write_best_markdown(best, cfg)

    plot_path_cfg = cfg.get("plot_path", "plots/thailand_price_trend.png")
    plot_path = plot_price_trend(history_path, REPO_ROOT / plot_path_cfg)

    print(render_markdown_table(best))
    log(f"Best itinerary ID: {best['itinerary_id']}")
    log(f"Markdown summary: {best_md}")
    log(f"Total Amadeus API calls this run: {client.calls}")

    send_email_notification(best, cfg, history_path, best_md, plot_path)


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        missing = str(e)
        log(f"Missing required environment variable: {missing}")
        sys.exit(2)
    except Exception as e:
        log(f"Fatal error: {e}")
        sys.exit(1)
