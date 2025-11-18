#!/usr/bin/env python3
"""
ORD → Europe nonstop fare monitor
- Queries Amadeus Flight Offers (nonStop=True) for fixed dates
- Writes today's snapshot to prices.csv
- Appends/maintains a de-duplicated time-series at data/prices_history.csv
- Renders one PNG line chart per destination in charts/
- Emails an HTML summary with inline charts via Gmail SMTP

ENV VARS REQUIRED (set in GitHub Actions "Secrets and variables → Actions"):
  AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
  SMTP_USER, SMTP_PASS              # Gmail address + App Password
  TO_EMAIL                          # recipient
  FROM_EMAIL (optional)             # defaults to SMTP_USER
  SMTP_HOST (optional, default smtp.gmail.com)
  SMTP_PORT (optional, default 587)

Author: you + ChatGPT
"""

import os, csv, base64, datetime, ssl, smtplib, pathlib, sys, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

from amadeus import Client
from amadeus.client.errors import ClientError, ResponseError

from dateutil import tz
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

# ========= USER CONFIG =========
DEPARTURE = "2026-03-22"
RETURN    = "2026-04-02"
ORIGIN    = "ORD"
# Full ORD→Europe nonstop list (edit to your shortlist to cut API calls)
DESTS = [
    "AMS","BCN","BEG","BRU","CPH","DUB","FRA","IST","KRK","LIS",
    "LHR","MAD","MXP","MUC","CDG","KEF","FCO","VIE","WAW","ZRH"
]

# Email / SMTP
TO_EMAIL   = os.getenv("TO_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL") or os.getenv("SMTP_USER")
SMTP_HOST  = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER  = os.getenv("SMTP_USER")
SMTP_PASS  = os.getenv("SMTP_PASS")

SEND_EMAIL = True  # set False to disable sending (still writes files)

# Paths
REPO_ROOT = pathlib.Path(".")
DATA_DIR  = REPO_ROOT / "data"
CHART_DIR = REPO_ROOT / "charts"
HIST_CSV  = DATA_DIR / "prices_history.csv"
TODAY_CSV = REPO_ROOT / "prices.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ========= AMADEUS CLIENT =========
# Use Sandbox or Production based on which keys you provide.
amadeus = Client(
    client_id=os.environ["AMADEUS_CLIENT_ID"],
    client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
    hostname="production",   # ✅ not "api.amadeus.com"
)

# ========= HELPERS =========

def central_time_now_iso():
    return datetime.datetime.now(tz.gettz("America/Chicago")).strftime("%Y-%m-%d %H:%M")

def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)

def amadeus_status_and_body(exc):
    """Extract HTTP status and JSON body safely from Amadeus exceptions."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    body = getattr(resp, "body", None)
    return status, body

def short_error_tag(exc):
    """Create a compact error tag for CSV/email."""
    status, body = amadeus_status_and_body(exc)
    tag = f"ERROR {status}" if status else f"ERROR {type(exc).__name__}"
    # Try to add first error title/detail from Amadeus payload
    try:
        errs = (body or {}).get("errors", [])
        if errs:
            title = errs[0].get("title") or errs[0].get("code") or ""
            detail = errs[0].get("detail") or ""
            extra = (f" – {title}: {detail}").strip()
            if extra and extra != "– :":
                tag += extra[:170]
    except Exception:
        pass
    return tag

def best_nonstop(dest):
    """
    Return dict with best nonstop price & metadata:
        {'destination', 'price_usd', 'airlines', 'out_departure', 'ret_departure', 'seats_left'}
    Return {'_error': '...'} on API error.
    Return None if 200 OK but no offers found.
    """
    try:
        resp = amadeus.shopping.flight_offers_search.get(
            originLocationCode=ORIGIN,
            destinationLocationCode=dest,
            departureDate=DEPARTURE,
            returnDate=RETURN,
            adults=1,
            nonStop='true',
            currencyCode="USD",
            max=20
        )
    except (ClientError, ResponseError) as e:
        status, body = amadeus_status_and_body(e)
        log(f"[Amadeus {dest}] HTTP {status} {type(e).__name__} body={json.dumps(body or {}, ensure_ascii=False)[:500]}")
        return {"_error": short_error_tag(e)}

    offers = resp.data or []
    if not offers:
        log(f"[Amadeus {dest}] 200 OK but no nonstop offers returned.")
        return None

    # Choose the lowest total price
    offers.sort(key=lambda o: float(o["price"]["grandTotal"]))
    o = offers[0]
    carriers = sorted({seg["carrierCode"]
                       for itin in o["itineraries"] for seg in itin["segments"]})
    out0 = o["itineraries"][0]["segments"][0]["departure"]["at"]
    ret0 = o["itineraries"][1]["segments"][0]["departure"]["at"] if len(o["itineraries"]) > 1 else ""
    return {
        "destination": dest,
        "price_usd": float(o["price"]["grandTotal"]),
        "airlines": ",".join(carriers),
        "out_departure": out0,
        "ret_departure": ret0,
        "seats_left": o.get("numberOfBookableSeats", "")
    }

def write_today_csv(rows):
    with open(TODAY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date_checked","origin","destination","price_usd","airlines",
            "out_departure","ret_departure","seats_left"
        ])
        writer.writeheader()
        writer.writerows(rows)
    log(f"Wrote {TODAY_CSV} with {len(rows)} rows")

def append_and_clean_history(rows):
    """Append today's rows to history, coerce types, and keep 1 row per (date,dest)."""
    new_df = pd.DataFrame(rows)
    if HIST_CSV.exists():
        hist = pd.read_csv(HIST_CSV, dtype=str)
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df

    # Type fixes
    hist["date_checked"] = pd.to_datetime(hist["date_checked"], errors="coerce").dt.date.astype("string")
    # Sort so the "first" we keep is the lowest price for that (date,dest)
    # Coerce price to numeric for sort (errors -> NaN so they sort after)
    _tmp = hist.copy()
    _tmp["price_num"] = pd.to_numeric(_tmp["price_usd"], errors="coerce")

    _tmp = (_tmp
            .sort_values(["date_checked", "destination", "price_num"], na_position="last")
            .drop_duplicates(["date_checked", "destination"], keep="first")
            .drop(columns=["price_num"]))

    _tmp.to_csv(HIST_CSV, index=False)
    log(f"Updated history at {HIST_CSV} with {len(_tmp)} total rows")
    return _tmp

import matplotlib.dates as mdates

def make_charts(hist_df):
    """
    Render one line chart per destination with readable date ticks:
    - x-axis shows month/day only (MM/DD)
    - vertical tick labels
    - tick density adapts to number of points
    Returns: [(dest, cid, path), ...]
    """
    cids = []

    df = hist_df.copy()
    df["price_num"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["date_checked"] = pd.to_datetime(df["date_checked"], errors="coerce")

    for dest in DESTS:
        sub = df[(df["destination"] == dest) & (~df["price_num"].isna())].sort_values("date_checked")
        if sub.empty:
            continue

        x = sub["date_checked"].dt.floor("D")
        y = sub["price_num"].to_numpy()

        plt.figure(figsize=(6, 3))
        plt.plot(x, y, marker="o")
        plt.title(f"{ORIGIN} → {dest} (nonstop)\n{DEPARTURE} → {RETURN}")
        plt.xlabel("Date checked")
        plt.ylabel("Lowest RT price (USD)")

        ax = plt.gca()

        n = len(x)
        if n == 1:
            # Single data point: set a small window around it so the tick renders once
            center = x.iloc[0]
            ax.set_xlim(center - pd.Timedelta(days=1), center + pd.Timedelta(days=1))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.set_xticks([center])
        else:
            # Aim for ~6–8 ticks max
            interval = max(n // 8, 1)
            # Put ticks exactly on your sampled data points to avoid duplicate-looking labels
            ax.set_xticks(x.iloc[::interval])
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            # Tighten limits to your data (+/- 1 day margin so labels don't clip)
            ax.set_xlim(x.min() - pd.Timedelta(days=1), x.max() + pd.Timedelta(days=1))

        # Month/Day only, vertical labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(axis="x", labelrotation=90)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = CHART_DIR / f"{dest}.png"
        plt.savefig(outpath, dpi=150)
        plt.close()

        cid = f"cid_{dest.lower()}@charts"
        cids.append((dest, cid, outpath))

    log(f"Rendered {len(cids)} chart(s) to {CHART_DIR}/")
    return cids



def build_email_bodies(today_rows, chart_cids):
    """Return (text_body, html_body)."""
    priced = []
    for r in today_rows:
        try:
            p = float(r["price_usd"])
            priced.append((r["destination"], p, r["airlines"]))
        except Exception:
            pass
    priced.sort(key=lambda x: x[1])

    # Text (fallback)
    lines = [
        f"ORD → Europe (nonstop) {DEPARTURE} → {RETURN}",
        f"Checked {central_time_now_iso()} CT",
        ""
    ]
    if priced:
        lines.append("Top 10 lowest today:")
        for dest, p, carriers in priced[:10]:
            lines.append(f"  {dest}: ${p:.0f} via {carriers}")
    else:
        lines.append("No priced nonstop results today.")
    # Errors/missing
    errors = [r for r in today_rows if isinstance(r["price_usd"], str) and r["price_usd"].startswith("ERROR")]
    missing = [r["destination"] for r in today_rows if r["price_usd"] == ""]
    if errors:
        lines += ["", "Errors:"]
        for r in errors:
            lines.append(f'  {r["destination"]}: {r["price_usd"]}')
    if missing:
        lines += ["", "No nonstop result returned for: " + ", ".join(missing)]
    text_body = "\n".join(lines)

    # HTML summary table
    rows_html = "".join([
        f"<tr><td>{d}</td><td>${p:,.0f}</td><td>{a}</td></tr>"
        for d, p, a in priced[:10]
    ]) or '<tr><td colspan="3">No priced nonstop results today</td></tr>'

    imgs_html = ""
    for dest, cid, _path in sorted(chart_cids, key=lambda x: x[0]):
        imgs_html += f"""
        <div style="display:inline-block;margin:6px;">
          <div style="font:12px sans-serif;text-align:center;margin-bottom:2px;">{dest}</div>
          <img src="cid:{cid}" alt="{dest}" style="width:360px;height:auto;border:1px solid #ddd;border-radius:4px;">
        </div>
        """

    # Errors/missing (HTML)
    errors_html = ""
    if errors:
        items = "".join([f"<li><code>{e['destination']}</code>: {e['price_usd']}</li>" for e in errors])
        errors_html = f"<p style='color:#a33'><b>Errors</b></p><ul>{items}</ul>"
    missing_html = ""
    if missing:
        missing_html = f"<p style='color:#555'><b>No nonstop result</b> for: {', '.join(missing)}</p>"

    html_body = f"""
    <html>
      <body style="font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
        <h2 style="margin:0 0 8px 0;">ORD → Europe (nonstop)</h2>
        <div>{DEPARTURE} → {RETURN} • Checked {central_time_now_iso()} CT</div>

        <p style="margin-top:14px;margin-bottom:6px;"><b>Lowest 10 today</b></p>
        <table cellpadding="6" cellspacing="0" border="0" style="border-collapse:collapse;font-size:14px;">
          <thead><tr><th align="left">Dest</th><th align="left">Price</th><th align="left">Airlines</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>

        {errors_html}
        {missing_html}

        <p style="margin:16px 0 6px 0;"><b>Price history charts</b> (one per destination)</p>
        {imgs_html or '<div style="color:#777;">Charts will appear after at least one successful run with prices.</div>'}

        <p style="margin-top:16px;">CSV attached: <code>prices.csv</code> (today). History persists at <code>data/prices_history.csv</code>.</p>
      </body>
    </html>
    """
    return text_body, html_body

def send_gmail(subject, html_body, text_body, attachments=None, inline_images=None):
    """Send email with inline images + attachments via Gmail SMTP."""
    if not all([SMTP_USER, SMTP_PASS, TO_EMAIL, FROM_EMAIL]):
        log("Email skipped: missing SMTP or address env vars.")
        return

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    for cid, path in (inline_images or []):
        with open(path, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
        msg.attach(img)

    for path in (attachments or []):
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    log("Email sent via Gmail SMTP.")

# ========= MAIN =========

def main():
    today = datetime.date.today().isoformat()
    rows = []

    # Query each destination
    for d in DESTS:
        info = best_nonstop(d)
        if info is None:  # 200 OK but no offers
            rows.append({
                "date_checked": today, "origin": ORIGIN, "destination": d,
                "price_usd": "", "airlines": "", "out_departure": "", "ret_departure": "", "seats_left": ""
            })
        elif isinstance(info, dict) and info.get("_error"):
            rows.append({
                "date_checked": today, "origin": ORIGIN, "destination": d,
                "price_usd": info["_error"], "airlines": "", "out_departure": "", "ret_departure": "", "seats_left": ""
            })
        else:
            rows.append({
                "date_checked": today, "origin": ORIGIN, "destination": d,
                "price_usd": f'{info["price_usd"]:.2f}',
                "airlines": info["airlines"],
                "out_departure": info["out_departure"],
                "ret_departure": info["ret_departure"],
                "seats_left": info["seats_left"]
            })

    write_today_csv(rows)
    hist = append_and_clean_history(rows)
    chart_cids = make_charts(hist)

    text_body, html_body = build_email_bodies(rows, chart_cids)

    if SEND_EMAIL:
        inline_imgs = [(cid, str(path)) for _, cid, path in chart_cids]
        send_gmail(
            subject=f"[Fare Check] ORD→Europe nonstop {DEPARTURE}–{RETURN}",
            html_body=html_body,
            text_body=text_body,
            attachments=[str(TODAY_CSV)],
            inline_images=inline_imgs
        )

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
