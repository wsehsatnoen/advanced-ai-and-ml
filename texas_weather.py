#!/usr/bin/env python3
"""
texas_weather.py - weather covariates for a Texas-only M5 DeepAR model.

Observed weather comes from NOAA GHCN-Daily for four Texas airports, pulled from
NCEI's Access Data Service (free, no key). Forecasts are the official National
Weather Service Point Forecast Matrices (PFMs) those same airports received at
the time, pulled from the Iowa Environmental Mesonet's archive of NWS text
products (free, no key). A PFM gives 7 days of daily highs and lows plus
12-hour chances of rain, and is issued several times a day by the Fort Worth
(FWD), Houston/Galveston (HGX) and Austin/San Antonio (EWX) offices.

    DFW  Dallas/Fort Worth Intl   office FWD
    IAH  Houston Intercontinental office HGX
    SAT  San Antonio Intl         office EWX
    AUS  Austin-Bergstrom         office EWX

Every feature is a population-weighted average of the four metros (2015 Census
metro estimates), so all 9,147 Texas item-store series share one weather signal.

Features
  wx_*   the day's weather
           history ......... observed
           forecast window . the NWS forecast for days 1-7, climatology after
         wx_tmax_anom, wx_tmin_anom   degrees F vs the 1981-2010 normal
         wx_rain                      rain >= 0.01 in (0/1), or the chance of rain
         wx_freeze, wx_heat           low <= 32F / high >= 100F (0/1, or odds)
  fc7_*  the next 7 days, as the NWS forecast them that morning
         fc7_tmax_anom (mean), fc7_rain (mean chance),
         fc7_freeze (largest share of people under a forecast freeze on any day),
         fc7_heat (same, for a forecast high >= 100F)
  diag_* coverage diagnostics, not model inputs

"That morning" means the latest PFM issued by 6:00 a.m. local time.

Commands
  1) Download (run once, about 1 GB of NWS text, cached and gzipped):
       python texas_weather.py fetch --cache weather_cache
  2) Training features for every M5 calendar day:
       python texas_weather.py build --cache weather_cache \
           --calendar m5/calendar.csv --out texas_weather_train.csv
  3) Forecast-window features for each validation/test window:
       python texas_weather.py horizon --cache weather_cache \
           --calendar m5/calendar.csv --horizon 28 \
           --first-days 2016-02-01,2016-02-29,2016-03-28,2016-04-25 \
           --out texas_weather_horizon.csv

How to use the two outputs
  Join texas_weather_train.csv to every Texas series on `date` and list the
  wx_* and fc7_* columns as known future inputs (in PyTorch Forecasting:
  time_varying_known_reals). For a window starting on day F, overwrite the
  weather columns for dates F..F+27 with the horizon rows for first_day F,
  because the training rows there use weather nobody had on day F. Fit any
  scaling on training dates only.

  Comparisons worth running: no weather; weather with --mode forecast (the
  realistic result); training rows left in place for the window (an
  upper bound with perfect hindsight); and --mode climatology (does the
  forecast add anything over seasonal averages).

Requires Python 3.9+, pandas and numpy.

Data credits
  Observations: NOAA NCEI, Global Historical Climatology Network - Daily.
  Forecasts: NOAA National Weather Service, archived by the Iowa Environmental
  Mesonet (Iowa State University).
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings

STATIONS = {
    # GHCN-Daily id, PFM office, airport location as printed in the PFM, 2015 metro population
    "DFW": dict(ghcn="USW00003927", office="FWD", lat=32.91, lon=-97.03, pop=7_102_796),
    "IAH": dict(ghcn="USW00012960", office="HGX", lat=29.96, lon=-95.35, pop=6_656_947),
    "SAT": dict(ghcn="USW00012921", office="EWX", lat=29.55, lon=-98.47, pop=2_384_075),
    "AUS": dict(ghcn="USW00013904", office="EWX", lat=30.19, lon=-97.67, pop=2_000_860),
}

OBS_START, OBS_END = date(1981, 1, 1), date(2016, 6, 30)
NORMALS_START, NORMALS_END = pd.Timestamp("1981-01-01"), pd.Timestamp("2010-12-31")
PFM_START, PFM_END = date(2011, 1, 1), date(2016, 6, 30)
M5_FIRST_DAY, M5_DAYS = date(2011, 1, 29), 1969        # d_1 .. d_1969

FREEZE_F, HEAT_F, RAIN_IN = 32.0, 100.0, 0.01
LOOKAHEAD_DAYS = 7
MORNING_CUTOFF_HOUR = 6        # use the latest PFM issued by 6:00 a.m. local time
MAX_STALE_HOURS = 36           # older than this counts as no forecast
FEATURES = ["tmax_anom", "tmin_anom", "rain", "freeze", "heat"]

NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
USER_AGENT = "texas-weather-m5/1.0 (student forecasting project)"

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- download

def http_get(url: str, params: dict, retries: int = 4, timeout: int = 300) -> str:
    full = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:300]
            if err.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 30 * attempt
                log(f"  HTTP {err.code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {err.code} from {full}\n{body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt < retries:
                wait = 15 * attempt
                log(f"  {err}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Could not reach {full}: {err}") from None
    raise RuntimeError(f"Gave up on {full}")


def fetch_observations(cache: Path, refresh: bool = False) -> None:
    """GHCN-Daily highs, lows and precipitation for the four airports, one file per year."""
    out_dir = cache / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ",".join(s["ghcn"] for s in STATIONS.values())
    for year in range(OBS_START.year, OBS_END.year + 1):
        path = out_dir / f"ghcn_{year}.csv"
        if path.exists() and not refresh:
            continue
        start = max(date(year, 1, 1), OBS_START)
        end = min(date(year, 12, 31), OBS_END)
        text = http_get(NCEI_URL, {
            "dataset": "daily-summaries", "stations": ids,
            "startDate": start.isoformat(), "endDate": end.isoformat(),
            "dataTypes": "TMAX,TMIN,PRCP", "units": "standard",
            "format": "csv", "includeAttributes": "false",
        })
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        if "STATION" not in first_line.upper():
            raise RuntimeError(f"Unexpected NCEI response for {year}:\n{text[:300]}")
        path.write_text(text, encoding="utf-8")
        log(f"  observations {year}: {max(len(text.splitlines()) - 1, 0)} rows")
        time.sleep(0.5)


def month_starts(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def fetch_forecasts(cache: Path, refresh: bool = False) -> None:
    """Every archived PFM from the three offices, one gzipped file per office per month."""
    out_dir = cache / "pfm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for office in sorted({s["office"] for s in STATIONS.values()}):
        for start in month_starts(PFM_START, PFM_END):
            path = out_dir / f"PFM{office}_{start:%Y-%m}.txt.gz"
            if path.exists() and not refresh:
                continue
            nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
            text = http_get(IEM_URL, {
                "pil": f"PFM{office}", "sdate": start.isoformat(),
                "edate": (nxt + timedelta(days=1)).isoformat(),   # 1-day overlap, deduped later
                "limit": 9999, "fmt": "text",
            })
            if text.lstrip().startswith("ERROR"):
                log(f"  PFM{office} {start:%Y-%m}: none archived ({text.strip()[:80]})")
                text = ""
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(text)
            count = len(WMO_RE.findall(text.replace("\r", "")))
            log(f"  PFM{office} {start:%Y-%m}: {count} products")
            time.sleep(1.0)


# ---------------------------------------------------------------- PFM parsing

WMO_RE = re.compile(r"^FOUS\d\d K([A-Z]{3}) \d{6}(?: [A-Z]{3})?[ \t]*$", re.M)
ISSUE_RE = re.compile(r"^\s*(\d{3,4}) (AM|PM) [A-Z]{3} [A-Z]{3} ([A-Z]{3}) +(\d{1,2}) (\d{4})\s*$", re.M)
LATLON_RE = re.compile(r"(\d{1,2}\.\d+)N\s+(\d{1,3}\.\d+)W")
LOCAL_ROW_RE = re.compile(r"^[A-Z]{3} +[36]HRLY")
HOUR_RE = re.compile(r"(?<!\S)\d{2}(?!\S)")
DATE_RE = re.compile(r"(\d{2})/(\d{2})(?:/(\d{2}))?")
TOKEN_RE = re.compile(r"\S+")
VALUE_ROWS = {"MIN/MAX": "temp", "MAX/MIN": "temp", "POP 12HR": "pop", "QPF 12HR": "qpf"}


def split_products(text: str):
    text = text.replace("\r", "").replace("\x01", "").replace("\x03", "")
    starts = [(m.start(), m.group(1)) for m in WMO_RE.finditer(text)]
    for i, (pos, office) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        yield office, text[pos:end]


def parse_issue_time(product: str):
    m = ISSUE_RE.search(product)
    if not m:
        return None
    hhmm, ampm, mon, day, year = m.groups()
    hhmm = hhmm.zfill(4)
    hour, minute = int(hhmm[:2]), int(hhmm[2:])
    if ampm == "AM" and hour == 12:
        hour = 0
    elif ampm == "PM" and hour != 12:
        hour += 12
    try:
        return datetime(int(year), MONTHS[mon], int(day), hour, minute)
    except (KeyError, ValueError):
        return None


def match_station(office: str, block: str):
    m = LATLON_RE.search(block)
    if not m:
        return None
    lat, lon = float(m.group(1)), -float(m.group(2))
    for code, s in STATIONS.items():
        if s["office"] == office and abs(s["lat"] - lat) <= 0.06 and abs(s["lon"] - lon) <= 0.06:
            return code
    return None


def _number(token: str):
    token = token.strip()
    if not token or token.upper().startswith("M"):
        return None
    try:
        if "-" in token[1:]:                       # QPF ranges like 0.10-0.24
            lo, hi = token.rsplit("-", 1)
            return (float(lo) + float(hi)) / 2
        return float(token)
    except ValueError:
        return None


def _parse_table(date_line, local_line, utc_line, body, issued, out):
    utc = [(m.end(), int(m.group())) for m in HOUR_RE.finditer(utc_line, 9)]
    local = {m.end(): int(m.group()) for m in HOUR_RE.finditer(local_line, 9)}
    if not utc:
        return
    dm = DATE_RE.search(date_line)
    if not dm:
        return
    month, day = int(dm.group(1)), int(dm.group(2))
    if dm.group(3):
        year = 2000 + int(dm.group(3))
    else:
        year = issued.year + (1 if month < issued.month - 6 else 0) - (1 if month > issued.month + 6 else 0)
    try:
        current = date(year, month, day)
    except ValueError:
        return
    # Sanity check: the 3-hourly table starts on the issue date, the 6-hourly one a few days later.
    # Anything else means the layout was misread, and a skipped table beats mislabeled days.
    delta = (current - issued.date()).days
    if ("3HRLY" in local_line and not -1 <= delta <= 1) or ("6HRLY" in local_line and not 1 <= delta <= 5):
        return
    columns = {}                                   # end position -> (utc hour, local date)
    previous = None
    for end, utc_hour in utc:
        hour = local.get(end, local.get(end - 1, local.get(end + 1)))
        if hour is None:
            hour = (utc_hour - 6) % 24             # CST if the local row is unreadable
        if previous is not None and hour < previous:
            current += timedelta(days=1)
        previous = hour
        columns[end] = (utc_hour, current)
    for line in body:
        for label, kind in VALUE_ROWS.items():
            if not line.startswith(label):
                continue
            for tok in TOKEN_RE.finditer(line, len(label)):
                col = columns.get(tok.end()) or columns.get(tok.end() + 1) or columns.get(tok.end() - 1)
                value = _number(tok.group())
                if col is None or value is None:
                    continue
                utc_hour, local_date = col
                if (kind == "temp" and not -30 <= value <= 125) or (kind == "pop" and not 0 <= value <= 100):
                    continue
                rec = out.setdefault(local_date, {})
                if kind == "temp":
                    if utc_hour == 12:
                        rec["tmin"] = value
                    elif utc_hour == 0:
                        rec["tmax"] = value
                elif kind == "pop":
                    if utc_hour == 12:
                        rec["pop_night"] = value     # 12-hour period ending ~7 a.m.
                    elif utc_hour == 0:
                        rec["pop_day"] = value       # 12-hour period ending ~7 p.m.
                elif kind == "qpf":
                    if utc_hour == 12:
                        rec["qpf_night"] = value
                    elif utc_hour == 0:
                        rec["qpf_day"] = value


def parse_block(block: str, issued: datetime) -> dict:
    lines = block.split("\n")
    out: dict = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("DATE"):
            i += 1
            continue
        local_line = utc_line = None
        j = i + 1
        while j < len(lines) and j <= i + 4:
            if LOCAL_ROW_RE.match(lines[j]) and not lines[j].startswith("UTC"):
                local_line = lines[j]
            elif lines[j].startswith("UTC") and "HRLY" in lines[j]:
                utc_line = lines[j]
                break
            j += 1
        if local_line is None or utc_line is None:
            i += 1
            continue
        k = j + 1
        while k < len(lines) and not lines[k].startswith("DATE"):
            k += 1
        _parse_table(lines[i], local_line, utc_line, lines[j + 1:k], issued, out)
        i = k
    return out


def parse_pfm_text(text: str) -> list:
    records = []
    for office, product in split_products(text):
        issued = parse_issue_time(product)
        if issued is None:
            continue
        for block in re.split(r"^\$\$[ \t]*$", product, flags=re.M):
            station = match_station(office, block)
            if station is None:
                continue
            for target, values in parse_block(block, issued).items():
                records.append({"station": station, "office": office, "issued": issued,
                                "target_date": target, **values})
    return records


def load_forecasts(cache: Path) -> pd.DataFrame:
    files = sorted((cache / "pfm").glob("PFM*.txt.gz"))
    if not files:
        sys.exit(f"No forecast files in {cache / 'pfm'}. Run the fetch command first.")
    records = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            records.extend(parse_pfm_text(fh.read()))
    fc = pd.DataFrame(records)
    if fc.empty:
        sys.exit("The forecast files contained no readable PFMs for the four airports.")
    for col in ["tmax", "tmin", "pop_day", "pop_night", "qpf_day", "qpf_night"]:
        if col not in fc:
            fc[col] = np.nan
    fc["issued"] = pd.to_datetime(fc["issued"])
    fc["target_date"] = pd.to_datetime(fc["target_date"])
    fc = (fc.sort_values(["station", "issued", "target_date"])
            .drop_duplicates(["station", "issued", "target_date"], keep="last")
            .reset_index(drop=True))
    return fc


def load_observations(cache: Path) -> pd.DataFrame:
    files = sorted((cache / "obs").glob("ghcn_*.csv"))
    if not files:
        sys.exit(f"No observation files in {cache / 'obs'}. Run the fetch command first.")
    raw = pd.concat([pd.read_csv(p, dtype=str) for p in files if p.stat().st_size > 0], ignore_index=True)
    raw.columns = [c.strip().upper() for c in raw.columns]
    by_id = {s["ghcn"]: code for code, s in STATIONS.items()}
    raw["station"] = raw["STATION"].str.strip().map(by_id)
    unknown = raw.loc[raw["station"].isna(), "STATION"].unique()
    if len(unknown):
        log(f"  ignoring unexpected station ids: {', '.join(unknown[:5])}")
    obs = pd.DataFrame({
        "station": raw["station"],
        "date": pd.to_datetime(raw["DATE"].str.strip()),
        "tmax": pd.to_numeric(raw.get("TMAX"), errors="coerce"),
        "tmin": pd.to_numeric(raw.get("TMIN"), errors="coerce"),
        "prcp": pd.to_numeric(raw.get("PRCP"), errors="coerce"),
    }).dropna(subset=["station"])
    obs = obs.sort_values(["station", "date"]).drop_duplicates(["station", "date"], keep="last")
    missing = set(STATIONS) - set(obs["station"])
    if missing:
        sys.exit(f"No observations for {', '.join(sorted(missing))}; check the GHCN ids in STATIONS.")
    return obs.reset_index(drop=True)


# ---------------------------------------------------------------- climatology

def harmonics(doy, k: int = 3) -> np.ndarray:
    t = 2 * np.pi * np.asarray(doy, dtype=float) / 365.25
    cols = [np.ones_like(t)]
    for i in range(1, k + 1):
        cols += [np.cos(i * t), np.sin(i * t)]
    return np.column_stack(cols)


class Climatology:
    """Smooth day-of-year normals and event odds per airport, fit on 1981-2010."""

    def __init__(self, obs: pd.DataFrame):
        self.coef = {}
        for station, g in obs.groupby("station"):
            base = g[(g["date"] >= NORMALS_START) & (g["date"] <= NORMALS_END)]
            if base["date"].dt.year.nunique() < 10:
                base = g[g["date"] < pd.Timestamp(M5_FIRST_DAY)]
                log(f"  {station}: under 10 years of 1981-2010 data, normals use all years before 2011")
            targets = {
                "tmax": base["tmax"],
                "tmin": base["tmin"],
                "rain": (base["prcp"] >= RAIN_IN).astype(float).where(base["prcp"].notna()),
                "freeze": (base["tmin"] <= FREEZE_F).astype(float).where(base["tmin"].notna()),
                "heat": (base["tmax"] >= HEAT_F).astype(float).where(base["tmax"].notna()),
            }
            doy = base["date"].dt.dayofyear.to_numpy()
            for name, y in targets.items():
                ok = y.notna().to_numpy()
                if ok.sum() < 365 * 5:
                    sys.exit(f"Not enough {name} history for {station} to fit normals.")
                coef, *_ = np.linalg.lstsq(harmonics(doy[ok]), y.to_numpy()[ok].astype(float), rcond=None)
                self.coef[(station, name)] = coef

    def value(self, station: str, name: str, dates) -> np.ndarray:
        doy = pd.DatetimeIndex(dates).dayofyear
        v = harmonics(doy) @ self.coef[(station, name)]
        return np.clip(v, 0.0, 1.0) if name in ("rain", "freeze", "heat") else v

    def features(self, station: str, dates) -> pd.DataFrame:
        dates = pd.DatetimeIndex(dates)
        return pd.DataFrame({
            "tmax_anom": 0.0, "tmin_anom": 0.0,
            "rain": self.value(station, "rain", dates),
            "freeze": self.value(station, "freeze", dates),
            "heat": self.value(station, "heat", dates),
        }, index=dates)


# ---------------------------------------------------------------- features

def observed_features(obs_station: pd.DataFrame, clim: Climatology, station: str) -> pd.DataFrame:
    d = obs_station.index
    tmax, tmin, prcp = obs_station["tmax"], obs_station["tmin"], obs_station["prcp"]
    return pd.DataFrame({
        "tmax_anom": tmax - clim.value(station, "tmax", d),
        "tmin_anom": tmin - clim.value(station, "tmin", d),
        "rain": (prcp >= RAIN_IN).astype(float).where(prcp.notna()),
        "freeze": (tmin <= FREEZE_F).astype(float).where(tmin.notna()),
        "heat": (tmax >= HEAT_F).astype(float).where(tmax.notna()),
    }, index=d)


def forecast_features(fc: pd.DataFrame, clim: Climatology) -> pd.DataFrame:
    """Turn raw PFM rows into the same five daily features as the observations."""
    parts = []
    for station, g in fc.groupby("station"):
        d = pd.DatetimeIndex(g["target_date"])
        pop = np.fmax(g["pop_day"].to_numpy(dtype=float), g["pop_night"].to_numpy(dtype=float))
        parts.append(pd.DataFrame({
            "station": station, "office": g["office"].to_numpy(), "issued": g["issued"].to_numpy(),
            "target_date": d,
            "tmax_anom": g["tmax"].to_numpy(dtype=float) - clim.value(station, "tmax", d),
            "tmin_anom": g["tmin"].to_numpy(dtype=float) - clim.value(station, "tmin", d),
            "rain": pop / 100.0,
            "freeze": np.where(g["tmin"].isna(), np.nan, (g["tmin"] <= FREEZE_F).astype(float)),
            "heat": np.where(g["tmax"].isna(), np.nan, (g["tmax"] >= HEAT_F).astype(float)),
        }))
    return pd.concat(parts, ignore_index=True)


def weighted(frames: dict) -> pd.DataFrame:
    """Population-weighted average across airports, skipping airports with no value."""
    w = pd.Series({st: float(STATIONS[st]["pop"]) for st in frames})
    out = {}
    for f in FEATURES:
        vals = pd.DataFrame({st: frames[st][f] for st in frames})
        wsum = vals.notna().mul(w, axis=1).sum(axis=1)
        out[f] = vals.fillna(0.0).mul(w, axis=1).sum(axis=1) / wsum.replace(0.0, np.nan)
    return pd.DataFrame(out)


def morning_issuance(fc: pd.DataFrame, dates) -> pd.DataFrame:
    """For each date and office: the latest PFM issued by the morning cutoff, if fresh enough."""
    anchors = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    anchors["cutoff"] = anchors["date"] + pd.Timedelta(hours=MORNING_CUTOFF_HOUR)
    anchors = anchors.sort_values("cutoff")
    out = []
    for office in sorted({s["office"] for s in STATIONS.values()}):
        times = pd.DataFrame({"issued": np.sort(fc.loc[fc["office"] == office, "issued"].unique())})
        if times.empty:
            continue
        m = pd.merge_asof(anchors, times, left_on="cutoff", right_on="issued",
                          direction="backward", tolerance=pd.Timedelta(hours=MAX_STALE_HOURS))
        m["office"] = office
        out.append(m[["date", "office", "issued"]])
    return pd.concat(out, ignore_index=True)


class ForecastIndex:
    """Fast lookup of one issuance's daily features for one airport."""

    def __init__(self, fc_feat: pd.DataFrame, clim: Climatology):
        self.clim = clim
        self.groups = {key: g.set_index("target_date")[FEATURES]
                       for key, g in fc_feat.groupby(["station", "issued"])}

    def station_days(self, station: str, issued, targets):
        """Forecast values for target dates, climatology wherever the forecast is silent.
        Returns (features, share of cells that came from the forecast)."""
        targets = pd.DatetimeIndex(targets)
        fallback = self.clim.features(station, targets)
        frame = self.groups.get((station, issued)) if issued is not None and not pd.isna(issued) else None
        if frame is None:
            return fallback, 0.0
        got = frame.reindex(targets)
        covered = float(got.notna().to_numpy().mean())
        return got.combine_first(fallback)[FEATURES], covered


def summarize_lookahead(days: pd.DataFrame) -> dict:
    return {
        "fc7_tmax_anom": days["tmax_anom"].mean(),
        "fc7_rain": days["rain"].mean(),
        "fc7_freeze": days["freeze"].max(),
        "fc7_heat": days["heat"].max(),
    }


def m5_calendar(path) -> pd.DataFrame:
    if path:
        cal = pd.read_csv(path, usecols=["date", "d"], parse_dates=["date"])
    else:
        dates = pd.date_range(M5_FIRST_DAY, periods=M5_DAYS, freq="D")
        cal = pd.DataFrame({"date": dates, "d": [f"d_{i}" for i in range(1, M5_DAYS + 1)]})
    return cal.sort_values("date").reset_index(drop=True)


def station_offices():
    return {st: s["office"] for st, s in STATIONS.items()}


def build_training(obs, fc, clim, calendar) -> pd.DataFrame:
    dates = pd.DatetimeIndex(calendar["date"])
    obs_by = {st: g.set_index("date").reindex(dates) for st, g in obs.groupby("station")}
    wx = weighted({st: observed_features(obs_by[st], clim, st) for st in STATIONS})
    wx_cov = pd.DataFrame({st: obs_by[st][["tmax", "tmin", "prcp"]].notna().all(axis=1) for st in STATIONS})
    w = pd.Series({st: float(STATIONS[st]["pop"]) for st in STATIONS})

    fc_feat = forecast_features(fc, clim)
    index = ForecastIndex(fc_feat, clim)
    picks = morning_issuance(fc, dates).set_index(["date", "office"])["issued"]
    offices = station_offices()

    rows = []
    for t in dates:
        targets = pd.date_range(t + pd.Timedelta(days=1), periods=LOOKAHEAD_DAYS, freq="D")
        frames, cover = {}, {}
        for st, office in offices.items():
            issued = picks.get((t, office))
            frames[st], cover[st] = index.station_days(st, issued, targets)
        summary = summarize_lookahead(weighted(frames))
        summary["diag_fc7_coverage"] = sum(cover[st] * w[st] for st in cover) / w.sum()
        rows.append(summary)
    fc7 = pd.DataFrame(rows, index=dates)

    out = pd.concat([wx.add_prefix("wx_"), fc7], axis=1)
    out["diag_wx_coverage"] = wx_cov.mul(w, axis=1).sum(axis=1) / w.sum()
    out.insert(0, "date", dates)
    out.insert(1, "d", calendar["d"].to_numpy())
    return out.reset_index(drop=True)


def build_horizon(fc, clim, calendar, first_days, horizon, mode) -> pd.DataFrame:
    fc_feat = forecast_features(fc, clim)
    index = ForecastIndex(fc_feat, clim)
    offices = station_offices()
    d_lookup = dict(zip(pd.DatetimeIndex(calendar["date"]), calendar["d"]))
    rows = []
    for first in first_days:
        first = pd.Timestamp(first)
        picks = morning_issuance(fc, [first]).set_index("office")["issued"] if mode == "forecast" else pd.Series(dtype=object)
        issued_by_station = {st: picks.get(office) for st, office in offices.items()}
        if mode == "forecast":
            got = {o: str(picks.get(o)) for o in sorted(set(offices.values()))}
            log(f"  window {first.date()}: PFMs used {got}")
        for lead, t in enumerate(pd.date_range(first, periods=horizon, freq="D")):
            today, la, used = {}, {}, 0.0
            for st in offices:
                today[st], cov = index.station_days(st, issued_by_station[st], [t])
                used += cov * STATIONS[st]["pop"]
                targets = pd.date_range(t + pd.Timedelta(days=1), periods=LOOKAHEAD_DAYS, freq="D")
                la[st], _ = index.station_days(st, issued_by_station[st], targets)
            wx_today = weighted(today).iloc[0]
            share = used / sum(s["pop"] for s in STATIONS.values())
            row = {"first_day": first.date(), "date": t.date(), "d": d_lookup.get(t), "lead": lead,
                   **{f"wx_{f}": wx_today[f] for f in FEATURES},
                   **summarize_lookahead(weighted(la)),
                   "wx_source": "forecast" if share > 0.5 else "climatology"}
            rows.append(row)
    return pd.DataFrame(rows)


def accuracy_report(obs, fc) -> pd.DataFrame:
    """Forecast error by lead for the morning PFMs, a check that parsing worked."""
    dates = pd.date_range(fc["issued"].min().normalize(), fc["issued"].max().normalize(), freq="D")
    picks = morning_issuance(fc, dates).dropna(subset=["issued"])
    chosen = fc.merge(picks[["office", "issued", "date"]], on=["office", "issued"], how="inner")
    chosen["lead"] = (chosen["target_date"] - chosen["date"]).dt.days
    joined = chosen.merge(obs.rename(columns={"date": "target_date", "tmax": "obs_tmax",
                                              "tmin": "obs_tmin", "prcp": "obs_prcp"}),
                          on=["station", "target_date"], how="inner")
    joined = joined[(joined["lead"] >= 0) & (joined["lead"] <= LOOKAHEAD_DAYS)]
    pop = np.fmax(joined["pop_day"], joined["pop_night"]) / 100
    rain = (joined["obs_prcp"] >= RAIN_IN).astype(float).where(joined["obs_prcp"].notna())
    joined = joined.assign(err_max=(joined["tmax"] - joined["obs_tmax"]).abs(),
                           err_min=(joined["tmin"] - joined["obs_tmin"]).abs(),
                           brier=(pop - rain) ** 2)
    return (joined.groupby("lead")
                  .agg(high_MAE_F=("err_max", "mean"), low_MAE_F=("err_min", "mean"),
                       rain_brier=("brier", "mean"), forecasts=("target_date", "size"))
                  .round(2))


# ---------------------------------------------------------------- commands

def _rounded(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(4)
    return out


def cmd_fetch(args):
    cache = Path(args.cache)
    log("Observations (NOAA GHCN-Daily via NCEI)...")
    fetch_observations(cache, args.refresh)
    log("Forecasts (NWS Point Forecast Matrices via Iowa Environmental Mesonet)...")
    fetch_forecasts(cache, args.refresh)
    log(f"Done. Cached under {cache.resolve()}")


def _load(args):
    cache = Path(args.cache)
    log("Loading observations...")
    obs = load_observations(cache)
    log("Fitting 1981-2010 normals...")
    clim = Climatology(obs)
    log("Parsing NWS forecasts (takes a few minutes the first time)...")
    fc = load_forecasts(cache)
    per_year = fc.groupby(fc["issued"].dt.year)["issued"].nunique()
    log("  forecast issuances per year: " + ", ".join(f"{y}: {n}" for y, n in per_year.items()))
    return obs, clim, fc


def cmd_build(args):
    obs, clim, fc = _load(args)
    calendar = m5_calendar(args.calendar)
    log("Building daily features...")
    table = build_training(obs, fc, clim, calendar)
    _rounded(table).to_csv(args.out, index=False)
    log(f"Wrote {len(table)} days to {args.out}")
    log(f"  days with observations: {table['diag_wx_coverage'].gt(0).mean():.1%}; "
        f"mean forecast coverage: {table['diag_fc7_coverage'].mean():.1%}")
    log("\nNWS forecast error by lead (days after the morning forecast):")
    log(accuracy_report(obs, fc).to_string())


def cmd_horizon(args):
    obs, clim, fc = _load(args)
    calendar = m5_calendar(args.calendar)
    first_days = [d.strip() for d in args.first_days.split(",") if d.strip()]
    log(f"Building {args.horizon}-day windows ({args.mode}) for {', '.join(first_days)}...")
    table = build_horizon(fc, clim, calendar, first_days, args.horizon, args.mode)
    _rounded(table).to_csv(args.out, index=False)
    log(f"Wrote {len(table)} rows to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download observations and archived NWS forecasts")
    f.add_argument("--cache", default="weather_cache")
    f.add_argument("--refresh", action="store_true", help="download again even if cached")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("build", help="training features for every M5 calendar day")
    b.add_argument("--cache", default="weather_cache")
    b.add_argument("--calendar", help="M5 calendar.csv (optional; defaults to d_1..d_1969)")
    b.add_argument("--out", default="texas_weather_train.csv")
    b.set_defaults(func=cmd_build)

    h = sub.add_parser("horizon", help="weather as known on the first day of each forecast window")
    h.add_argument("--cache", default="weather_cache")
    h.add_argument("--calendar", help="M5 calendar.csv (optional)")
    h.add_argument("--first-days", required=True, help="comma-separated first forecast days, YYYY-MM-DD")
    h.add_argument("--horizon", type=int, default=28)
    h.add_argument("--mode", choices=["forecast", "climatology"], default="forecast")
    h.add_argument("--out", default="texas_weather_horizon.csv")
    h.set_defaults(func=cmd_horizon)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
