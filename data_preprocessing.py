"""B1: preprocess the M5 Texas stores into one long Parquet table.

Portions of this code are generated with Claude (Opus 5.5 version), a large language model by
Anthropic (Anthropic, 2026), on November 15, 2026. Reviewed,
tested, and run by Shane Darilek. See README.md, "Use of AI", for details.

Pipeline:
    1. sales_train_evaluation.csv -> Texas rows only -> wide to long
    2. join calendar.csv on d        (adds date, wm_yr_wk, events, snap_TX)
    3. join sell_prices.csv on store_id, item_id, wm_yr_wk
    4. drop days before each series' first sale, forward-fill price gaps, events -> "none"
    5. join observed Texas weather on date (wx_* and fc7_* only, never diag_*)
    6. encode categoricals, downcast numerics, write Parquet

Weather note: every row gets the weather from texas_weather_train.csv. Inside a CV or
test window, the model must see the horizon file's weather instead (forecast for days
1-7, climatology for 8-28). Which rows count as "inside the window" depends on the
fold, so that swap happens at split time, not here.

Usage:
    python data_preprocessing.py --m5-dir m5_cache \
        --weather weather_cache/texas_weather_train.csv --out data/tx_long.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
CALENDAR_COLS = ["d", "date", "wm_yr_wk", "wday", "month", "year", *EVENT_COLS, "snap_TX"]


def load_texas_sales(path: Path) -> pd.DataFrame:
    """Read the wide sales file, keep the three Texas stores, reshape to one row per series-day."""
    header = pd.read_csv(path, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    wide = pd.read_csv(path, dtype={c: np.int16 for c in day_cols})
    wide = wide[wide["state_id"] == "TX"].drop(columns="state_id")
    wide[ID_COLS] = wide[ID_COLS].astype("category")

    long = wide.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    long["d_num"] = long["d"].str.slice(2).astype(np.int16)
    long["sales"] = long["sales"].astype(np.int16)
    return long


def add_calendar(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    cal = pd.read_csv(path, usecols=CALENDAR_COLS, parse_dates=["date"])
    out = df.merge(cal, on="d", how="left", validate="many_to_one")
    assert out["date"].notna().all(), "some d values are missing from calendar.csv"
    return out.drop(columns="d")


def add_prices(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, dtype={"sell_price": np.float32})
    prices = prices[prices["store_id"].str.startswith("TX")]
    for col in ["store_id", "item_id"]:
        prices[col] = pd.Categorical(prices[col], categories=df[col].cat.categories)
    return df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left",
                    validate="many_to_one")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Trim pre-launch days, then fill price and event gaps (in that order)."""
    first_sale = df.loc[df["sales"] > 0].groupby("id", observed=True)["d_num"].min()
    never_sold = df["id"].nunique() - len(first_sale)
    df = df[df["d_num"] >= df["id"].map(first_sale).astype("float32")]  # NaN (never sold) -> dropped
    print(f"  series with no sales at all (dropped): {never_sold}")

    df = df.sort_values(["id", "d_num"], ignore_index=True)
    still_missing = df["sell_price"].isna().sum()
    df["sell_price"] = df.groupby("id", observed=True)["sell_price"].ffill()
    # A first sale can land in a week with no price row; borrow the next known price.
    df["sell_price"] = df.groupby("id", observed=True)["sell_price"].bfill()
    print(f"  price gaps filled: {still_missing:,}; still missing: {df['sell_price'].isna().sum():,}")

    df[EVENT_COLS] = df[EVENT_COLS].fillna("none")
    return df


def add_weather(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    wx = pd.read_csv(path, parse_dates=["date"])
    keep = ["date"] + [c for c in wx.columns if c.startswith(("wx_", "fc7_"))]
    wx = wx[keep]
    wx[keep[1:]] = wx[keep[1:]].astype(np.float32)
    out = df.merge(wx, on="date", how="left", validate="many_to_one")
    assert out[keep[1:]].notna().all().all(), "weather has missing values after the join"
    return out


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    df[EVENT_COLS] = df[EVENT_COLS].astype("category")
    df[["wday", "month"]] = df[["wday", "month"]].astype(np.int8)
    df[["year", "wm_yr_wk"]] = df[["year", "wm_yr_wk"]].astype(np.int16)  # wm_yr_wk tops out at 11621
    df["snap_TX"] = df["snap_TX"].astype(np.int8)
    for col in ID_COLS:
        df[col] = df[col].cat.remove_unused_categories()
    return df


def check(df: pd.DataFrame) -> None:
    assert not df.duplicated(["id", "d_num"]).any(), "duplicate series-day rows"
    assert df.isna().sum().sum() == 0, f"NaNs left:\n{df.isna().sum()[lambda s: s > 0]}"
    assert set(df["store_id"].cat.categories) == {"TX_1", "TX_2", "TX_3"}
    print(f"  rows: {len(df):,}   series: {df['id'].nunique():,}")
    print(f"  dates: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--m5-dir", type=Path, default=Path("m5_cache"))
    p.add_argument("--weather", type=Path, default=Path("weather_cache/texas_weather_train.csv"))
    p.add_argument("--out", type=Path, default=Path("data/tx_long.parquet"))
    args = p.parse_args()

    print("1/6 sales: Texas filter, wide to long")
    df = load_texas_sales(args.m5_dir / "sales_train_evaluation.csv")
    print("2/6 calendar")
    df = add_calendar(df, args.m5_dir / "calendar.csv")
    print("3/6 prices")
    df = add_prices(df, args.m5_dir / "sell_prices.csv")
    print("4/6 clean")
    df = clean(df)
    print("5/6 weather")
    df = add_weather(df, args.weather)
    print("6/6 downcast, check, write")
    df = downcast(df)
    check(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False, compression="zstd")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()