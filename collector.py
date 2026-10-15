#!/usr/bin/env python3
"""Public Bitstamp BTCUSD standard-candle data, with freshness checks."""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
UTC = dt.timezone.utc

def fetch_daily():
    end = int(dt.datetime.now(UTC).timestamp())
    parts = []
    with requests.Session() as session:
        for _ in range(10):
            response = session.get(URL, params={
                "step": 86400, "limit": 1000, "end": end,
                "exclude_current_candle": "true",
            }, timeout=35)
            response.raise_for_status()
            payload = response.json().get("data", {}).get("ohlc", [])
            if not payload:
                raise RuntimeError("Bitstamp returned no OHLC candles")
            f = pd.DataFrame(payload)
            f["date"] = pd.to_datetime(pd.to_numeric(f["timestamp"]), unit="s", utc=True)
            for col in ("open", "high", "low", "close"):
                f[col] = pd.to_numeric(f[col], errors="raise")
            parts.append(f[["date", "open", "high", "low", "close"]])
            oldest = f["date"].min()
            if oldest <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3650):
                break
            end = int(oldest.timestamp()) - 1
            time.sleep(0.2)
    else:
        raise RuntimeError("Insufficient historical daily candles")
    return pd.concat(parts, ignore_index=True)

def rsi_wilder(prices, n=14):
    diff = prices.diff()
    gains = diff.clip(lower=0)
    losses = -diff.clip(upper=0)
    out = pd.Series(float("nan"), index=prices.index)
    if len(prices) < n + 15:
        raise ValueError("Insufficient monthly history for RSI and its SMA")
    g = gains.iloc[1:n + 1].mean()
    l = losses.iloc[1:n + 1].mean()
    def value(a, b):
        return 100.0 if b == 0 and a > 0 else (50.0 if b == 0 else 100 - 100 / (1 + a / b))
    out.iloc[n] = value(g, l)
    for i in range(n + 1, len(prices)):
        g = (g * (n - 1) + gains.iloc[i]) / n
        l = (l * (n - 1) + losses.iloc[i]) / n
        out.iloc[i] = value(g, l)
    return out

def main():
    now = dt.datetime.now(UTC)
    df = fetch_daily()
    df = df.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    today = pd.Timestamp(now.date(), tz="UTC")
    df = df[df.index < today] # Never include an unfinished daily candle.
    if len(df) < 1000:
        raise ValueError("Insufficient candle history")
    if not (df.index == df.index.normalize()).all():
        raise ValueError("Non-midnight daily candle timestamps")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Missing OHLC fields")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC prices")
    if (df.high < df[["open", "close", "low"]].max(axis=1)).any() or (df.low > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC range")
    missing = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC").difference(df.index)
    critical = missing[missing >= df.index[-1] - pd.Timedelta(days=600)]
    if len(critical):
        raise ValueError("Missing recent daily candles: " + ", ".join(str(x.date()) for x in critical[-8:]))
    lag = (today - df.index[-1]).days
    if lag > 2:
        raise ValueError(f"Stale history: last daily candle is {df.index[-1].date()}")
    close = df.close
    short = close.ewm(span=150, adjust=False).mean()
    long = close.rolling(471, min_periods=471).mean() * 0.745
    bottom = (long > short) & (long.shift(1) <= short.shift(1))
    week = close.resample("W-SUN").last()
    # Exclude current unfinished week, even when today's UTC date is Sunday.
    last_closed_sunday = today - pd.Timedelta(days=((today.weekday() + 1) % 7 or 7))
    week = week[week.index <= last_closed_sunday]
    month = close.resample("ME").last()
    month = month[month.index < pd.Timestamp(now.year, now.month, 1, tz="UTC")]
    if len(week) < 60 or len(month) < 35:
        raise ValueError("Insufficient historical weekly/monthly candles")
    rsi = rsi_wilder(month)
    rsi_avg = rsi.rolling(14, min_periods=14).mean()
    ribbon = {str(n): week.ewm(span=n, adjust=False).mean() for n in (20, 25, 30, 35, 40, 45, 50, 55)}
    values = {k: round(float(v.iloc[-1]), 4) for k, v in ribbon.items()}
    orders = (20, 25, 30, 35, 40, 45, 50, 55)
    result = {
        "source": "Bitstamp BTCUSD public OHLC; standard non-Heikin-Ashi daily candles",
        "generated_utc": now.isoformat(),
        "last_completed_candles": {
            "day": str(df.index[-1].date()),
            "week_ending_sunday": str(week.index[-1].date()),
            "month_ending": str(month.index[-1].date())
        },
        "quality": {
            "daily_candle_count": int(len(df)),
            "recent_missing_days": int(len(critical)),
            "old_missing_days": int(len(missing) - len(critical)),
            "last_day_lag": int(lag)
        },
        "monthly_rsi": {
            "rsi_14": round(float(rsi.iloc[-1]), 4),
            "sma_14_of_rsi": round(float(rsi_avg.iloc[-1]), 4),
            "above": bool(rsi.iloc[-1] > rsi_avg.iloc[-1]),
            "bullish_crossover_last_month": bool(
                rsi.iloc[-1] > rsi_avg.iloc[-1] and rsi.iloc[-2] <= rsi_avg.iloc[-2]
            )
        },
        "weekly_ema_ribbon": {
            "ema_values": values,
            "completed_week_close": round(float(week.iloc[-1]), 2),
            "close_above_all": bool(week.iloc[-1] > max(values.values())),
            "fully_bullish_ordered": bool(all(values[str(a)] > values[str(b)] for a, b in zip(orders, orders[1:])))
        },
        "pi_cycle_bottom": {
            "ema_150": round(float(short.iloc[-1]), 4),
            "sma_471_times_0_745": round(float(long.iloc[-1]), 4),
            "crossed_today": bool(bottom.iloc[-1]),
            "recent_historical_crossovers": [str(i.date()) for i in bottom[bottom].index[-10:]]
        }
    }
    Path("latest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"COLLECTOR FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
