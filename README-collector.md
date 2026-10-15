# B-Bottom-Watch
Public market-data collection only. No account credentials, investment positions, borrowing details, or trading instructions are stored here.

GitHub Actions runs daily at 01:40 UTC and can also be started manually in the Actions tab. It fetches completed daily BTC/USD Bitstamp OHLC candles and publishes `latest.json` **only if data-quality checks pass**. Failures are visible in Actions and do not overwrite the last good snapshot.

The JSON contains the latest completed UTC day, week (Sunday close) and month, monthly RSI(14) with 14-month RSI SMA, weekly EMA ribbon (20/25/30/35/40/45/50/55), and Pi Cycle Bottom (150-day EMA versus 471-day SMA × 0.745). All calculations use standard exchange candles, never Heikin Ashi. A signal can never be interpreted as a recommendation to invest.

A timestamp and candle dates must be checked before consuming `latest.json`; a successful historical run does **not** prove a snapshot is current. An independent ChatGPT scheduled-task reader has not yet been verified.
