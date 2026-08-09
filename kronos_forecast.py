"""
kronos_forecast.py — Kronos foundation-model forecast, used as a confirmation layer
on the screener's Buy Top 20 flow.

Kronos (https://github.com/shiyu-coder/Kronos, vendored as a git submodule under
vendor/kronos) predicts future OHLCV candles from a window of historical OHLCV. A
stock only "confirms" if Kronos's forecast close FORECAST_DAYS trading days out is
still above today's close — this is a second, independent opinion layered on top of
the SEPA screen, not a replacement for it.

Model + tokenizer weights are pulled from the Hugging Face Hub on first use (~500MB
for Kronos-base) and cached locally by huggingface_hub. CPU inference for ~20 stocks
in one batched call takes on the order of a minute.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yfinance as yf

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor", "kronos")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

FORECAST_DAYS = 10
CONTEXT_DAYS  = 400            # <= Kronos-base's 512 max_context
TOKENIZER_ID  = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID      = "NeoQuasar/Kronos-base"

_predictor = None


def _get_predictor():
    """Lazy-load the Kronos model/tokenizer once per process."""
    global _predictor
    if _predictor is None:
        from model import Kronos, KronosPredictor, KronosTokenizer
        tokenizer  = KronosTokenizer.from_pretrained(TOKENIZER_ID)
        model      = Kronos.from_pretrained(MODEL_ID)
        _predictor = KronosPredictor(model, tokenizer, max_context=512)
    return _predictor


def _fetch_ohlcv(symbol: str, market: str) -> pd.DataFrame | None:
    """Last CONTEXT_DAYS of daily OHLCV for one symbol, columns lowercased for Kronos."""
    ticker = f"{symbol}.NS" if market == "india" else symbol
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).reset_index()
    df = df.rename(columns={"date": "timestamp", "Date": "timestamp"})
    df = df.tail(CONTEXT_DAYS)
    if len(df) < 60 or df[["open", "high", "low", "close"]].isnull().values.any():
        return None
    return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def forecast_symbols(symbols: list[str], market: str = "india") -> list[dict]:
    """Batched Kronos forecast for each symbol, in the input order. Symbols with
    insufficient/missing history are reported with trend='unknown' rather than
    silently dropped, so the caller can decide how to treat them."""
    predictor = _get_predictor()

    histories: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _fetch_ohlcv(sym, market)
        if df is not None:
            histories[sym] = df

    results = []
    ok_symbols = list(histories.keys())
    if ok_symbols:
        # predict_batch requires equal-length history across the batch — trim all
        # to the shortest window so every symbol runs in a single forward pass.
        min_len = min(len(histories[s]) for s in ok_symbols)
        df_list, x_ts_list, y_ts_list = [], [], []
        for sym in ok_symbols:
            df = histories[sym].tail(min_len).reset_index(drop=True)
            future_ts = pd.bdate_range(df["timestamp"].iloc[-1], periods=FORECAST_DAYS + 1)[1:]
            df_list.append(df[["open", "high", "low", "close", "volume"]])
            x_ts_list.append(df["timestamp"])
            y_ts_list.append(pd.Series(future_ts))

        preds = predictor.predict_batch(
            df_list=df_list, x_timestamp_list=x_ts_list, y_timestamp_list=y_ts_list,
            pred_len=FORECAST_DAYS, T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )

        for sym, df, pred_df in zip(ok_symbols, df_list, preds):
            last_close     = float(df["close"].iloc[-1])
            forecast_close = float(pred_df["close"].iloc[-1])
            pct_change     = (forecast_close / last_close - 1) * 100
            results.append({
                "symbol":         sym,
                "last_close":     round(last_close, 2),
                "forecast_close": round(forecast_close, 2),
                "pct_change":     round(pct_change, 2),
                "trend":          "up" if forecast_close > last_close else "down",
            })

    missing = [s for s in symbols if s not in histories]
    for sym in missing:
        results.append({"symbol": sym, "trend": "unknown", "reason": "insufficient price history"})

    order = {s: i for i, s in enumerate(symbols)}
    results.sort(key=lambda r: order[r["symbol"]])
    return results
