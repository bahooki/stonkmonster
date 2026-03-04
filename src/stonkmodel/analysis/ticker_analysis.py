from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from stonkmodel.data.fmp import FMPClient

ProgressCallback = Callable[[float, str], None]


def _emit_progress(callback: ProgressCallback | None, pct: float, message: str) -> None:
    if callback is None:
        return
    callback(max(0.0, min(100.0, float(pct))), str(message))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= 1e-12:
        return float(default)
    out = float(num) / float(den)
    if not np.isfinite(out):
        return float(default)
    return out


def _clean_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "", str(symbol or "").strip().upper())


@dataclass
class TickerAnalysisConfig:
    years_of_data: int = 10
    dcf_duration_years: int = 5
    terminal_growth_rate: float = 0.02
    benchmark_symbol: str = "VOO"


class TickerAnalyzer:
    """Port of legacy tickerAnalysis logic on top of the project's FMP client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://financialmodelingprep.com/stable",
        request_workers: int = 8,
        config: TickerAnalysisConfig | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("FMP API key is required for ticker analysis")
        self.client = FMPClient(api_key=api_key, base_url=base_url)
        self.request_workers = max(1, int(request_workers))
        self.config = config or TickerAnalysisConfig()

    @staticmethod
    def _geo_growth(yrsdata: int, records: list[dict[str, Any]], field: str) -> tuple[float, int]:
        if yrsdata <= 0:
            return 0.0, 0
        geogrowth = 1.0
        num_skip = 0
        for yr in range(0, yrsdata):
            if (yr + 1) >= len(records):
                num_skip = yrsdata - yr
                break
            start = _safe_float(records[yr + 1].get(field), default=np.nan)
            end = _safe_float(records[yr].get(field), default=np.nan)
            if (not np.isfinite(start)) or (not np.isfinite(end)) or start <= 0 or end <= 0:
                num_skip = yrsdata - yr
                break
            geogrowth = (end / start) * geogrowth
        geoyrsdata = max(0, int(yrsdata - num_skip))
        return float(geogrowth), geoyrsdata

    @staticmethod
    def _monthly_returns(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["date", "ret"])
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["date", "ret"])
        if "date" not in frame.columns and "datetime" in frame.columns:
            frame["date"] = frame["datetime"]
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True)
        frame = frame.dropna(subset=["date"]).copy()
        if frame.empty:
            return pd.DataFrame(columns=["date", "ret"])
        if "adjClose" in frame.columns:
            close = pd.to_numeric(frame["adjClose"], errors="coerce")
        elif "close" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce")
        else:
            close = pd.Series(np.nan, index=frame.index)
        frame["close_value"] = close
        frame = frame.dropna(subset=["close_value"]).sort_values("date").copy()
        if frame.empty:
            return pd.DataFrame(columns=["date", "ret"])
        monthly = frame.set_index("date")["close_value"].resample("ME").last().dropna().to_frame("close")
        monthly["ret"] = monthly["close"].pct_change()
        monthly = monthly.dropna(subset=["ret"]).reset_index()[["date", "ret"]]
        return monthly

    def _estimate_beta(self, symbol: str, benchmark_symbol: str) -> float:
        end = date.today()
        start = end - timedelta(days=int(365.25 * 5.2))
        stock_rows = self.client.get_historical_price_eod(symbol, start=start, end=end)
        bench_rows = self.client.get_historical_price_eod(benchmark_symbol, start=start, end=end)
        stock = self._monthly_returns(stock_rows).rename(columns={"ret": "stock_ret"})
        bench = self._monthly_returns(bench_rows).rename(columns={"ret": "bench_ret"})
        if stock.empty or bench.empty:
            return 1.0
        merged = stock.merge(bench, on="date", how="inner").dropna()
        if len(merged) < 12:
            return 1.0
        var = float(np.var(merged["bench_ret"], ddof=0))
        if not np.isfinite(var) or var <= 1e-12:
            return 1.0
        cov = float(np.cov(merged["stock_ret"], merged["bench_ret"], ddof=0)[0, 1])
        beta = cov / var
        if not np.isfinite(beta):
            return 1.0
        return float(np.clip(beta, -3.0, 5.0))

    def _estimate_risk_free_rate(self) -> float:
        default_rate = 0.0425
        try:
            rows = self.client.get_market_risk_premium()
        except Exception:
            return default_rate
        if not rows:
            return default_rate
        first = rows[0]
        value = _safe_float(first.get("riskFreeRate"), default=np.nan)
        if not np.isfinite(value):
            return default_rate
        if value > 1.0:
            value = value / 100.0
        return float(np.clip(value, 0.0, 0.2))

    def _compute_rsi_ppo(self, symbol: str) -> tuple[float, float]:
        end = date.today()
        start = end - timedelta(days=210)
        rows = self.client.get_historical_price_eod(symbol, start=start, end=end)
        if not rows:
            return float("nan"), float("nan")
        frame = pd.DataFrame(rows)
        if frame.empty:
            return float("nan"), float("nan")
        if "date" not in frame.columns and "datetime" in frame.columns:
            frame["date"] = frame["datetime"]
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True)
        if "adjClose" in frame.columns:
            close = pd.to_numeric(frame["adjClose"], errors="coerce")
        else:
            close = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["close_value"] = close
        frame = frame.dropna(subset=["date", "close_value"]).sort_values("date")
        if len(frame) < 30:
            return float("nan"), float("nan")
        series = frame["close_value"].astype(float)
        rsi = float(RSIIndicator(close=series, window=14).rsi().iloc[-1])
        ema_fast = series.ewm(span=12, adjust=False).mean()
        ema_slow = series.ewm(span=26, adjust=False).mean()
        ppo = float((((ema_fast - ema_slow) / ema_slow) * 100.0).iloc[-1])
        if not np.isfinite(rsi):
            rsi = float("nan")
        if not np.isfinite(ppo):
            ppo = float("nan")
        return rsi, ppo

    def analyze_ticker(
        self,
        symbol: str,
        risk_free_rate: float | None = None,
        benchmark_symbol: str | None = None,
    ) -> dict[str, Any]:
        ticker = _clean_symbol(symbol)
        if not ticker:
            return {"symbol": str(symbol), "error": "invalid_symbol"}

        benchmark = _clean_symbol(benchmark_symbol or self.config.benchmark_symbol) or "VOO"
        quote = self.client.get_quote(ticker)
        qtr_income = self.client.get_income_statement(ticker, period="quarter", limit=12)
        ann_income = self.client.get_income_statement(ticker, period="annual", limit=15)
        qtr_balance = self.client.get_balance_sheet_statement(ticker, period="quarter", limit=4)
        qtr_cash = self.client.get_cash_flow_statement(ticker, period="quarter", limit=12)
        ann_cash = self.client.get_cash_flow_statement(ticker, period="annual", limit=15)

        if not quote:
            return {"symbol": ticker, "error": "no_quote_data"}

        price = _safe_float(quote.get("price"), default=0.0)
        market_cap = _safe_float(quote.get("marketCap"), default=0.0)
        shares_out = _safe_float(
            quote.get("sharesOutstanding"),
            default=_safe_float(quote.get("sharesOutStanding"), default=0.0),
        )
        yrly_high = _safe_float(quote.get("yearHigh"), default=np.nan)
        yrly_low = _safe_float(quote.get("yearLow"), default=np.nan)
        fifty_avg = _safe_float(quote.get("priceAvg50"), default=np.nan)
        two_hundred_avg = _safe_float(quote.get("priceAvg200"), default=np.nan)

        latest_balance = qtr_balance[0] if qtr_balance else {}
        total_liab = _safe_float(latest_balance.get("totalLiabilities"), default=0.0)
        total_stock_equity = _safe_float(latest_balance.get("totalStockholdersEquity"), default=0.0)
        total_debt = _safe_float(latest_balance.get("totalDebt"), default=0.0)
        cash = _safe_float(latest_balance.get("cashAndCashEquivalents"), default=0.0)
        pref_stock = _safe_float(latest_balance.get("preferredStock"), default=0.0)
        latest_qtr_date = latest_balance.get("date")

        ttm_n = int(min(4, len(qtr_income), len(qtr_cash)))
        prior_n = int(min(4, max(0, len(qtr_income) - 4)))

        four_q_rev = 0.0
        prior_four_q_rev = 0.0
        four_q_gross = 0.0
        prior_four_q_gross = 0.0
        four_q_income = 0.0
        four_q_interest = 0.0
        four_q_tax = 0.0
        four_q_pretax = 0.0
        four_q_ebit = 0.0
        four_q_operating_profit = 0.0
        prior_four_q_operating_profit = 0.0
        depreciation = 0.0
        prior_depreciation = 0.0
        ocf = 0.0

        for qtr in range(ttm_n):
            inc = qtr_income[qtr]
            csh = qtr_cash[qtr]
            four_q_rev += _safe_float(inc.get("revenue"), default=0.0)
            four_q_gross += _safe_float(inc.get("grossProfit"), default=0.0)
            four_q_income += _safe_float(inc.get("netIncome"), default=0.0)
            four_q_interest += _safe_float(inc.get("interestExpense"), default=0.0)
            dep_q = _safe_float(inc.get("depreciationAndAmortization"), default=0.0)
            depreciation += dep_q
            four_q_ebit += _safe_float(inc.get("ebitda"), default=0.0) - dep_q
            four_q_operating_profit += _safe_float(inc.get("grossProfit"), default=0.0) - _safe_float(
                inc.get("operatingExpenses"),
                default=0.0,
            ) - dep_q
            four_q_tax += _safe_float(inc.get("incomeTaxExpense"), default=0.0)
            four_q_pretax += _safe_float(inc.get("incomeBeforeTax"), default=0.0)
            ocf += _safe_float(csh.get("netCashProvidedByOperatingActivities"), default=0.0)

        for qtr in range(prior_n):
            idx = qtr + 4
            inc = qtr_income[idx]
            prior_four_q_rev += _safe_float(inc.get("revenue"), default=0.0)
            prior_four_q_gross += _safe_float(inc.get("grossProfit"), default=0.0)
            prior_dep_q = _safe_float(inc.get("depreciationAndAmortization"), default=0.0)
            prior_depreciation += prior_dep_q
            prior_four_q_operating_profit += _safe_float(inc.get("grossProfit"), default=0.0) - _safe_float(
                inc.get("operatingExpenses"),
                default=0.0,
            ) - prior_dep_q

        if len(ann_income) > self.config.years_of_data:
            yrsdata = self.config.years_of_data
        else:
            yrsdata = max(0, len(ann_income) - 1)

        gm = _safe_ratio(four_q_gross, four_q_rev, default=0.0) if four_q_rev > 0 else 0.0
        gp = _safe_ratio(four_q_operating_profit, four_q_rev, default=0.0) if four_q_rev > 0 else 0.0
        yoy_rev = _safe_ratio(four_q_rev, prior_four_q_rev, default=0.0) - 1.0 if prior_four_q_rev > 0 else 0.0
        yoy_gm = 0.0
        if prior_four_q_gross > 0 and prior_four_q_rev > 0 and four_q_gross > 0 and four_q_rev > 0:
            yoy_gm = _safe_ratio((four_q_gross / four_q_rev), (prior_four_q_gross / prior_four_q_rev), default=0.0) - 1.0

        int_cov = _safe_ratio(four_q_ebit, four_q_interest, default=np.nan)
        taxrate = _safe_ratio(four_q_tax, four_q_pretax, default=0.0)
        taxrate = max(0.0, taxrate)

        georevgrowth_factor, rev_yrs = self._geo_growth(yrsdata, ann_income, "revenue")
        georevgrowth = ((georevgrowth_factor ** (1.0 / rev_yrs)) - 1.0) if rev_yrs > 0 else 0.0

        if georevgrowth > 0.30:
            rev_val = 15.0 * four_q_rev
        elif georevgrowth > 0.25:
            rev_val = 12.0 * four_q_rev
        elif georevgrowth > 0.20:
            rev_val = 9.0 * four_q_rev
        elif georevgrowth > 0.18:
            rev_val = 5.0 * four_q_rev
        elif georevgrowth > 0.15:
            rev_val = 4.0 * four_q_rev
        elif georevgrowth > 0.10:
            rev_val = 2.5 * four_q_rev
        else:
            rev_val = four_q_rev

        geocf_factor, ocf_yrs = self._geo_growth(yrsdata, ann_cash, "netCashProvidedByOperatingActivities")
        ocf_growth = ((geocf_factor ** (1.0 / ocf_yrs)) - 1.0) if ocf_yrs > 0 else 0.0

        beta = self._estimate_beta(ticker, benchmark)
        rf = float(risk_free_rate) if risk_free_rate is not None else self._estimate_risk_free_rate()
        rf = float(np.clip(rf, 0.0, 0.2))
        capm = rf + beta * (0.08 - rf)
        debt_cost = rf + 0.03
        capital_den = total_stock_equity + total_liab
        if capital_den > 0:
            wacc = (total_stock_equity / capital_den) * capm + (total_liab / capital_den) * debt_cost * (1.0 - taxrate)
        else:
            wacc = capm
        wacc = float(np.clip(wacc, 0.02, 0.35))

        ocf_val = 0.0
        if ocf > 0 and ocf_yrs >= 3:
            dcf = 0.0
            duration = int(self.config.dcf_duration_years)
            for yr in range(1, duration + 1):
                dcf += (ocf * ((1.0 + ocf_growth) ** yr)) / ((1.0 + wacc) ** yr)
            denom = max(1e-9, wacc - float(self.config.terminal_growth_rate))
            term_dcf = (
                (ocf * ((1.0 + ocf_growth) ** duration)) * (1.0 + float(self.config.terminal_growth_rate)) / denom
            ) / ((1.0 + wacc) ** duration)
            ocf_val = dcf + term_dcf + cash - total_debt - pref_stock

        if shares_out > 0:
            target_price = ((ocf_val * 0.60) + (rev_val * 0.40)) / shares_out * 0.90
        else:
            target_price = 0.0

        if yoy_rev > 0.50:
            target_price *= 1.20
        elif yoy_rev > 0.25:
            target_price *= 1.15
        elif yoy_rev > 0.10:
            target_price *= 1.10
        elif yoy_rev > 0.0:
            pass
        elif yoy_rev > -0.10:
            target_price *= (1.0 + yoy_rev)
        else:
            target_price *= 0.80

        if gm > 0.70:
            target_price *= 1.15
        elif gm > 0.60:
            target_price *= 1.10
        elif gm > 0.50:
            target_price *= 1.05
        elif gm > 0.40:
            pass
        elif gm > 0.30:
            target_price *= 0.90
        elif gm > 0.20:
            target_price *= 0.80
        else:
            target_price *= 0.70

        target_price = max(0.0, float(target_price))
        upside = _safe_ratio(target_price, price, default=np.nan) - 1.0 if price > 0 else np.nan
        pe_ttm = _safe_ratio(market_cap, four_q_income, default=np.nan) if four_q_income > 0 else np.nan

        gdx = None
        if np.isfinite(fifty_avg) and np.isfinite(two_hundred_avg) and np.isfinite(price):
            if fifty_avg > two_hundred_avg:
                prefix = "Gv" if price < fifty_avg else "G^"
                denom = two_hundred_avg - yrly_high
                ratio = _safe_ratio(two_hundred_avg - price, denom, default=np.nan)
            else:
                prefix = "Dv" if price < fifty_avg else "D^"
                denom = two_hundred_avg - yrly_low
                ratio = -_safe_ratio(two_hundred_avg - price, denom, default=np.nan)
            if np.isfinite(ratio):
                gdx = f"{prefix} {ratio:.2f}"
            else:
                gdx = prefix

        rsi14, ppo_12_26 = self._compute_rsi_ppo(ticker)
        signal = "Buy" if np.isfinite(upside) and upside > 0 else "Sell/Neutral"
        confidence = float(np.clip(abs(_safe_float(upside, default=0.0)), 0.0, 1.0))

        return {
            "symbol": ticker,
            "price": float(price),
            "target_price": float(round(target_price, 4)),
            "upside_pct": float(upside) if np.isfinite(upside) else np.nan,
            "signal": signal,
            "signal_strength": confidence,
            "latest_quarter_date": latest_qtr_date,
            "revenue_ttm": float(four_q_rev),
            "gross_margin_ttm": float(gm),
            "operating_profit_margin_ttm": float(gp),
            "yoy_revenue_growth": float(yoy_rev),
            "yoy_gross_margin_change": float(yoy_gm),
            "pe_ttm": float(pe_ttm) if np.isfinite(pe_ttm) else np.nan,
            "interest_coverage": float(int_cov) if np.isfinite(int_cov) else np.nan,
            "operating_cashflow_ttm": float(ocf),
            "revenue_growth_years": int(rev_yrs),
            "revenue_geometric_growth": float(georevgrowth),
            "ocf_growth_years": int(ocf_yrs),
            "ocf_geometric_growth": float(ocf_growth),
            "beta_5y": float(beta),
            "risk_free_rate": float(rf),
            "wacc": float(wacc),
            "rsi_14": float(rsi14) if np.isfinite(rsi14) else np.nan,
            "ppo_12_26": float(ppo_12_26) if np.isfinite(ppo_12_26) else np.nan,
            "gdx_ratio": gdx,
            "benchmark_symbol": benchmark,
            "error": "",
        }

    def analyze_many(
        self,
        symbols: list[str],
        workers: int | None = None,
        risk_free_rate: float | None = None,
        benchmark_symbol: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        cleaned = []
        seen: set[str] = set()
        for raw in symbols:
            sym = _clean_symbol(raw)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            cleaned.append(sym)
        if not cleaned:
            return pd.DataFrame(columns=["symbol", "error"])

        use_workers = max(1, int(workers or self.request_workers))
        resolved_rf = float(risk_free_rate) if risk_free_rate is not None else self._estimate_risk_free_rate()
        total = len(cleaned)
        out_rows: list[dict[str, Any]] = []
        done = 0
        _emit_progress(progress_callback, 0.0, f"Starting ticker analysis for {total} symbols")

        with ThreadPoolExecutor(max_workers=use_workers) as executor:
            futures = {
                executor.submit(
                    self.analyze_ticker,
                    sym,
                    risk_free_rate=resolved_rf,
                    benchmark_symbol=benchmark_symbol,
                ): sym
                for sym in cleaned
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "symbol": sym,
                        "price": np.nan,
                        "target_price": np.nan,
                        "upside_pct": np.nan,
                        "signal": "error",
                        "signal_strength": 0.0,
                        "error": str(exc),
                    }
                out_rows.append(row)
                done += 1
                _emit_progress(progress_callback, (done / total) * 100.0, f"Analyzed {done}/{total}: {sym}")

        frame = pd.DataFrame(out_rows)
        if frame.empty:
            return frame
        if "upside_pct" in frame.columns:
            frame["upside_pct"] = pd.to_numeric(frame["upside_pct"], errors="coerce")
        if "signal_strength" in frame.columns:
            frame["signal_strength"] = pd.to_numeric(frame["signal_strength"], errors="coerce")
        return frame.sort_values(["upside_pct", "signal_strength"], ascending=[False, False], na_position="last").reset_index(
            drop=True
        )
