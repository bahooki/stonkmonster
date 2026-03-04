# Spread Strategy Research Mapping

This file documents how the spread backtest overlays are mapped to published research.

## 1) Model-vs-Model Relative-Strength Spread

### Implementation
- Build period returns for each model component from OOS model signals.
- Rank components on trailing risk-adjusted return (`mean / std`) over a rolling lookback.
- Go long strongest and short weakest component.
- Require a minimum edge to trade, and optionally apply switching cost when pair changes.

### Research links
- Gatev, Goetzmann, Rouwenhorst (2006), *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*  
  Source: [NBER Working Paper 7032](https://www.nber.org/papers/w7032)
- Hansen, Lunde, Nason (2011), *The Model Confidence Set*  
  Source: [Econometric Society publication page](https://econpapers.repec.org/article/ecmemetrp/v_3a79_3ay_3a2011_3ai_3a2_3ap_3a453-497.htm)

## 2) Pattern-vs-Pattern Relative-Strength Spread

### Implementation
- Aggregate OOS returns by candlestick pattern component.
- Rank patterns on rolling risk-adjusted performance.
- Long the strongest pattern and short the weakest pattern.

### Research links
- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling Losers*  
  Source: [Journal of Finance issue page](https://afajof.org/issue/volume-48-issue-1/)
- Gatev, Goetzmann, Rouwenhorst (2006), relative-value spread framework  
  Source: [NBER Working Paper 7032](https://www.nber.org/papers/w7032)

## 3) Beta/Style Neutral Overlay

### Implementation
- Compute rolling component beta to market proxy period returns.
- Re-weight long/short legs to target near-zero net market beta.
- Add a size-neutral blend using component market-cap z-exposures.

### Research links
- Frazzini and Pedersen (2014), *Betting Against Beta*  
  Source: [NBER Working Paper 16601](https://www.nber.org/papers/w16601)

## 4) Regime-Conditioned Spread Switching

### Implementation
- Build regime state (`risk_on`, `risk_off`, `neutral`) from macro/market stress and trend indicators.
- In risk-on regimes, prefer model-spread variant.
- In risk-off regimes, prefer pattern-spread variant.
- In neutral regimes, blend both.

### Research links
- Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*  
  Source: [Econometric Society publication page](https://econpapers.repec.org/article/ecmemetrp/v_3a57_3ay_3a1989_3ai_3a2_3ap_3a357-84.htm)
- Moreira and Muir (2017), *Volatility-Managed Portfolios*  
  Source: [NBER Working Paper 22208](https://www.nber.org/papers/w22208)

## 5) Optional Volatility Targeting

### Implementation
- Estimate trailing realized spread volatility.
- Scale next-period spread return toward a user-selected annualized vol target (with cap).

### Research links
- Moreira and Muir (2017), volatility management framework  
  Source: [NBER Working Paper 22208](https://www.nber.org/papers/w22208)
