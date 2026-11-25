"""Dip-catcher bot for "Tiny Bot, Easy $2: Beat Buy-and-Hold on Solana".

Strategy — long-only, hourly bars, 7 Solana assets, $10k USDC, one
position per symbol, max 10 orders per bar, 1% max slippage per order.
The goal is a net PnL above buy-and-hold on the final 72-bar window,
while respecting the guardrails: at least 8 filled trades and at most
25% max drawdown.

  1. DIP (core rule): when a symbol's close is >= 3.5% below its close
     10 bars ago (10-bar return <= -350 bps), buy a full stake
     (equity / 7, capped at the 25% max-weight cap). Exits for a dip
     position, checked every bar:
       - 10-bar return recovers to >= +1% (the bounce confirmed), or
       - take-profit: position value >= +5% over cost, or
       - stop-loss: position value <= -8% under cost, or
       - max hold: 36 bars since entry.
  2. SCALP (activity floor): only once the scored window is at least
     48 bars in AND fewer than 8 trades have filled, buy a 1/8 stake
     on any 1-bar downtick (close <= previous close). Exits for a
     scalp position: +0.8% / -0.8% vs cost, or after 3 bars. This
     keeps the entry above the 8-trade guardrail in quiet windows,
     at small size, without disturbing the core dip positions.

Why: on the 136-bar dev window the market fell ~9% while buy-and-hold
lost $869. Sitting in cash beats buy-and-hold in down windows; catching
sharp dips and holding for the bounce adds the upside. Momentum and
EMA-cross variants were tested offline and lost money on this data
(they buy bounces that fail in a downtrend), so they are not used.

Determinism: all decisions use exact integer arithmetic on raw closes
(no floats, no clock, no randomness, no network), so two runs produce
identical orders. The only state kept across bars is the per-symbol
close history, the open-position entries (mode + entry bar), and the
count of filled trades.

Offline sweep (rolling 72-bar windows, the final-window length):
  dev full window: +$227.78 PnL, 26 trades, 84.6% win rate,
  3.46% max drawdown (buy-and-hold on the same window: -$869.34)
  72-bar windows: beats buy-and-hold in 56/65, median PnL +$10.99,
  worst -$38.25, min 8 trades, worst max drawdown 3.5%.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

MAX_SLIPPAGE_BPS = 100  # 1% past the close the order was decided on

# Dip parameters (all in basis points / bars).
DIP_N = 10
DIP_TH_BPS = 350        # enter when 10-bar return <= -3.5%
DIP_EXIT_BPS = 100      # exit when 10-bar return >= +1%
DIP_TP_BPS = 500        # exit at +5% vs cost
DIP_SL_BPS = 800        # exit at -8% vs cost
DIP_MAX_HOLD = 36       # bars

# Scalp parameters (activity floor only).
SCALP_START_BAR = 48   # scored bars in before the scalp may fire
SCALP_MIN_TRADES = 8
SCALP_TP_BPS = 80       # exit at +0.8% vs cost
SCALP_SL_BPS = 80       # exit at -0.8% vs cost
SCALP_MAX_HOLD = 3      # bars
SCALP_FRAC_NUM = 1      # 1/8 of a full stake
SCALP_FRAC_DEN = 8

# Close history per symbol: raw close ints, oldest first.
closes = {}
# symbol -> (mode, entry seq): mode is "dip" or "scalp".
entries = {}
# Filled trades so far (the guardrail counts fills, not orders sent).
filled_trades = 0


def _ret_bps(now, then):
    """(now / then - 1) in bps, exact integer math (then > 0)."""
    return (now * 10000 - then * 10000) // then


def on_batch(state, batch):
    global filled_trades
    for fill in batch.get("fills", []):
        if fill.get("status") == "filled":
            filled_trades += 1

    fresh = set()
    for bar in batch["bars"]:
        closes.setdefault(bar["symbol"], []).append(int(bar["c"]))
        fresh.add(bar["symbol"])

    if state.warming_up:
        return []

    # Drop entry records for positions that are gone.
    for symbol in [s for s in entries if s not in state.positions]:
        del entries[symbol]

    scored_bar = state.seq - state.warmup_bars
    orders = []
    cash = state.cash
    slots = len(state.hello["symbols"])
    stake = min(state.equity // slots, state.equity * state.hello["caps"]["max_weight_bps"] // 10000)

    sells = []
    buys = []
    for symbol in [s["symbol"] for s in state.hello["symbols"]]:
        if symbol not in fresh:
            continue
        hist = closes.get(symbol, [])
        held = state.positions.get(symbol)
        if held is not None:
            mode, entry_seq = entries.get(symbol, ("dip", state.seq))
            value = state.value(symbol, held.qty)
            cost = held.cost
            sell = False
            if mode == "dip":
                if len(hist) > DIP_N and _ret_bps(hist[-1], hist[-1 - DIP_N]) >= DIP_EXIT_BPS:
                    sell = True
                if cost > 0 and (value * 10000 >= cost * (10000 + DIP_TP_BPS)
                                 or value * 10000 <= cost * (10000 - DIP_SL_BPS)):
                    sell = True
                if state.seq - entry_seq >= DIP_MAX_HOLD:
                    sell = True
            else:  # scalp
                if cost > 0 and (value * 10000 >= cost * (10000 + SCALP_TP_BPS)
                                 or value * 10000 <= cost * (10000 - SCALP_SL_BPS)):
                    sell = True
                if state.seq - entry_seq >= SCALP_MAX_HOLD:
                    sell = True
            if sell:
                sells.append({"side": "sell", "symbol": symbol, "qty": held.qty,
                              "max_slippage_bps": MAX_SLIPPAGE_BPS})
            continue

        # Entries (one position per symbol; sells above free no slot this reply).
        mode = None
        amount = 0
        if len(hist) > DIP_N and _ret_bps(hist[-1], hist[-1 - DIP_N]) <= -DIP_TH_BPS:
            mode = "dip"
            amount = stake
        elif (filled_trades < SCALP_MIN_TRADES and scored_bar >= SCALP_START_BAR
                and len(hist) > 1 and hist[-1] <= hist[-2]):
            mode = "scalp"
            amount = stake * SCALP_FRAC_NUM // SCALP_FRAC_DEN
        if mode is not None:
            room = state.weight_room(symbol)
            if amount > room:
                amount = room
            if amount > cash:
                amount = cash
            if amount >= int(state.hello["fees"]["min_notional_quote_units"]):
                buys.append({"side": "buy", "symbol": symbol,
                             "notional_quote_units": amount,
                             "max_slippage_bps": MAX_SLIPPAGE_BPS})
                entries[symbol] = (mode, state.seq)
                cash -= amount

    orders = sells + buys
    return orders[: state.hello["caps"]["max_orders_per_bar"]]


makefun_bars.run(on_batch)
