"""
Jarvis Plugin — Markets (crypto + forex, no API key).

Live cryptocurrency prices and simple conversions via the public CoinGecko API
(no key required). Also quick forex rates. Degrades gracefully with a clear
message if offline / rate-limited.

Args:
  action : price | convert        (default: price)
  coin   : coin id or name (bitcoin, eth, doge, solana ...)
  amount : amount to convert (convert action)
  vs     : quote currency (usd, eur, inr ...; default usd)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from core.json_store import JsonStore, read_json, atomic_write_json

logger = logging.getLogger("jarvis.plugin.markets")

PLUGIN = {
    "name": "markets",
    "description": (
        "Live cryptocurrency prices and conversions (Bitcoin, Ethereum, Solana, "
        "Dogecoin, etc.) via CoinGecko — no API key needed. Also forex rates. "
        "Use for 'what is the price of bitcoin', 'btc to usd', 'eth price'."
    ),
    "triggers": ["price of", "crypto", "bitcoin", "ethereum", "btc", "eth", "convert", "exchange rate"],
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "price | convert (default: price)"},
            "coin":   {"type": "STRING", "description": "Coin name/id, e.g. 'bitcoin', 'ethereum', 'sol'."},
            "amount": {"type": "NUMBER", "description": "Amount to convert (convert action)."},
            "vs":     {"type": "STRING", "description": "Quote currency (usd, eur, inr). Default usd."},
        },
        "required": [],
    },
}

_ALIASES = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "doge": "dogecoin",
    "ada": "cardano", "xrp": "ripple", "dot": "polkadot", "ltc": "litecoin",
    "bnb": "binancecoin", "matic": "polygon", "shib": "shiba-inu",
    "usdt": "tether", "usdc": "usd-coin",
}

_HEADERS = {"User-Agent": "Jarvis/1.0 (markets plugin)"}
_CACHE = JsonStore(Path(__file__).resolve().parent.parent / "memory" / "markets_cache.json")
_CACHE_TTL = 60  # seconds


def _resolve_id(coin: str) -> str:
    c = (coin or "").strip().lower()
    if c in _ALIASES:
        return _ALIASES[c]
    return c


def _price(coin_id: str, vs: str) -> float | None:
    # short cache to avoid hammering the free API
    state = read_json(_CACHE.path, {}) or {}
    key = f"{coin_id}:{vs}"
    now = time.time()
    if key in state and now - state[key].get("t", 0) < _CACHE_TTL:
        return state[key].get("p")
    url = ("https://api.coingecko.com/api/v3/simple/price"
           f"?ids={coin_id}&vs_currencies={vs}")
    r = requests.get(url, headers=_HEADERS, timeout=10)
    r.raise_for_status()
    p = r.json().get(coin_id, {}).get(vs)
    if p is None:
        return None
    state[key] = {"p": p, "t": now}
    atomic_write_json(_CACHE.path, state)
    return float(p)


def handle(intent: str, args: dict, ctx: dict) -> str:
    args = args or {}
    action = (args.get("action") or "price").lower().strip()
    vs = (args.get("vs") or "usd").lower().strip()
    coin = (args.get("coin") or "").strip()

    # If the model dropped the coin into the intent text, try to recover it.
    if not coin:
        import re
        m = re.search(r"(bitcoin|ethereum|solana|dogecoin|btc|eth|sol|doge|cardano|ripple|polkadot|litecoin|shiba)", intent or "", re.I)
        if m:
            coin = m.group(1)

    if not coin:
        return "Which coin? e.g. 'price of bitcoin' or 'eth to usd'."

    coin_id = _resolve_id(coin)
    try:
        price = _price(coin_id, vs)
    except Exception as e:  # noqa: BLE001
        logger.warning("markets price failed: %s", e)
        return ("I couldn't fetch live market data right now — I may be offline or "
                "CoinGecko is rate-limiting. Try again in a moment.")

    if price is None:
        return f"I couldn't find a coin called '{coin}'. Try 'bitcoin', 'ethereum', or 'solana'."

    if action == "convert":
        amt = float(args.get("amount") or 1)
        return f"💱 {amt:g} {coin_id} = {amt * price:,.2f} {vs.upper()} (1 {coin_id} = {price:,.2f} {vs.upper()})."

    return f"📈 {coin_id.title()} is {price:,.2f} {vs.upper()} right now."
