#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANTVEXA SIGNAL  -  Telegram Signal Bot (Virtual Candles, English UI)
=========================================================================
Signal lifecycle (per signal):
  1) Signal photo is sent WITHOUT a result (sent once, NEVER edited).
     The bottom banner shows the ENTRY TIME (next full minute).
  2) When the entry candle closes, a NEW photo is sent (as a reply to the
     signal) with the new candle and the result:
         WIN      - the entry candle went the signal's way
         MTG WIN  - the entry candle lost, the martingale candle won
         LOSS     - both candles lost
  3) The WINS / LOSSES counters shown on the images come from these real
     results (saved in .stats.json), not from random numbers.

Requirements:
    pip install python-telegram-bot==22.8 matplotlib numpy
Single file - nothing else to copy (optional: the 'ايموجي' emoji file next to it).

Commands:
  /start /stop /resume /signal /buy /sell /chart /status /resetstats
"""

import asyncio
import io
import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Circle, Polygon

from telegram import Update, MessageEntity, ReplyParameters
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import NetworkError, RetryAfter



# ============================================
# General Configuration
# ============================================
BOT_VERSION = "4.4.1"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_ENV_CHAT_ID = os.environ.get("AUTO_SIGNAL_CHAT_ID", "") or os.environ.get("TELEGRAM_CHAT_ID", "")
AUTO_SIGNAL_CHAT_ID = ""      # set in main(): override / typed at startup

# >>> PUT YOUR OWN chat_id HERE (a number, e.g. "123456789"; a channel/group
# looks like "-1001234567890"). When set, it OVERRIDES the environment
# variables, the .chatid_cache file and /start.  Leave "" to use /start.
CHAT_ID_OVERRIDE = ""
ALWAYS_ASK_CHAT_ID = True        # ask for the chat_id at EVERY start; old saved/env ids are wiped
ALLOW_START_TO_REGISTER = False  # True = any /start changes the target (not recommended)
REGISTRATION_OPEN = False        # internal: opens when you press Enter at the chat_id prompt

TOKEN_CACHE_FILE = Path(__file__).parent / ".token_cache"
CHATID_CACHE_FILE = Path(__file__).parent / ".chatid_cache"
STATS_FILE = Path(__file__).parent / ".stats.json"

AUTO_SIGNAL_ENABLED = True
AUTO_SIGNAL_GAP_AFTER_RESULT = 10  # seconds to wait after a result, then send the next signal
AUTO_SIGNAL_START_DELAY = 3       # seconds to wait after polling starts

CHART_CANDLES_COUNT = 30
CANDLE_SECONDS = 60               # M1 candle length (tests may lower this)
ENTRY_MIN_LEAD_SECONDS = 5        # entry = the NEXT full minute; if less than this is left, use the one after

EMOJI_FILE_PATH = Path(__file__).parent / "ايموجي"
SECTION_SEP = "━━━━━━━━━━━━━━━━"

SUPPORTED_SYMBOLS = [
    ("USDJPY_otc", 158.000),
    ("EURUSD_otc", 1.0850),
    ("GBPUSD_otc", 1.2650),
    ("AUDUSD_otc", 0.6550),
    ("USDCHF_otc", 0.9050),
    ("USDCAD_otc", 1.3650),
    ("NZDUSD_otc", 0.6050),
    ("USDZAR_otc", 18.500),
    ("USDTRY_otc", 32.200),
    ("USDMXN_otc", 17.300),
    ("USDBRL_otc", 5.150),
    ("USDSAR_otc", 3.7500),
    ("UKBRENT_otc", 51.200),
    ("USDARS_otc", 920.000),
]

CHAT_FORBIDDEN = False            # set when Telegram rejects the target chat
STATS = {"wins": 0, "losses": 0}  # real results (direct WIN + MTG WIN = win)
LIFECYCLE_TASKS: set = set()
LAST_LIFECYCLE_TASK: Optional[asyncio.Task] = None
_RENDER_LOCK: Optional[asyncio.Lock] = None


# ============================================
# Stats persistence
# ============================================
def load_stats() -> None:
    try:
        data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        STATS["wins"] = int(data.get("wins", 0))
        STATS["losses"] = int(data.get("losses", 0))
    except Exception:
        pass


def save_stats() -> None:
    try:
        STATS_FILE.write_text(json.dumps(STATS), encoding="utf-8")
    except Exception as e:
        print(f"WARNING: failed to save stats: {e}")


load_stats()


# ============================================
# Special animated-emoji loader
# ============================================
def _find_emoji_file() -> Optional[Path]:
    """Look for the emoji file next to the script (name 'ايموجي', with or
    without .json/.txt - Windows often hides the extension), or via the
    EMOJI_FILE environment variable."""
    folder = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("EMOJI_FILE")
    if env:
        candidates.append(Path(env))
    candidates.append(EMOJI_FILE_PATH)
    candidates += sorted(folder.glob("ايموجي*"))
    candidates += [folder / "emoji.json", folder / "emoji.txt", folder / "emoji"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_special_emoji_map() -> dict:
    path = _find_emoji_file()
    if path is None:
        print(f"WARNING: special-emoji file not found in: {Path(__file__).resolve().parent}")
        print("         Put a file named 'ايموجي' (or ايموجي.json / ايموجي.txt) next to this script.")
        print("         Regular (non-animated) emoji will be used meanwhile - the bot works normally.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        emoji_list = data.get("RestrictedEmoji", {}).get("emoji", [])
        result = {item["emoji"]: item["custom_emoji_id"] for item in emoji_list}
        print(f"OK - loaded {len(result)} animated emoji from: {path.name}")
        return result
    except Exception as e:
        print(f"WARNING: failed to read special-emoji file ({path}): {e}")
        print("WARNING: regular (non-animated) emoji will be used as fallback.")
        return {}


SPECIAL_EMOJI_MAP = load_special_emoji_map()


def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def build_custom_emoji_entities(text: str, emoji_map: dict) -> List[MessageEntity]:
    """Build CUSTOM_EMOJI entities (greedy longest match, multi-codepoint safe)."""
    if not emoji_map:
        return []

    by_first_char: dict = {}
    for emoji_str in emoji_map:
        if emoji_str:
            by_first_char.setdefault(emoji_str[0], []).append(emoji_str)
    for k in by_first_char:
        by_first_char[k].sort(key=len, reverse=True)

    entities: List[MessageEntity] = []
    pos = 0
    offset_utf16 = 0
    while pos < len(text):
        char = text[pos]
        matched = None
        if char in by_first_char:
            for emoji_str in by_first_char[char]:
                if text.startswith(emoji_str, pos):
                    matched = emoji_str
                    break
        if matched:
            entities.append(MessageEntity(
                type=MessageEntity.CUSTOM_EMOJI,
                offset=offset_utf16,
                length=utf16_len(matched),
                custom_emoji_id=emoji_map[matched],
            ))
            pos += len(matched)
            offset_utf16 += utf16_len(matched)
        else:
            pos += 1
            offset_utf16 += utf16_len(char)
    return entities


# ============================================
# Virtual candles
# ============================================
def generate_virtual_candles(
    count: int = CHART_CANDLES_COUNT,
    start_price: float = 1.1000,
    seed: Optional[int] = None,
) -> List[dict]:
    """Regime-switching virtual candles. Each: time/open/high/low/close/volume."""
    rng = np.random.default_rng(seed)
    candles: List[dict] = []
    start_time = int((datetime.now() - timedelta(minutes=count + 1)).timestamp())

    price = float(start_price)
    scale = max(start_price / 1.1, 1.0) ** 0.5
    base_volatility = rng.uniform(0.0004, 0.0010) * scale

    seg1_end = int(count * 0.33)
    seg2_end = int(count * 0.66)
    regime_trends = [rng.choice([-1, 0, 1]) for _ in range(3)]
    regime_strengths = [rng.uniform(0.00015, 0.0005) * scale for _ in range(3)]

    for i in range(count):
        seg = 0 if i < seg1_end else (1 if i < seg2_end else 2)
        trend_dir, trend_strength = regime_trends[seg], regime_strengths[seg]
        if rng.random() < 0.08:
            trend_dir = -trend_dir if trend_dir != 0 else rng.choice([-1, 1])

        open_price = price
        close_price = open_price + trend_dir * trend_strength + rng.normal(0, base_volatility)

        body_max = max(open_price, close_price)
        body_min = min(open_price, close_price)
        wick_range = base_volatility * rng.uniform(0.5, 1.4)
        high_price = body_max + rng.uniform(0.00005 * scale, wick_range)
        low_price = body_min - rng.uniform(0.00005 * scale, wick_range)

        candles.append({
            "time": start_time + (i + 1) * 60,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": int(rng.uniform(100, 1500)),
        })
        price = close_price
    return candles


def generate_next_candles(base_candles: List[dict], k: int, first_time: datetime) -> List[dict]:
    """
    Continue the virtual series by k candles. Neutral random walk: the next
    candle has no knowledge of (and no bias toward) the signal direction,
    so results are decided purely by the simulated candle.
    """
    rng = np.random.default_rng()
    closes = np.array([c["close"] for c in base_candles])
    sigma = float(np.std(np.diff(closes))) or abs(closes[-1]) * 0.0005
    out: List[dict] = []
    price = base_candles[-1]["close"]
    for i in range(k):
        o = price
        c = o + rng.normal(0, sigma)
        wick = sigma * rng.uniform(0.2, 0.9)
        out.append({
            "time": int((first_time + timedelta(seconds=CANDLE_SECONDS * i)).timestamp()),
            "open": float(o),
            "high": float(max(o, c) + rng.uniform(0.05, 1.0) * wick),
            "low": float(min(o, c) - rng.uniform(0.05, 1.0) * wick),
            "close": float(c),
            "volume": int(rng.uniform(100, 1500)),
        })
        price = c
    return out


def candle_wins(direction: str, candle: dict) -> bool:
    if direction == "BUY":
        return candle["close"] > candle["open"]
    return candle["close"] < candle["open"]


def decide_signal_direction(candles: List[dict]) -> str:
    if len(candles) < 2:
        return random.choice(["BUY", "SELL"])
    return "BUY" if candles[-1]["close"] > candles[-2]["close"] else "SELL"


# ============================================
# Signal object / captions
# ============================================
def _fmt_price(p: float) -> str:
    if p >= 10:
        return f"{p:.3f}"
    return f"{p:.4f}" if p >= 1 else f"{p:.5f}"


def build_signal(direction: Optional[str] = None) -> dict:
    """Create a new signal: virtual candles aligned to the entry minute."""
    symbol, base_price = random.choice(SUPPORTED_SYMBOLS)
    candles = generate_virtual_candles(CHART_CANDLES_COUNT, base_price,
                                       seed=random.randint(0, 1_000_000))

    now = datetime.now()
    # Entry = the next full minute (sent at 02:53:xx -> entry 02:54:00).
    entry_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    if (entry_time - now).total_seconds() < ENTRY_MIN_LEAD_SECONDS:
        entry_time += timedelta(minutes=1)

    # Re-label candle times so the last base candle is the minute before entry.
    n = len(candles)
    for i, c in enumerate(candles):
        c["time"] = int((entry_time - timedelta(seconds=CANDLE_SECONDS * (n - i))).timestamp())

    if direction is None:
        direction = decide_signal_direction(candles)

    return {
        "symbol": symbol,
        "candles": candles,
        "direction": direction,
        "entry_time": entry_time,
        "strength": random.randint(85, 99),
        "payout": random.randint(80, 92),
        "support": min(c["low"] for c in candles),
        "resistance": max(c["high"] for c in candles),
    }


# ---------------------------------------------------------------------------
# Premium (animated) emoji used in the captions.  name -> (fallback char, id)
# The fallback char is what non-premium clients / clients without support see.
# ---------------------------------------------------------------------------
CE = {
    "bolt":        ("\u26A1\uFE0F",  "5220195537520711716"),   # ⚡️ lightning
    "fire":        ("\U0001F525",    "5220166546491459639"),   # 🔥 glowing fire
    "dollar":      ("\U0001F4B5",    "5197434882321567830"),   # 💵 currency
    "dollar_neon": ("\U0001F4B2",    "5373350287429872269"),   # 💲 currency (neon)
    "soon":        ("\U0001F51C",    "5440621591387980068"),   # 🔜 clock / entry
    "hourglass":   ("\u231B",        "5386367538735104399"),   # ⌛ loading circle
    "timer":       ("\u23F1\uFE0F",  "5382194935057372936"),   # ⏱ timer / trade duration
    "up":          ("\U0001F53C",    "5449683594425410231"),   # 🔼 rising
    "down":        ("\U0001F53D",    "5447183459602669338"),   # 🔽 falling
    "candle":      ("\U0001F56F\uFE0F", "5451882707875276247"),  # 🕯 candles
    "chart_up":    ("\U0001F4C8",    "5244837092042750681"),   # 📈 rising chart
    "chart_down":  ("\U0001F4C9",    "5246762912428603768"),   # 📉 falling chart
    "crown":       ("\U0001F451",    "5217822164362739968"),   # 👑 crown
    "wink":        ("\U0001F609",    "5339267587337370029"),   # 😉 robot smile
    "devil":       ("\U0001F608",    "5197645099495862838"),   # 😈 robot devil
    "salute":      ("\U0001FAE1",    "5323772371830588991"),   # 🫡 robot salute
    "money_face":  ("\U0001F911",    "5436386989857320953"),   # 🤑 profit
    "thumbs_up":   ("\U0001F44D",    "5323547156630483403"),   # 👍
    "thumbs_down": ("\U0001F44E",    "5197396124536682206"),   # 👎
    "mind_blown":  ("\U0001F92F",    "5197564405650307134"),   # 🤯
    "check":       ("\u2705",        "5438176453621457379"),   # ✅ check mark
    "cross":       ("\u274C",        "5438630285635757876"),   # ❌ wrong mark
    "check2":      ("\u2714\uFE0F",  "5206607081334906820"),   # ✔️ another check
    "cross2":      ("\u274C",        "5210952531676504517"),   # ❌ another cross
}

# ---------------------------------------------------------------------------
# "BloodyFontEmoji" pack: 26 custom emoji in order = the letters A..Z.
# Used to write the currency-pair name in the captions.  (fallback char, id)
# ---------------------------------------------------------------------------
LETTER_EMOJI = [
    ("\U0001F170\uFE0F", "5293991227513914037"),  # A
    ("\U0001F171\uFE0F", "5294446571356697709"),  # B
    ("\U0001FA78", "5323692545568424149"),         # C
    ("\U0001FA78", "5294029641701407930"),         # D
    ("\U0001FA78", "5327938799345349736"),         # E
    ("\U0001FA78", "5310097750010901912"),         # F
    ("\U0001FA78", "5298628306134909270"),         # G
    ("\U0001FA78", "5325878958799994800"),         # H
    ("\U0001FA78", "5330144144792763349"),         # I
    ("\U0001FA78", "5294448748905119870"),         # J
    ("\U0001FA78", "5312383462886356958"),         # K
    ("\U0001FA78", "5463362571341942623"),         # L
    ("\U0001FA78", "5330094426251344221"),         # M
    ("\U0001FA78", "5307644490461231051"),         # N
    ("\U0001FA78", "5327844718086733432"),         # O
    ("\U0001FA78", "5314463451123300950"),         # P
    ("\U0001FA78", "5332815043220225405"),         # Q
    ("\U0001FA78", "5330292450013494017"),         # R
    ("\U0001FA78", "5321450791683241246"),         # S
    ("\U0001FAF5", "5330388403877853223"),         # T
    ("\U0001FA78", "5328162635860948105"),         # U
    ("\U0001FA78", "5332614094585345389"),         # V
    ("\U0001FA78", "5332470243245702221"),         # W
    ("\U0001FA78", "5334637865995352639"),         # X
    ("\u2753",     "5298741607372178279"),         # Y
    ("\U0001FA78", "5334671517064117716"),         # Z
]
for _i, _entry in enumerate(LETTER_EMOJI):
    CE[f"L_{chr(65 + _i)}"] = _entry


def symbol_parts(symbol: str, emoji_symbol: bool = True) -> list:
    """
    Pair name for the captions: the currency letters as custom emoji, the '/'
    plain, and 'OTC' in the normal text font.   EURUSD_otc -> E U R / U S D  OTC
    With emoji_symbol=False (fallback when emoji are refused) letters stay text.
    """
    base = symbol.replace("_otc", "").upper()
    if len(base) == 6 and base.isalpha():
        base = f"{base[:3]}/{base[3:]}"
    parts: list = []
    for ch in base:
        if emoji_symbol and "A" <= ch <= "Z":
            parts.append(("ce", f"L_{ch}"))
        else:
            parts.append(fx(ch))
    parts.append(f" {fx('OTC')}")
    return parts


# Bot name is placed between these two emoji (names from CE above).
HEADER_EMOJI_LEFT = "crown"
HEADER_EMOJI_RIGHT = "crown"

CAPTION_TIMER_LINE = False    # True = add a "Starts in mm:ss" line under the signal
BRAND_FANCY = "\U0001D444\U0001D448\U0001D434\U0001D441\U0001D447\U0001D449\U0001D438\U0001D44B\U0001D434"   # 𝑄𝑈𝐴𝑁𝑇𝑉𝐸𝑋𝐴
QUOTEX_FANCY = "\U0001D444\U0001D448\U0001D442\U0001D447\U0001D438\U0001D44B"                                   # 𝑄𝑈𝑂𝑇𝐸𝑋
LINE_SEP = "\u2501" * 14      # ━━━━━━━━━━━━━━


def fx(text: str) -> str:
    """Distinctive font: Unicode 'sans-serif bold' (A-Z a-z 0-9), e.g. Entry -> 𝗘𝗻𝘁𝗿𝘆."""
    out = []
    for ch in text:
        o = ord(ch)
        if "A" <= ch <= "Z":
            out.append(chr(0x1D5D4 + o - 65))
        elif "a" <= ch <= "z":
            out.append(chr(0x1D5EE + o - 97))
        elif "0" <= ch <= "9":
            out.append(chr(0x1D7EC + o - 48))
        else:
            out.append(ch)
    return "".join(out)


def pretty_symbol(symbol: str) -> str:
    """EURUSD_otc -> 'EUR/USD OTC' ; UKBRENT_otc -> 'UKBRENT OTC'."""
    base = symbol.replace("_otc", "").upper()
    if len(base) == 6 and base.isalpha():
        base = f"{base[:3]}/{base[3:]}"
    return f"{base} OTC"


def _compose(parts) -> tuple:
    """
    parts: list of  str | ("ce", name) | ("code", text).
    Returns (text, entities) with correct UTF-16 offsets. Emoji of the old
    emoji file are still applied to the remaining plain text.
    """
    text = ""
    ents: List[MessageEntity] = []
    off = 0
    for p in parts:
        if isinstance(p, str):
            text += p
            off += utf16_len(p)
        elif p[0] == "ce":
            ch, cid = CE[p[1]]
            ents.append(MessageEntity(type=MessageEntity.CUSTOM_EMOJI, offset=off,
                                      length=utf16_len(ch), custom_emoji_id=cid))
            text += ch
            off += utf16_len(ch)
        elif p[0] == "code":
            ents.append(MessageEntity(type=MessageEntity.CODE, offset=off,
                                      length=utf16_len(p[1])))
            text += p[1]
            off += utf16_len(p[1])
    for e in build_custom_emoji_entities(text, SPECIAL_EMOJI_MAP):
        if not any(e.offset < x.offset + x.length and x.offset < e.offset + e.length for x in ents):
            ents.append(e)
    ents.sort(key=lambda e: e.offset)
    return text, ents


def _header_parts() -> list:
    return [("ce", HEADER_EMOJI_LEFT), f" {BRAND_FANCY} ", ("ce", HEADER_EMOJI_RIGHT), "\n",
            f"{LINE_SEP}\n"]


def build_signal_caption(sig: dict, emoji_symbol: bool = True) -> tuple:
    """Signal caption (no result yet) -> (text, entities)."""
    is_buy = sig["direction"] == "BUY"
    dir_icon = ("ce", "up") if is_buy else ("ce", "down")
    dir_text = (f" {fx('CALL')} \u2191 \u00B7 {fx('Bullish')}\n" if is_buy
                else f" {fx('PUT')} \u2193 \u00B7 {fx('Bearish')}\n")

    parts = _header_parts() + [
        ("ce", "dollar_neon"), " ", *symbol_parts(sig["symbol"], emoji_symbol), "\n",
        dir_icon, dir_text,
        ("ce", "soon"), f" {fx('Entry')} ", ("code", sig["entry_time"].strftime("%H:%M")), "\n",
        ("ce", "timer"), f" {fx('Period')} ", ("code", "M1"), "\n",
        ("ce", "fire"), f" {fx('Strength')} {fx(str(sig['strength']) + '%')}\n",
        ("ce", "money_face"), f" {fx('Payout')} {fx(str(sig['payout']) + '%')}\n",
        f"{LINE_SEP}\n",
    ]
    if CAPTION_TIMER_LINE:
        rem = max(0, int((sig["entry_time"] - datetime.now()).total_seconds()))
        label = f"Starts in {rem // 60:02d}:{rem % 60:02d}" if rem > 0 else "Go now"
        parts += [("ce", "hourglass"), f" {fx(label)}\n"]
    parts += [f"\U0001F3E6 {QUOTEX_FANCY}"]
    return _compose(parts)


def build_result_caption(sig: dict, status: str, last_candle: dict,
                         emoji_symbol: bool = True) -> tuple:
    """Result caption -> (text, entities)."""
    is_buy = sig["direction"] == "BUY"
    dir_icon = ("ce", "up") if is_buy else ("ce", "down")
    dir_text = f" {fx('CALL')} \u2191\n" if is_buy else f" {fx('PUT')} \u2193\n"

    won = status in ("WIN", "MTG_WIN")
    label = {"WIN": "WIN", "MTG_WIN": "MTG WIN", "LOSS": "LOSS"}[status]
    payout_txt = f"+{sig['payout']}%" if won else "-100%"
    # the signal ends when its last candle closes: entry + 1 candle (WIN) or + 2 (MTG WIN / LOSS)
    ended = sig["entry_time"] + timedelta(seconds=CANDLE_SECONDS * (1 if status == "WIN" else 2))

    parts = _header_parts() + [
        ("ce", "dollar_neon"), " ", *symbol_parts(sig["symbol"], emoji_symbol), "\n",
        dir_icon, dir_text,
        ("ce", "soon"), f" {fx('Entry')} ", ("code", sig["entry_time"].strftime("%H:%M")), "\n",
        ("ce", "timer"), f" {fx('Period')} ", ("code", "M1"), "\n",
        ("ce", "hourglass"), f" {fx('Ended')} ", ("code", ended.strftime("%H:%M")), "\n",
        f"{LINE_SEP}\n",
        ("ce", "thumbs_up") if won else ("ce", "thumbs_down"), f" {fx('Result')} {fx(label)} ",
        ("ce", "check2") if won else ("ce", "cross2"), "\n",
        ("ce", "money_face") if won else ("ce", "dollar"), f" {fx('Payout')} {fx(payout_txt)}\n",
        f"{LINE_SEP}\n",
        f"\U0001F3E6 {QUOTEX_FANCY}",
    ]
    return _compose(parts)


def build_status_message() -> str:
    total = STATS["wins"] + STATS["losses"]
    wr = f"{STATS['wins'] / total * 100:.0f}%" if total else "--"
    return (
        "\U0001F4CA Bot Status\n"
        f"{SECTION_SEP}\n"
        "\U0001F49A Status   : Online\n"
        f"\U0001F381 Version  : {BOT_VERSION}\n"
        f"\U0001F56F Candles   : {CHART_CANDLES_COUNT} virtual per chart\n"
        f"\U0001F504 Auto     : 1 signal at a time, next {AUTO_SIGNAL_GAP_AFTER_RESULT}s after each result\n"
        f"\U0001F4CE Target   : {AUTO_SIGNAL_CHAT_ID or 'unset'}\n"
        f"\U0001F3C1 Results  : {STATS['wins']}W / {STATS['losses']}L ({wr})\n"
        f"\U0001F3A8 Animated Emoji: {'Loaded \u2705' if SPECIAL_EMOJI_MAP else 'Unavailable \u274C'}\n"
        "\U0001F916 Library  : python-telegram-bot v22.8\n"
    )


def build_welcome_message() -> str:
    return (
        "\U0001F44B Welcome to QUANTVEXA\n"
        "\n"
        "\U0001F916 A Telegram bot that auto-sends trading signals with a professional "
        "chart image (virtual candles, no exchange connection).\n"
        "\n"
        "\U0001F504 Each signal is sent with an entry countdown; after the entry candle "
        "closes the chart is sent again with the result (WIN / MTG WIN / LOSS).\n"
        "\n"
        "\U0001F4DD Commands:\n"
        "  /start       - Register your chat_id + start AUTO-SIGNAL\n"
        "  /stop        - Stop the AUTO-SIGNAL loop\n"
        "  /resume      - Resume the AUTO-SIGNAL loop\n"
        "  /signal      - Send one random signal now\n"
        "  /buy         - Send a BUY (CALL) signal now\n"
        "  /sell        - Send a SELL (PUT) signal now\n"
        "  /chart       - Chart only, no signal\n"
        "  /status      - Show bot status\n"
        "  /resetstats  - Reset WINS / LOSSES counters\n"
    )


# ============================================
# Chart renderer - neon style (1600x900)
# Black + electric blue + red + green, glowing candles, cut-corner panels.
# No eye, no sigil rings, no hazard stripes.
# ============================================

# ---------------- palette ----------------
BG     = "#000000"
PANEL  = "#04060C"
TILE   = "#080C18"
EDGE   = "#14265A"
BLUE   = "#1F6BFF"
SKY    = "#4CC9FF"
RED    = "#FF1030"
GREEN  = "#00FF66"
BONE   = "#E8EEFF"
MUTED  = "#8C98B3"
DIM    = "#46526E"
GRIDC  = "#0B1222"
TRACK  = "#0F1830"

W, H = 1600, 900
MONO = "DejaVu Sans Mono"
BRAND = "QUANTVEXA"

RESULTS = {
    "WIN":     ("WIN",     GREEN, "WON ON THE ENTRY CANDLE",      "\u2714 WIN"),
    "MTG_WIN": ("MTG WIN", GREEN, "WON ON MARTINGALE STEP 1",     "\u2714 MTG WIN"),
    "LOSS":    ("LOSS",    RED,   "LOST AFTER MARTINGALE STEP 1", "\u2716 LOSS"),
}


# ---------------- helpers ----------------
def _pts(x, y, w, h, c):
    """Rectangle with two cut corners (top-left, bottom-right)."""
    return [(x + c, y), (x + w, y), (x + w, y + h - c),
            (x + w - c, y + h), (x, y + h), (x, y + c)]


def _panel(ax, x, y, w, h, c=16, fc=PANEL, ec=EDGE, lw=1.5, cut=RED, z=1, glow=False):
    pts = _pts(x, y, w, h, c)
    if glow:
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=ec,
                             linewidth=lw * 4, alpha=0.10, zorder=z))
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=z, joinstyle="miter"))
    if cut and c >= 8:  # the two cut edges burn red
        ax.plot([x, x + c], [y + c, y], color=cut, lw=lw + 1.2, zorder=z + 0.1,
                solid_capstyle="butt")
        ax.plot([x + w - c, x + w], [y + h, y + h - c], color=cut, lw=lw + 1.2,
                zorder=z + 0.1, solid_capstyle="butt")


def _t(ax, x, y, s, size=11, color=BONE, ha="left", va="center",
       weight="bold", family=None, z=6, glow=None, **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                fontweight=weight, family=family, zorder=z, **kw)
    if glow:
        t.set_path_effects([pe.Stroke(linewidth=5, foreground=glow, alpha=0.20), pe.Normal()])
    return t


def _fmt(v: float) -> str:
    if v >= 10:
        return f"{v:.3f}"
    return f"{v:.4f}" if v >= 1 else f"{v:.5f}"


def _lerp(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


def render_signal_chart(
    candles: List[dict],
    symbol: str,
    direction: str,
    payout: Optional[int] = None,
    entry_time: Optional[datetime] = None,
    status: str = "PENDING",
    n_base: Optional[int] = None,
    stats: Optional[Tuple[int, int]] = None,
    confidence: Optional[int] = None,
    now: Optional[datetime] = None,
) -> bytes:
    now = now or datetime.now()
    if entry_time is None:
        entry_time = now + timedelta(minutes=1)

    is_buy = direction == "BUY"
    dcol = GREEN if is_buy else RED
    dtxt = "CALL" if is_buy else "PUT"
    dsub = "BUY  \u00B7  M1  \u00B7  OTC" if is_buy else "SELL  \u00B7  M1  \u00B7  OTC"

    is_result = status in RESULTS
    if is_result:
        r_title, rcol, r_sub, r_pill = RESULTS[status]

    n = len(candles)
    nb = n_base if n_base else n
    o = np.array([c["open"] for c in candles])
    h = np.array([c["high"] for c in candles])
    l = np.array([c["low"] for c in candles])
    c_ = np.array([c["close"] for c in candles])
    vol = np.array([c["volume"] for c in candles], dtype=float)
    times = [datetime.fromtimestamp(c["time"]) for c in candles]

    def sma(a, w):
        return np.array([a[max(0, i - w + 1): i + 1].mean() for i in range(len(a))])

    ma7, ma14, ma25 = sma(c_, 7), sma(c_, 14), sma(c_, 25)

    support, resistance = l[:nb].min(), h[:nb].max()
    rng = max(resistance - support, 1e-9)
    data_lo, data_hi = min(l.min(), support), max(h.max(), resistance)
    span = max(data_hi - data_lo, 1e-9)
    ymin, ymax = data_lo - span * 0.25, data_hi + span * 0.25
    cur = c_[-1]
    entry_price = c_[nb - 1]

    sym = symbol.replace("_otc", "-OTC").upper()
    pay = payout if payout is not None else 85
    conf = confidence if confidence is not None else random.randint(90, 99)
    wins, losses = stats if stats else (0, 0)
    total = wins + losses
    wr = (wins / total * 100) if total else None

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_xlim(0, W)
    bg.set_ylim(H, 0)
    bg.axis("off")

    # ================= CHART PANEL =================
    _panel(bg, 20, 20, 1130, 860, c=24, fc=PANEL, ec=BLUE, lw=1.8, glow=True)

    # header: brand | symbol | chips | status | payout   (no eye)
    _t(bg, 44, 52, BRAND, 16, BONE)
    bg.plot([206, 206], [34, 70], color=EDGE, lw=1.4, zorder=4)
    _t(bg, 228, 52, sym, 22, BONE)
    _panel(bg, 528, 37, 52, 30, c=8, fc=TILE, ec=BLUE, lw=1.3, cut=None)
    _t(bg, 554, 52, "M1", 10.5, BLUE, ha="center")
    _panel(bg, 590, 37, 58, 30, c=8, fc=TILE, ec=SKY, lw=1.3, cut=None)
    _t(bg, 619, 52, "OTC", 10.5, SKY, ha="center")

    live_col = rcol if is_result else GREEN
    live_txt = "RESULT" if is_result else "LIVE"
    bg.add_patch(Circle((968, 52), 6, facecolor=live_col, edgecolor="none", zorder=5))
    bg.add_patch(Circle((968, 52), 11, facecolor=live_col, edgecolor="none", zorder=4, alpha=0.2))
    _t(bg, 984, 52, live_txt, 10, live_col)
    _panel(bg, 1066, 37, 68, 30, c=8, fc="#03140A", ec=GREEN, lw=1.3, cut=None)
    _t(bg, 1100, 52, f"{pay}%", 11, GREEN, ha="center")

    lx = 44
    for lab, col in [("MA 7", BLUE), ("MA 14", SKY), ("MA 25", BONE)]:
        bg.plot([lx, lx + 22], [88, 88], color=col, lw=2.6, zorder=6, solid_capstyle="round")
        _t(bg, lx + 30, 88, lab, 9, MUTED, family=MONO)
        lx += 100

    AX_X, AX_Y, AX_W, AX_H = 40, 106, 1010, 660
    VX_Y, VX_H = 782, 62

    def rect(x, y, w, hh):
        return [x / W, 1 - (y + hh) / H, w / W, hh / H]

    pad = 3.6
    x0, x1 = -0.9, n - 1 + pad
    ax = fig.add_axes(rect(AX_X, AX_Y, AX_W, AX_H), zorder=2)
    ax.set_facecolor("none")
    ax.set_xlim(x0, x1)
    ax.set_ylim(ymin, ymax)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def py(v):
        return AX_Y + (ymax - v) / (ymax - ymin) * AX_H

    def px(xv):
        return AX_X + (xv - x0) / (x1 - x0) * AX_W

    for t in np.linspace(ymin, ymax, 8):
        ax.axhline(t, color=GRIDC, lw=1.0, zorder=1)
        if abs(t - cur) > (ymax - ymin) * 0.05:
            bg.text(AX_X + AX_W + 12, py(t), _fmt(t), color=DIM, fontsize=8.5,
                    ha="left", va="center", family=MONO, zorder=5)
    for gx in range(0, n, 3):
        ax.axvline(gx, color=GRIDC, lw=0.8, zorder=1)

    # S / R
    ax.axhline(resistance, color=RED, lw=1.0, ls=(0, (5, 4)), alpha=0.95, zorder=3)
    ax.axhline(support, color=GREEN, lw=1.0, ls=(0, (5, 4)), alpha=0.95, zorder=3)
    for val, col, tag, up in [(resistance, RED, "R", True), (support, GREEN, "S", False)]:
        yy = py(val)
        off = -26 if up else 8
        _panel(bg, AX_X + 6, yy + off, 116, 20, c=6, fc=BG, ec=col, lw=1.0, cut=None, z=7)
        _t(bg, AX_X + 64, yy + off + 10, f"{tag}  {_fmt(val)}", 8.5, col,
           ha="center", family=MONO, z=8)

    # bands + entry line
    ax.axvspan(nb - 1 - 0.8, nb - 1 + 0.8, color=dcol, alpha=0.10, zorder=2, lw=0)
    ax.axvline(nb - 0.5, color=SKY, lw=1.3, ls=(0, (3, 3)), alpha=0.95, zorder=4)
    _panel(bg, px(nb - 0.5) + 6, AX_Y + 6, 66, 20, c=6, fc=BG, ec=SKY, lw=1.0, cut=None, z=7)
    _t(bg, px(nb - 0.5) + 39, AX_Y + 16, "ENTRY", 8.5, SKY, ha="center", z=8)
    if is_result:
        ax.axvspan(nb - 0.5, n - 0.5, color=rcol, alpha=0.09, zorder=2, lw=0)
        ax.hlines(entry_price, nb - 0.5, x1, color=SKY, lw=1.0, ls=(0, (4, 3)),
                  alpha=0.85, zorder=4)
    else:
        ax.axvspan(nb - 0.5, x1, color=dcol, alpha=0.045, zorder=2, lw=0)

    # MAs (neon glow)
    xs = np.arange(n)
    for series, col in [(ma25, BONE), (ma14, SKY), (ma7, BLUE)]:
        ax.plot(xs, series, color=col, lw=1.8, zorder=4,
                path_effects=[pe.Stroke(linewidth=6, foreground=col, alpha=0.16), pe.Normal()])

    # candles
    bw = 0.60
    for i in range(n):
        col = GREEN if c_[i] >= o[i] else RED
        ax.plot([i, i], [l[i], h[i]], color=col, lw=1.6, zorder=5, solid_capstyle="butt",
                path_effects=[pe.Stroke(linewidth=5, foreground=col, alpha=0.14), pe.Normal()])
        lo, hi = min(o[i], c_[i]), max(o[i], c_[i])
        ax.add_patch(Rectangle((i - bw / 2, lo), bw, max(hi - lo, span * 0.004),
                               facecolor=col, edgecolor=col, lw=0.5, zorder=6))

    # current price line + flag
    ax.axhline(cur, color=BLUE, lw=0.9, ls=(0, (2, 3)), alpha=0.95, zorder=3)
    yy = py(cur)
    bg.add_patch(Polygon(_pts(AX_X + AX_W + 6, yy - 14, 82, 28, 8), closed=True,
                         facecolor=BLUE, edgecolor="none", zorder=8))
    _t(bg, AX_X + AX_W + 47, yy, _fmt(cur), 9.5, "#FFFFFF", ha="center", family=MONO, z=9)

    # direction marker + badge
    s_i = nb - 1
    if is_buy:
        tip, bpos, mk, va = l[s_i] - span * 0.06, l[s_i] - span * 0.15, "^", "top"
    else:
        tip, bpos, mk, va = h[s_i] + span * 0.06, h[s_i] + span * 0.15, "v", "bottom"
    ax.plot([s_i], [tip], marker=mk, markersize=14, color=dcol, zorder=9, clip_on=False,
            path_effects=[pe.Stroke(linewidth=10, foreground=dcol, alpha=0.28), pe.Normal()])
    ax.annotate(dtxt, xy=(s_i, bpos), ha="center", va=va, color="#000000",
                fontsize=10, fontweight="bold", zorder=10, annotation_clip=False,
                bbox=dict(boxstyle="square,pad=0.45", facecolor=dcol, edgecolor=dcol))

    if is_result:
        if status in ("MTG_WIN", "LOSS") and n > nb:
            ax.plot([nb], [h[nb] + span * 0.05], marker="x", markersize=11, color=RED,
                    markeredgewidth=2.8, zorder=9, clip_on=False)
        ax.annotate(r_pill, xy=(n - 1, h[-1] + span * 0.09), ha="center", va="bottom",
                    color="#000000", fontsize=10, fontweight="bold", zorder=10,
                    annotation_clip=False,
                    bbox=dict(boxstyle="square,pad=0.45", facecolor=rcol, edgecolor=rcol))

    # volume
    vx = fig.add_axes(rect(AX_X, VX_Y, AX_W, VX_H), zorder=2)
    vx.set_facecolor("none")
    vx.set_xlim(x0, x1)
    vx.set_ylim(0, vol.max() * 1.1)
    for s in vx.spines.values():
        s.set_visible(False)
    vx.set_yticks([])
    for i in range(n):
        col = GREEN if c_[i] >= o[i] else RED
        vx.add_patch(Rectangle((i - 0.3, 0), 0.6, vol[i], facecolor=col,
                               edgecolor="none", alpha=0.45))
    tp = list(range(0, n, 3))
    vx.set_xticks(tp)
    vx.set_xticklabels([times[i].strftime("%H:%M") for i in tp])
    vx.tick_params(axis="x", colors=DIM, labelsize=8.5, length=0, pad=6)
    for lab in vx.get_xticklabels():
        lab.set_family(MONO)

    # ================= TICKET COLUMN =================
    RX, RW = 1166, 414

    # order (top -> bottom): confidence, details, performance, levels, CALL/PUT, timer
    D = -126          # shift of the four cards (they moved up)
    B = 654           # y of the CALL/PUT block (sits right above the timer banner)

    # segmented confidence (slanted blades)
    _panel(bg, RX, 146 + D, RW, 88, c=16, fc=PANEL, ec=EDGE, lw=1.3)
    _t(bg, RX + 28, 170 + D, "CONFIDENCE", 9.5, MUTED)
    _t(bg, RX + RW - 24, 170 + D, f"{conf}%", 14, dcol, ha="right")
    segs, gap = 20, 4
    sw = (RW - 56 - gap * (segs - 1)) / segs
    filled = round(segs * conf / 100)
    for i in range(segs):
        sx0 = RX + 28 + i * (sw + gap)
        col = dcol if i < filled else TRACK
        bg.add_patch(Polygon([(sx0 + 4, 192 + D), (sx0 + sw + 4, 192 + D),
                              (sx0 + sw, 216 + D), (sx0, 216 + D)],
                             closed=True, facecolor=col, edgecolor="none", zorder=4))

    # trade details
    _panel(bg, RX, 248 + D, RW, 200, c=16, fc=PANEL, ec=EDGE, lw=1.3)
    _t(bg, RX + 28, 274 + D, "TRADE DETAILS", 9.5, MUTED)
    info = [
        ("Entry Time", entry_time.strftime("%H:%M"), SKY),
        ("Market", "OTC", BLUE),
        ("Payout", f"{pay}%", GREEN),
        ("Martingale", "1 Step", RED),
    ]
    for i, (lab, val, col) in enumerate(info):
        y = 312 + D + i * 34
        bg.add_patch(Polygon([(RX + 30, y - 5), (RX + 39, y), (RX + 30, y + 5)], closed=True,
                             facecolor=col, edgecolor="none", zorder=5))
        _t(bg, RX + 48, y, lab, 11.5, MUTED, weight="normal")
        _t(bg, RX + RW - 24, y, val, 13, col, ha="right", family=MONO)
        if i < len(info) - 1:
            bg.plot([RX + 28, RX + RW - 24], [y + 17, y + 17], color=EDGE, lw=0.8, zorder=3)

    # performance
    _panel(bg, RX, 462 + D, RW, 116, c=16, fc=PANEL, ec=EDGE, lw=1.3)
    _t(bg, RX + 28, 486 + D, "PERFORMANCE", 9.5, MUTED)
    _t(bg, RX + RW - 24, 486 + D, f"{wr:.0f}%" if wr is not None else "--", 14, GREEN, ha="right")
    bg.add_patch(Rectangle((RX + 28, 500 + D), RW - 52, 8, facecolor=TRACK, edgecolor="none", zorder=3))
    if wr:
        bg.add_patch(Rectangle((RX + 28, 500 + D), max((RW - 52) * wr / 100, 8), 8,
                               facecolor=GREEN, edgecolor="none", zorder=4))
    for i, (num, lab, col) in enumerate([(wins, "WINS", GREEN), (losses, "LOSSES", RED), (total, "TOTAL", BLUE)]):
        bx_ = RX + 28 + i * 124
        _panel(bg, bx_, 520 + D, 114, 44, c=9, fc=TILE, ec=EDGE, lw=0.9, cut=None)
        _t(bg, bx_ + 57, 536 + D, str(num), 16, col, ha="center", family=MONO)
        _t(bg, bx_ + 57, 553 + D, lab, 7.5, MUTED, ha="center")

    # price vs levels gauge
    _panel(bg, RX, 592 + D, RW, 174, c=16, fc=PANEL, ec=EDGE, lw=1.3)
    _t(bg, RX + 28, 616 + D, "PRICE VS LEVELS", 9.5, MUTED)
    gx0, gw, gy, gh = RX + 32, RW - 64, 672 + D, 14
    steps = 48
    swd = gw / steps
    for i in range(steps):
        col = _lerp(GREEN, RED, i / (steps - 1))
        bg.add_patch(Rectangle((gx0 + i * swd, gy), swd + 0.6, gh, facecolor=col,
                               edgecolor="none", zorder=4))
    frac = float(np.clip((cur - support) / rng, 0, 1))
    mx = gx0 + frac * gw
    bg.plot([mx, mx], [gy - 3, gy + gh + 6], color="#FFFFFF", lw=2.2, zorder=7)
    bg.add_patch(Polygon([(mx, gy - 3), (mx - 7, gy - 14), (mx + 7, gy - 14)],
                         closed=True, facecolor="#FFFFFF", edgecolor="none", zorder=7))
    px_c = float(np.clip(mx, gx0 + 52, gx0 + gw - 52))
    _panel(bg, px_c - 52, 630 + D, 104, 26, c=7, fc=BLUE, ec=BLUE, lw=0, cut=None, z=8)
    _t(bg, px_c, 643 + D, _fmt(cur), 10.5, "#FFFFFF", ha="center", family=MONO, z=9)
    _t(bg, gx0, 708 + D, "SUPPORT", 8, MUTED, va="bottom")
    _t(bg, gx0, 712 + D, _fmt(support), 13, GREEN, va="top", family=MONO)
    _t(bg, gx0 + gw, 708 + D, "RESISTANCE", 8, MUTED, ha="right", va="bottom")
    _t(bg, gx0 + gw, 712 + D, _fmt(resistance), 13, RED, ha="right", va="top", family=MONO)

    # CALL / PUT block, right above the timer (no eye)
    _panel(bg, RX, B, RW, 112, c=22, fc=dcol, ec=dcol, lw=0, cut=BLUE, glow=False)
    bg.add_patch(Circle((RX + 62, B + 56), 30, facecolor="#000000", edgecolor="none", zorder=5, alpha=0.18))
    tri = ([(RX + 62, B + 42), (RX + 80, B + 70), (RX + 44, B + 70)] if is_buy
           else [(RX + 44, B + 42), (RX + 80, B + 42), (RX + 62, B + 70)])
    bg.add_patch(Polygon(tri, closed=True, facecolor="#000000", edgecolor="none", zorder=7))
    _t(bg, RX + 110, B + 46, dtxt, 38, "#000000")
    _t(bg, RX + 112, B + 86, dsub, 9.5, "#000000")
    _t(bg, RX + RW - 20, B + 18, f"ENTRY {entry_time.strftime('%H:%M')}", 9.5, "#000000",
       ha="right", family=MONO)

    # ================= STATUS BANNER (no hazard stripes) =================
    if is_result:
        bcol = rcol
        fill = "#03140A" if rcol == GREEN else "#18030A"
    else:
        bcol, fill = BLUE, "#030A1E"
    _panel(bg, RX, 780, RW, 100, c=20, fc=fill, ec=bcol, lw=2.4, cut=RED if not is_result else bcol,
           glow=True)
    cxm = RX + RW / 2
    if is_result:
        _t(bg, cxm, 800, "RESULT", 8.5, MUTED, ha="center")
        _t(bg, cxm, 835, r_title, 30, rcol, ha="center", glow=rcol)
        _t(bg, cxm, 864, r_sub, 7.5, MUTED, ha="center")
    else:
        _t(bg, cxm, 800, "ENTRY TIME", 8.5, MUTED, ha="center")
        _t(bg, cxm, 835, entry_time.strftime("%H:%M"), 32, SKY, ha="center",
           family=MONO, glow=BLUE)
        _t(bg, cxm, 864, "WAIT FOR THE ENTRY CANDLE", 7.5, MUTED, ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


# ============================================
# Rendering helper (one render at a time, off the event loop)
# ============================================
async def _render(**kwargs) -> bytes:
    global _RENDER_LOCK
    if _RENDER_LOCK is None:
        _RENDER_LOCK = asyncio.Lock()
    async with _RENDER_LOCK:
        return await asyncio.to_thread(render_signal_chart, **kwargs)


def _render_kwargs(sig: dict, candles: List[dict], status: str, n_base: int) -> dict:
    return dict(
        candles=candles,
        symbol=sig["symbol"],
        direction=sig["direction"],
        payout=sig["payout"],
        entry_time=sig["entry_time"],
        status=status,
        n_base=n_base,
        stats=(STATS["wins"], STATS["losses"]),
        confidence=sig["strength"],
    )


# ============================================
# Signal lifecycle: wait -> result
# ============================================
async def _sleep_until(moment: datetime) -> None:
    delay = (moment - datetime.now()).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)


WAIT_TICK_SECONDS = 2     # the "Waiting for result" counter changes every N seconds


ENTRY_LABEL = "Waiting for entry..."
WAIT_LABEL = "Waiting for result..."
MTG_LABEL = "Martingal..."


def _wait_caption(remaining: int, label: str = WAIT_LABEL) -> tuple:
    """'⌛ Waiting for result... 59s' (animated hourglass, seconds only) -> (text, entities)."""
    remaining = min(59, max(0, int(remaining)))
    return _compose([("ce", "hourglass"), f" {fx(label)} {remaining:02d}s"])


def _secs_left(target: datetime) -> int:
    return max(0, int(round((target - datetime.now()).total_seconds())))


async def _wait_ticker(bot, chat_id, msg_id: int, state: dict) -> None:
    """Edits the 'waiting' message every WAIT_TICK_SECONDS until it is cancelled."""
    while True:
        await asyncio.sleep(WAIT_TICK_SECONDS)
        text, ents = _wait_caption(_secs_left(state["target"]), state["label"])
        try:
            if state["entities"]:
                await bot.edit_message_text(chat_id=chat_id, message_id=msg_id,
                                            text=text, entities=ents)
            else:
                await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text)
        except asyncio.CancelledError:
            raise
        except RetryAfter as e:
            await asyncio.sleep(float(e.retry_after))
        except Exception as e:
            if "not modified" not in str(e).lower():
                print(f"wait counter edit skipped: {e}")


async def _signal_lifecycle(bot, chat_id, sig: dict, message) -> None:
    """Runs after the signal photo is sent: waits, then posts the result as a NEW message."""
    wait_msg = None
    ticker = None
    state = {"target": sig["entry_time"], "label": ENTRY_LABEL, "entities": True}
    try:
        base = sig["candles"]
        n_base = len(base)

        # ---- 0) signal sent: "Waiting for entry... 42" (seconds only, every 2 s) ----
        reply0 = ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True)
        text, ents = _wait_caption(_secs_left(state["target"]), state["label"])
        try:
            wait_msg = await bot.send_message(chat_id=chat_id, text=text, entities=ents,
                                              reply_parameters=reply0)
        except Exception as e:
            print(f"waiting message with animated emoji failed ({e}); retrying plain")
            state["entities"] = False
            try:
                wait_msg = await bot.send_message(chat_id=chat_id, text=text,
                                                  reply_parameters=reply0)
            except Exception as e2:
                print(f"waiting message skipped: {e2}")
        if wait_msg is not None:
            ticker = asyncio.create_task(_wait_ticker(bot, chat_id, wait_msg.message_id, state))

        # ---- 1) wait for entry (the signal photo itself is never edited) ----
        await _sleep_until(sig["entry_time"])
        # same message now counts the entry candle: "Waiting for result... 59"
        state["target"] = sig["entry_time"] + timedelta(seconds=CANDLE_SECONDS)
        state["label"] = WAIT_LABEL

        # ---- 2) entry candle ----
        await _sleep_until(sig["entry_time"] + timedelta(seconds=CANDLE_SECONDS + 1))
        c1 = generate_next_candles(base, 1, sig["entry_time"])[0]
        candles = base + [c1]

        if candle_wins(sig["direction"], c1):
            status = "WIN"
        else:
            # ---- 3) martingale candle ----
            state["target"] = sig["entry_time"] + timedelta(seconds=2 * CANDLE_SECONDS)
            state["label"] = MTG_LABEL
            await _sleep_until(sig["entry_time"] + timedelta(seconds=2 * CANDLE_SECONDS + 1))
            c2 = generate_next_candles(candles, 1, sig["entry_time"] + timedelta(seconds=CANDLE_SECONDS))[0]
            candles = candles + [c2]
            status = "MTG_WIN" if candle_wins(sig["direction"], c2) else "LOSS"

        # ---- 4) update real counters and send the result image ----
        if status == "LOSS":
            STATS["losses"] += 1
        else:
            STATS["wins"] += 1
        save_stats()

        png = await _render(**_render_kwargs(sig, candles, status, n_base))
        photo = io.BytesIO(png)
        photo.name = f"{sig['symbol']}_result.png"
        result_caption, result_entities = build_result_caption(sig, status, candles[-1])
        reply = ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True)
        try:
            await bot.send_photo(
                chat_id=chat_id, photo=photo, caption=result_caption,
                caption_entities=result_entities, reply_parameters=reply,
                read_timeout=60, write_timeout=60, connect_timeout=30,
            )
        except Exception as e:
            print(f"result with animated emoji failed ({e}); retrying plain")
            photo.seek(0)
            await bot.send_photo(
                chat_id=chat_id, photo=photo,
                caption=build_result_caption(sig, status, candles[-1], emoji_symbol=False)[0],
                reply_parameters=reply,
                read_timeout=60, write_timeout=60, connect_timeout=30,
            )
        print(f"OUT result {status} for {sig['symbol']} (W{STATS['wins']}/L{STATS['losses']})")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"ERROR in signal lifecycle: {e}")
    finally:
        if ticker is not None:
            ticker.cancel()
            try:
                await ticker
            except BaseException:
                pass
        if wait_msg is not None:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=wait_msg.message_id)
            except Exception as e:
                print(f"could not delete the waiting message: {e}")


def _spawn_lifecycle(bot, chat_id, sig, message) -> None:
    global LAST_LIFECYCLE_TASK
    task = asyncio.create_task(_signal_lifecycle(bot, chat_id, sig, message))
    LAST_LIFECYCLE_TASK = task
    LIFECYCLE_TASKS.add(task)
    task.add_done_callback(LIFECYCLE_TASKS.discard)


async def _send_signal_to(chat_id, context, direction: Optional[str] = None,
                          check_forbidden: bool = True) -> bool:
    """
    Send one signal (no result yet) and schedule its result.
    `context` may be a CallbackContext or an Application (both expose `.bot`).
    """
    global CHAT_FORBIDDEN

    try:
        sig = build_signal(direction)
        png = await _render(**_render_kwargs(sig, sig["candles"], "PENDING", len(sig["candles"])))
    except Exception as e:
        print(f"ERROR building payload: {e}")
        return False

    caption, entities = build_signal_caption(sig)
    plain_caption = build_signal_caption(sig, emoji_symbol=False)[0]
    photo_io = io.BytesIO(png)
    photo_io.name = f"{sig['symbol']}_chart.png"

    MAX_RETRIES = 3
    RETRY_DELAY = 3

    for use_entities in (True, False):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                photo_io.seek(0)
                kwargs = dict(chat_id=chat_id, photo=photo_io,
                              caption=caption if use_entities else plain_caption,
                              read_timeout=60, write_timeout=60, connect_timeout=30)
                if use_entities:
                    kwargs.update(caption_entities=entities, parse_mode=None)
                message = await context.bot.send_photo(**kwargs)
                print(f"OUT signal {sig['direction']} for {sig['symbol']} sent to chat_id={chat_id} "
                      f"({'animated' if use_entities else 'plain'} emoji, attempt {attempt})")
                _spawn_lifecycle(context.bot, chat_id, sig, message)
                return True
            except Exception as e:
                err_str = str(e)
                if check_forbidden and ("Forbidden" in err_str or "bot can't" in err_str.lower()):
                    print("\n" + "=" * 60)
                    print("CRITICAL: Telegram rejected this chat_id.")
                    print(f"   Telegram said : {err_str}")
                    print(f"   chat_id used  : {chat_id}")
                    print("   It is most likely a bot's id (bots cannot message bots).")
                    print("   Cached chat_id cleared. Send /start to your bot from")
                    print("   YOUR personal Telegram account to re-register.")
                    print("=" * 60 + "\n")
                    try:
                        CHATID_CACHE_FILE.unlink(missing_ok=True)
                    except Exception:
                        pass
                    CHAT_FORBIDDEN = True
                    return False
                print(f"ERROR send attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
    return False


# ============================================
# Auto-signal loop
# ============================================
AUTO_SIGNAL_TASK: Optional[asyncio.Task] = None


async def _ensure_auto_signal_task(application: Application) -> None:
    global AUTO_SIGNAL_TASK, CHAT_FORBIDDEN
    if AUTO_SIGNAL_TASK is not None and not AUTO_SIGNAL_TASK.done():
        return
    CHAT_FORBIDDEN = False
    AUTO_SIGNAL_TASK = asyncio.create_task(auto_signal_loop(application))


async def _stop_auto_signal_task(application: Application) -> None:
    """Stop the loop. Signals already sent still deliver their result."""
    global AUTO_SIGNAL_TASK
    task = AUTO_SIGNAL_TASK
    if task is not None and not task.done() and task is not asyncio.current_task():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    AUTO_SIGNAL_TASK = None


async def auto_signal_loop(application: Application) -> None:
    """
    ONE signal at a time:
        signal -> entry countdown -> result (after the 60 s trade)
        -> wait AUTO_SIGNAL_GAP_AFTER_RESULT seconds -> next signal
    """
    global LAST_LIFECYCLE_TASK
    await asyncio.sleep(AUTO_SIGNAL_START_DELAY)

    if not AUTO_SIGNAL_CHAT_ID:
        print("AUTO-SIGNAL: no chat_id configured; loop disabled.")
        return

    print(f"AUTO-SIGNAL: starting loop targeting chat_id={AUTO_SIGNAL_CHAT_ID}")
    print(f"AUTO-SIGNAL: one signal at a time; next signal {AUTO_SIGNAL_GAP_AFTER_RESULT}s "
          f"after each result")
    count = 0
    try:
        while True:
            count += 1
            print(f"\n--- AUTO-SIGNAL #{count} ---")
            LAST_LIFECYCLE_TASK = None
            ok = False
            try:
                ok = await _send_signal_to(AUTO_SIGNAL_CHAT_ID, application,
                                           direction=None, check_forbidden=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"ERROR sending auto signal #{count}: {e}")
            if CHAT_FORBIDDEN:
                print("AUTO-SIGNAL: stopping loop due to invalid chat_id.")
                return
            task = LAST_LIFECYCLE_TASK
            if ok and task is not None:
                try:
                    # shield: /stop must not kill a result that is about to be posted
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"ERROR while waiting for the result: {e}")
            print(f"--- signal #{count} finished; next signal in "
                  f"{AUTO_SIGNAL_GAP_AFTER_RESULT}s ---")
            await asyncio.sleep(AUTO_SIGNAL_GAP_AFTER_RESULT)
    except asyncio.CancelledError:
        print("AUTO-SIGNAL: loop cancelled.")


async def _print_identity(application: Application) -> None:
    """Show WHICH bot we are and WHO the auto-signals go to (diagnostics)."""
    try:
        me = await application.bot.get_me()
        print(f"BOT      : @{me.username} (id={me.id})   <- open THIS bot in Telegram")
    except Exception as e:
        print(f"WARNING: get_me failed: {e}")
    if AUTO_SIGNAL_CHAT_ID:
        try:
            chat = await application.bot.get_chat(AUTO_SIGNAL_CHAT_ID)
            name = getattr(chat, "full_name", None) or getattr(chat, "title", None) or ""
            print(f"TARGET   : chat_id={chat.id} type={chat.type} name='{name}' "
                  f"username=@{getattr(chat, 'username', None)}")
            print("           Not your account? Delete .chatid_cache, then send /start "
                  "to the bot from YOUR account.")
        except Exception as e:
            print(f"WARNING: cannot read the target chat ({e}).")
            print("         Send /start to the bot from your account to register it.")


async def post_init(application: Application) -> None:
    await _print_identity(application)
    if AUTO_SIGNAL_ENABLED and AUTO_SIGNAL_CHAT_ID:
        await _ensure_auto_signal_task(application)
    else:
        print("\n" + "=" * 60)
        print("NO chat_id configured yet.")
        print("   Send /start to your bot from YOUR personal Telegram account")
        print("   to register your chat_id and start the AUTO-SIGNAL loop.")
        print("=" * 60 + "\n")


async def post_shutdown(application: Application) -> None:
    await _stop_auto_signal_task(application)
    for t in list(LIFECYCLE_TASKS):
        t.cancel()
    if LIFECYCLE_TASKS:
        await asyncio.gather(*LIFECYCLE_TASKS, return_exceptions=True)


# ============================================
# Command handlers
# ============================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_SIGNAL_CHAT_ID, REGISTRATION_OPEN

    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    print(f"IN  /start from chat_id={chat_id} (type={chat_type})")

    if chat_type != "private":
        await update.message.reply_text(
            "\u26A0\uFE0F Please send /start to me in a PRIVATE chat "
            "to register your personal chat_id."
        )
        return

    text = build_welcome_message()
    entities = build_custom_emoji_entities(text, SPECIAL_EMOJI_MAP)
    await update.message.reply_text(text, entities=entities)

    if CHAT_ID_OVERRIDE:
        await update.message.reply_text(
            f"\u2139\uFE0F The target chat is fixed in the code: {CHAT_ID_OVERRIDE}\n"
            f"   (this chat's id is {chat_id}; use /id to compare)"
        )
        if AUTO_SIGNAL_ENABLED:
            await _ensure_auto_signal_task(context.application)
        return

    if (AUTO_SIGNAL_CHAT_ID and str(chat_id) != str(AUTO_SIGNAL_CHAT_ID)
            and not ALLOW_START_TO_REGISTER and not REGISTRATION_OPEN):
        await update.message.reply_text(
            f"\u2139\uFE0F Signals go to chat_id {AUTO_SIGNAL_CHAT_ID} "
            f"(entered when the bot started).\n"
            f"   Your chat_id is {chat_id}. Restart the bot and enter YOUR chat_id "
            f"to receive them here."
        )
        return

    if not AUTO_SIGNAL_CHAT_ID or AUTO_SIGNAL_CHAT_ID != str(chat_id):
        AUTO_SIGNAL_CHAT_ID = str(chat_id)
        REGISTRATION_OPEN = False
        _save_cache(CHATID_CACHE_FILE, AUTO_SIGNAL_CHAT_ID)
        print(f"REGISTERED chat_id={chat_id} (saved to {CHATID_CACHE_FILE.name})")
        await update.message.reply_text(
            f"\u2705 Your chat_id ({chat_id}) is now registered!\n"
            f"\U0001F501 AUTO-SIGNAL: one signal at a time, starting in {AUTO_SIGNAL_START_DELAY}s..."
        )
        if AUTO_SIGNAL_ENABLED:
            await _stop_auto_signal_task(context.application)
            await _ensure_auto_signal_task(context.application)
    else:
        if AUTO_SIGNAL_ENABLED and (AUTO_SIGNAL_TASK is None or AUTO_SIGNAL_TASK.done()):
            await _ensure_auto_signal_task(context.application)


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"IN  /stop from chat_id={update.effective_chat.id}")
    await _stop_auto_signal_task(context.application)
    await update.message.reply_text(
        "\U0001F6D1 AUTO-SIGNAL stopped.\n"
        "   Signals already sent will still post their result.\n"
        "   Use /resume to restart, or /signal for a one-off signal."
    )


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"IN  /resume from chat_id={update.effective_chat.id}")
    if not AUTO_SIGNAL_CHAT_ID:
        await update.message.reply_text("\u26A0\uFE0F No chat_id registered. Send /start first.")
        return
    await _ensure_auto_signal_task(context.application)
    await update.message.reply_text(
        "\u25B6\uFE0F AUTO-SIGNAL resumed.\n"
        f"   One signal at a time to chat_id={AUTO_SIGNAL_CHAT_ID}."
    )


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"IN  /signal from chat_id={update.effective_chat.id}")
    await _send_signal_to(update.effective_chat.id, context, direction=None,
                          check_forbidden=False)


async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"IN  /buy from chat_id={update.effective_chat.id}")
    await _send_signal_to(update.effective_chat.id, context, direction="BUY",
                          check_forbidden=False)


async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"IN  /sell from chat_id={update.effective_chat.id}")
    await _send_signal_to(update.effective_chat.id, context, direction="SELL",
                          check_forbidden=False)


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chart only (no result follow-up)."""
    chat_id = update.effective_chat.id
    try:
        sig = build_signal()
        png = await _render(**_render_kwargs(sig, sig["candles"], "PENDING", len(sig["candles"])))
    except Exception as e:
        await update.message.reply_text(f"ERROR rendering chart: {e}")
        return
    photo_io = io.BytesIO(png)
    photo_io.name = f"{sig['symbol']}_chart.png"
    await context.bot.send_photo(
        chat_id=chat_id,
        photo=photo_io,
        caption=f"\U0001F4CA {sig['symbol'].replace('_otc', '-OTC').upper()}  -  M1  -  Virtual Candles",
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = build_status_message()
    entities = build_custom_emoji_entities(text, SPECIAL_EMOJI_MAP)
    await update.message.reply_text(text, entities=entities)


async def error_handler(update, context: ContextTypes.DEFAULT_TYPE):
    """Quiet, one-line reporting instead of long tracebacks."""
    err = context.error
    if isinstance(err, RetryAfter):
        print(f"NOTE: Telegram asks to slow down - waiting {err.retry_after}s automatically.")
    elif isinstance(err, NetworkError):      # includes BadGateway and TimedOut
        print(f"NOTE: temporary Telegram/network hiccup ({type(err).__name__}: {err}) "
              f"- retrying automatically, nothing to do.")
    else:
        print(f"ERROR: {type(err).__name__}: {err}")


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/id - compare THIS chat with the auto-signal target."""
    cid = update.effective_chat.id
    ok = str(cid) == str(AUTO_SIGNAL_CHAT_ID)
    await update.message.reply_text(
        f"Your chat_id: {cid}\n"
        f"Auto-signal target: {AUTO_SIGNAL_CHAT_ID or 'unset'}\n"
        + ("\u2705 Match - signals are delivered to this chat."
           if ok else "\u274C Different - send /start here to make THIS chat the target.")
    )


async def resetstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATS["wins"] = 0
    STATS["losses"] = 0
    save_stats()
    await update.message.reply_text("\u2705 WINS / LOSSES counters reset to 0.")


# ============================================
# Interactive prompts (token + chat_id)
# ============================================
def _read_cache(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _save_cache(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
    except Exception as e:
        print(f"   WARNING: failed to save cache: {e}")


def prompt_for_token() -> str:
    cached = _read_cache(TOKEN_CACHE_FILE)
    if cached and ":" in cached and len(cached) >= 20:
        print(f"FOUND cached token in: {TOKEN_CACHE_FILE.name}")
        use = input("   Reuse it? (Y/n): ").strip().lower() or "y"
        if use in ("y", "yes"):
            return cached

    print("=" * 60)
    print("No Telegram bot token found!")
    print("   1) Open @BotFather in Telegram")
    print("   2) Send /newbot and follow the prompts")
    print("   3) Copy the token (format: 123456789:ABCdefGhI...)")
    print("=" * 60)

    while True:
        try:
            token = input("PASTE your bot token here: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return ""
        if not token:
            continue
        if ":" not in token or len(token) < 20:
            print("WARNING: invalid format (must contain ':' and be >= 20 chars).")
            continue
        print(f"   OK - token received ({token[:8]}...{token[-4:]})")
        break

    save = input("   Save token locally for next time? (Y/n): ").strip().lower() or "y"
    if save in ("y", "yes"):
        _save_cache(TOKEN_CACHE_FILE, token)
    return token


def prompt_for_chat_id_now():
    """Ask for the target chat_id at every start. Enter = skip (open registration)."""
    print("=" * 60)
    print("ENTER the chat_id that must receive the signals")
    print("   - Your own id : open the bot in Telegram and send /id")
    print("   - Channel/group ids look like -1001234567890 (bot must be admin)")
    print("   - Private chat: that person must have sent /start to the bot once")
    print("   - Press Enter to skip: then the FIRST /start you send registers you")
    print("=" * 60)
    while True:
        try:
            chat_id = input("chat_id > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return None
        if not chat_id:
            return ""
        if not chat_id.lstrip("-").isdigit():
            print("A chat_id is a number (it may start with '-'). Try again.")
            continue
        return chat_id


def prompt_for_chat_id() -> str:
    cached = _read_cache(CHATID_CACHE_FILE)
    if cached and cached.lstrip("-").isdigit():
        print(f"FOUND cached chat_id in: {CHATID_CACHE_FILE.name}")
        use = input("   Reuse it? (Y/n): ").strip().lower() or "y"
        if use in ("y", "yes"):
            return cached

    print("=" * 60)
    print("No chat_id configured for AUTO-SIGNAL.")
    print("   Easiest: press Enter to skip, then send /start to the bot")
    print("   from YOUR personal Telegram account.")
    print("   (Do not paste the bot's own id - bots cannot message bots.)")
    print("=" * 60)

    while True:
        try:
            chat_id = input("PASTE your chat_id here (Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return ""
        if not chat_id:
            return ""
        if not chat_id.lstrip("-").isdigit():
            print("WARNING: chat_id must be a number. Try again.")
            continue
        break

    save = input("   Save chat_id locally for next time? (Y/n): ").strip().lower() or "y"
    if save in ("y", "yes"):
        _save_cache(CHATID_CACHE_FILE, chat_id)
    return chat_id


# ============================================
# Entry point
# ============================================
def main():
    global TELEGRAM_BOT_TOKEN, AUTO_SIGNAL_CHAT_ID, REGISTRATION_OPEN

    if not TELEGRAM_BOT_TOKEN:
        TELEGRAM_BOT_TOKEN = prompt_for_token()
        if not TELEGRAM_BOT_TOKEN:
            print("ERROR: no token provided. Exiting.")
            return

    # ---- chat_id: ALWAYS asked at startup; old saved/env ids are wiped ----
    if CHAT_ID_OVERRIDE:
        AUTO_SIGNAL_CHAT_ID = str(CHAT_ID_OVERRIDE).strip()
        print(f"OK - using CHAT_ID_OVERRIDE from the code: {AUTO_SIGNAL_CHAT_ID}")
    elif ALWAYS_ASK_CHAT_ID:
        try:
            if CHATID_CACHE_FILE.exists():
                CHATID_CACHE_FILE.unlink()
                print("OK - old saved chat_id wiped (.chatid_cache deleted).")
        except Exception as e:
            print(f"WARNING: could not delete {CHATID_CACHE_FILE.name}: {e}")
        if _ENV_CHAT_ID:
            print(f"NOTE - ignoring the chat_id found in the environment ({_ENV_CHAT_ID}).")
        entered = prompt_for_chat_id_now()
        if entered is None:
            return
        AUTO_SIGNAL_CHAT_ID = entered
        if not entered:
            REGISTRATION_OPEN = True
            print("NOTE - no chat_id entered: the FIRST /start sent to the bot registers "
                  "that chat for this run only (nothing is saved).")
    else:
        AUTO_SIGNAL_CHAT_ID = _ENV_CHAT_ID or _read_cache(CHATID_CACHE_FILE)
        if not AUTO_SIGNAL_CHAT_ID:
            AUTO_SIGNAL_CHAT_ID = prompt_for_chat_id()

    print("=" * 60)
    print("STARTING QUANTVEXA BOT")
    print(f"   VERSION    : {BOT_VERSION}")
    print(f"   AUTO-SIGNAL: 1 at a time, next {AUTO_SIGNAL_GAP_AFTER_RESULT}s after each result "
          f"-> chat_id={AUTO_SIGNAL_CHAT_ID or 'unset'}")
    print(f"   RESULTS    : {STATS['wins']}W / {STATS['losses']}L")
    print(f"   EMOJI      : {'enabled' if SPECIAL_EMOJI_MAP else 'disabled'}")
    print("=" * 60)

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("buy", buy_command))
    app.add_handler(CommandHandler("sell", sell_command))
    app.add_handler(CommandHandler("chart", chart_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("resetstats", resetstats_command))
    app.add_handler(CommandHandler("id", id_command))

    app.add_error_handler(error_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
