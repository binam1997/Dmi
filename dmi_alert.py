import json
import os

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


IRAN_TZ = ZoneInfo("Asia/Tehran")


# =========================================================
# SETTINGS
# =========================================================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "5min"
BAR_MINUTES = 5

OUTPUT_SIZE = 1000
WARMUP_BARS = 300
MIN_CANDLES = WARMUP_BARS + 50

# signals older than this many closed candles are ignored (protects from late runs)
MAX_AGE_BARS = 3

STATE_FILE = "state.json"
ERROR_ALERT_MINUTES = 60

# signal logic
WINDOW = 10
RE_ENTRY = True

# MACD-V (MACD Dive LTF)
M_FAST = 30
M_SLOW = 100
M_SIG = 20
M_ATR = 100

# WPRBB (Loxx)
W_LEN = 30
W_SM = 20
BB_LEN = 100
W_HL = True

# stop / take profit
STOP_MODE = "Swing"  # "Swing" or "ATR"
SWING_LEN = 10
ATR_LEN = 14
ATR_BUF = 0.3
ATR_MULT = 2.0
RR = 1.5


# =========================================================
# HELPERS
# =========================================================

def safe_text(text):

    text = str(text)

    for secret in (TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN):
        if secret:
            text = text.replace(secret, "***")

    return text


def load_state():

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


def save_state(state):

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# =========================================================
# DATA
# =========================================================

def get_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
        "timezone": "Asia/Tehran"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise Exception(f"TwelveData Error: {data}")

    df = pd.DataFrame(data["values"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(IRAN_TZ)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # keep only fully closed candles (datetime = candle open time)
    now = pd.Timestamp.now(tz=IRAN_TZ)
    closed = (df["datetime"] + pd.Timedelta(minutes=BAR_MINUTES)) <= now
    df = df[closed].reset_index(drop=True)

    return df


# =========================================================
# INDICATORS
# =========================================================

def ema(series, length):

    return series.ewm(span=length, adjust=False).mean()


def rma(series, length):

    return series.ewm(alpha=1 / length, adjust=False).mean()


def calc_atr(df, length):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return rma(tr, length)


def calculate_indicators(df):

    # MACD-V
    df["macdv"] = (ema(df["close"], M_FAST) - ema(df["close"], M_SLOW)) / calc_atr(df, M_ATR)
    df["msig"] = ema(df["macdv"], M_SIG)

    # WPRBB
    hi = df["high"] if W_HL else df["close"]
    lo = df["low"] if W_HL else df["close"]
    highest = hi.rolling(W_LEN).max()
    lowest = lo.rolling(W_LEN).min()
    raw = (df["close"] - lowest) / (highest - lowest) * 200 - 100
    df["wpr"] = ema(raw, W_SM)
    df["wbas"] = ema(df["wpr"], BB_LEN)

    # stop helpers
    df["atr_s"] = calc_atr(df, ATR_LEN)
    df["swing_hi"] = df["high"].rolling(SWING_LEN).max()
    df["swing_lo"] = df["low"].rolling(SWING_LEN).min()

    return df


# =========================================================
# SIGNAL LOGIC
# =========================================================

def run_logic(df):

    t = df["datetime"].tolist()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    mv = df["macdv"].to_numpy()
    ms = df["msig"].to_numpy()
    wp = df["wpr"].to_numpy()
    wb = df["wbas"].to_numpy()
    a = df["atr_s"].to_numpy()
    sh = df["swing_hi"].to_numpy()
    sl = df["swing_lo"].to_numpy()

    signals = []
    last_dir = 0
    status = "-"
    stop_lv = None
    tp_lv = None
    lm_up = None
    lm_dn = None
    lw_up = None
    lw_dn = None

    for i in range(1, len(df)):

        if i < WARMUP_BARS:
            continue

        m_up = bool(mv[i] > ms[i] and mv[i - 1] <= ms[i - 1])
        m_dn = bool(mv[i] < ms[i] and mv[i - 1] >= ms[i - 1])
        w_up = bool(wp[i] > wb[i] and wp[i - 1] <= wb[i - 1])
        w_dn = bool(wp[i] < wb[i] and wp[i - 1] >= wb[i - 1])

        if m_up:
            lm_up = i
        if m_dn:
            lm_dn = i
        if w_up:
            lw_up = i
        if w_dn:
            lw_dn = i

        # stop / take profit hit (levels from the previous bar)
        hit_stop = False
        hit_tp = False

        if stop_lv is not None:
            if last_dir == 1:
                hit_stop = bool(l[i] <= stop_lv)
                hit_tp = bool(h[i] >= tp_lv)
            elif last_dir == -1:
                hit_stop = bool(h[i] >= stop_lv)
                hit_tp = bool(l[i] <= tp_lv)

        def within(last):
            return last is not None and (i - last) <= WINDOW

        buy_ev = (m_up and wp[i] > wb[i] and within(lw_up)) or (w_up and mv[i] > ms[i] and within(lm_up))
        sell_ev = (m_dn and wp[i] < wb[i] and within(lw_dn)) or (w_dn and mv[i] < ms[i] and within(lm_dn))

        free = RE_ENTRY and status != "OPEN"
        buy_sig = buy_ev and (last_dir != 1 or free)
        sell_sig = sell_ev and (last_dir != -1 or free)

        if buy_sig and sell_sig:
            continue

        if buy_sig or sell_sig:

            direction = 1 if buy_sig else -1

            if STOP_MODE == "Swing":
                if direction == 1:
                    stop = sl[i] - ATR_BUF * a[i]
                else:
                    stop = sh[i] + ATR_BUF * a[i]
            else:
                if direction == 1:
                    stop = c[i] - ATR_MULT * a[i]
                else:
                    stop = c[i] + ATR_MULT * a[i]

            if direction == 1:
                tp = c[i] + RR * (c[i] - stop)
            else:
                tp = c[i] - RR * (stop - c[i])

            stop_lv = float(stop)
            tp_lv = float(tp)
            last_dir = direction
            status = "OPEN"

            signals.append({
                "i": i,
                "time": t[i],
                "dir": direction,
                "entry": float(c[i]),
                "stop": stop_lv,
                "tp": tp_lv,
                "macdv": float(mv[i]),
                "msig": float(ms[i]),
                "wpr": float(wp[i]),
                "wbas": float(wb[i]),
            })

        elif hit_stop or hit_tp:

            stop_lv = None
            tp_lv = None
            status = "STOPPED" if hit_stop else "TP HIT"

    return signals


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload, timeout=20).raise_for_status()


def send_error_alert(error_text):

    # at most one error alert per ERROR_ALERT_MINUTES (the bot may run every few minutes)
    try:
        state = load_state()
        now = pd.Timestamp.now(tz=IRAN_TZ)

        last = state.get("last_error_time")
        if last:
            elapsed = (now - pd.Timestamp(last)).total_seconds()
            if elapsed < ERROR_ALERT_MINUTES * 60:
                return

        send_telegram(f"⚠️ ربات سیگنال MACD-V + WPRBB خطا داد:\n\n{error_text}")

        state["last_error_time"] = now.isoformat()
        save_state(state)
    except Exception:
        pass


def build_message(s, last_idx, last_close):

    is_buy = s["dir"] == 1
    side = "BUY" if is_buy else "SELL"
    icon = "🟢" if is_buy else "🔴"

    risk = abs(s["entry"] - s["stop"])
    close_time = (s["time"] + pd.Timedelta(minutes=BAR_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")

    age = last_idx - s["i"]
    late_line = ""
    if age >= 1:
        late_line = f"\n⏱ این سیگنال {age} کندل پیش صادر شده | قیمت فعلی: {last_close:.2f}"

    return f"""
{icon} سیگنال {side} ({SYMBOL} - ۵ دقیقه)

💰 ورود (کلوز کندل): {s['entry']:.2f}
🛑 استاپ: {s['stop']:.2f} (ریسک {risk:.2f})
🎯 حد سود: {s['tp']:.2f} ({RR}R)

📊 MACD-V: {s['macdv']:.3f} | سیگنال: {s['msig']:.3f}
📉 WPR: {s['wpr']:.1f} | میدلاین: {s['wbas']:.1f}

━━━━━━━━━━━━━━
⏱ زمان بسته شدن کندل: {close_time}{late_line}
""".strip()


# =========================================================
# MAIN
# =========================================================

def main():

    if os.environ.get("TEST_MESSAGE") == "1":
        send_telegram("✅ ربات سیگنال MACD-V + WPRBB فعال است")
        print("Test message sent.")
        return

    print("Getting market data...")
    df = get_data()

    if len(df) < MIN_CANDLES:
        print(f"Not enough candles yet ({len(df)} < {MIN_CANDLES}).")
        return

    print("Calculating indicators...")
    df = calculate_indicators(df)

    signals = run_logic(df)

    state = load_state()
    last_alert = None
    if state.get("last_alert_time"):
        last_alert = pd.Timestamp(state["last_alert_time"])

    last_idx = len(df) - 1
    last_row = df.iloc[-1]
    last_close = float(last_row["close"])

    print(f"Last closed candle: {last_row['datetime']} | close: {last_close:.2f}")
    print(
        f"MACD-V: {last_row['macdv']:.3f} | Signal: {last_row['msig']:.3f} | "
        f"WPR: {last_row['wpr']:.2f} | Mid: {last_row['wbas']:.2f}"
    )

    pending = []
    for s in signals:
        fresh = (last_idx - s["i"]) < MAX_AGE_BARS
        is_new = last_alert is None or s["time"] > last_alert
        if fresh and is_new:
            pending.append(s)

    print(f"Signals found: {len(signals)} | pending: {len(pending)}")

    for s in pending:

        message = build_message(s, last_idx, last_close)

        print(f"Sending Alert:\n{message}")
        send_telegram(message)

        # save right after each successful send to avoid duplicates
        state["last_alert_time"] = s["time"].isoformat()
        save_state(state)

    save_state(state)

    print("Execution completed.")


# =========================================================

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {safe_text(e)}")
        send_error_alert(safe_text(e))
        raise
        
