import os
import json
import requests
import pandas as pd

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "3min"
OUTPUT_SIZE = 500
STATE_FILE = "state.json"

DI_LENGTH_A = 14
DI_LENGTH_B = 100
SMA_LENGTH = 100


def fetch_candles():
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def wilder_smooth(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()


def compute_di(df, length):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    plus_dm = plus_dm.fillna(0)
    minus_dm = minus_dm.fillna(0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    smoothed_tr = wilder_smooth(tr, length)
    smoothed_plus_dm = wilder_smooth(plus_dm, length)
    smoothed_minus_dm = wilder_smooth(minus_dm, length)

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    return plus_di, minus_di


def detect_crossover(plus_di, minus_di):
    diff_now = plus_di.iloc[-1] - minus_di.iloc[-1]
    diff_prev = plus_di.iloc[-2] - minus_di.iloc[-2]

    if diff_prev <= 0 and diff_now > 0:
        return "bullish"
    if diff_prev >= 0 and diff_now < 0:
        return "bearish"
    return None


def current_side(plus_di, minus_di):
    return "bullish" if plus_di.iloc[-1] > minus_di.iloc[-1] else "bearish"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    resp = requests.post(url, data=payload, timeout=30)
    resp.raise_for_status()


def fa_dir(direction):
    return "صعودی" if direction == "bullish" else "نزولی"


def main():
    state = load_state()
    messages = []

    df = fetch_candles()
    latest_time = str(df["time"].iloc[-1])
    latest_close = df["close"].iloc[-1]
    latest_high = df["high"].iloc[-1]
    latest_low = df["low"].iloc[-1]

    plus_di_a, minus_di_a = compute_di(df, DI_LENGTH_A)
    plus_di_b, minus_di_b = compute_di(df, DI_LENGTH_B)
    sma_b = df["close"].rolling(SMA_LENGTH).mean()
    latest_sma = sma_b.iloc[-1]

    # ---- Condition 1: independent crossover alerts for DI14 and DI100 ----
    cross_a = detect_crossover(plus_di_a, minus_di_a)
    if cross_a is not None and state.get("cond1_last_bar_a") != latest_time:
        messages.append(
            f"طلا (XAU/USD) - 3min\n"
            f"شرط ۱: تقاطع DI{DI_LENGTH_A} {fa_dir(cross_a)}\n"
            f"قیمت: {latest_close:.2f}\n"
            f"زمان کندل: {latest_time}"
        )
        state["cond1_last_bar_a"] = latest_time

    cross_b = detect_crossover(plus_di_b, minus_di_b)
    if cross_b is not None and state.get("cond1_last_bar_b") != latest_time:
        messages.append(
            f"طلا (XAU/USD) - 3min\n"
            f"شرط ۱: تقاطع DI{DI_LENGTH_B} {fa_dir(cross_b)}\n"
            f"قیمت: {latest_close:.2f}\n"
            f"زمان کندل: {latest_time}"
        )
        state["cond1_last_bar_b"] = latest_time

    # ---- Condition 2: DI100 side and DI14 side aligned, every check ----
    side_a = current_side(plus_di_a, minus_di_a)
    side_b = current_side(plus_di_b, minus_di_b)

    if side_a == side_b and state.get("cond2_last_bar") != latest_time:
        messages.append(
            f"طلا (XAU/USD) - 3min\n"
            f"شرط ۲: DI{DI_LENGTH_B} و DI{DI_LENGTH_A} هم‌جهت {fa_dir(side_b)} هستند\n"
            f"قیمت: {latest_close:.2f}\n"
            f"زمان کندل: {latest_time}"
        )
        state["cond2_last_bar"] = latest_time

    # ---- Condition 3: price touches SMA100, in the direction of DI100 ----
    touched_sma = latest_low <= latest_sma <= latest_high
    if touched_sma and state.get("cond3_last_bar") != latest_time:
        messages.append(
            f"طلا (XAU/USD) - 3min\n"
            f"شرط ۳: قیمت به میانگین متحرک {SMA_LENGTH} برخورد کرد، DI{DI_LENGTH_B} {fa_dir(side_b)} است\n"
            f"قیمت: {latest_close:.2f} (میانگین: {latest_sma:.2f})\n"
            f"زمان کندل: {latest_time}"
        )
        state["cond3_last_bar"] = latest_time

    save_state(state)

    if messages:
        send_telegram_message("\n\n---\n\n".join(messages))
    else:
        print("No new signal.")


if __name__ == "__main__":
    main()
