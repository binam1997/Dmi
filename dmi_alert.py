import os
import json
import time
import requests
import pandas as pd

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
TIMEFRAMES = ["1min", "5min"]
OUTPUT_SIZE = 500
STATE_FILE = "state.json"

DI_LENGTH = 100


def fetch_candles(interval):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {SYMBOL} {interval}: {data}")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
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


def main():
    state = load_state()
    messages = []

    for interval in TIMEFRAMES:
        try:
            df = fetch_candles(interval)
        except Exception as e:
            print(f"Skipping {interval}: {e}")
            continue

        latest_time = str(df["time"].iloc[-1])
        latest_close = df["close"].iloc[-1]

        plus_di, minus_di = compute_di(df, DI_LENGTH)
        signal = detect_crossover(plus_di, minus_di)

        state_key = f"last_signal_{interval}"
        last_bar_key = f"last_bar_{interval}"

        if signal and state.get(last_bar_key) != latest_time:
            direction_fa = "صعودی (DI+ از بالای DI- عبور کرد)" if signal == "bullish" else "نزولی (DI- از بالای DI+ عبور کرد)"
            messages.append(
                f"طلا (XAU/USD) - تایم‌فریم {interval}\n"
                f"سیگنال DMI (طول {DI_LENGTH}): {direction_fa}\n"
                f"قیمت: {latest_close:.2f}\n"
                f"زمان کندل: {latest_time}"
            )
            state[state_key] = signal
            state[last_bar_key] = latest_time

        time.sleep(1)

    if messages:
        send_telegram_message("\n\n---\n\n".join(messages))
        save_state(state)
    else:
        print("No new signal.")


if __name__ == "__main__":
    main()
