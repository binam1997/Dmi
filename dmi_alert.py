import os
import json
import time
import requests
import pandas as pd

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
TIMEFRAMES = ["3min", "5min"]
OUTPUT_SIZE = 500
STATE_FILE = "state.json"

DI_TREND_LENGTH = 100
DI_TRIGGER_LENGTH = 14


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


def get_trend_state(plus_di, minus_di):
    if plus_di.iloc[-1] > minus_di.iloc[-1]:
        return "bullish"
    return "bearish"


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

        plus_di_100, minus_di_100 = compute_di(df, DI_TREND_LENGTH)
        trend_state = get_trend_state(plus_di_100, minus_di_100)

        plus_di_14, minus_di_14 = compute_di(df, DI_TRIGGER_LENGTH)
        new_trigger = detect_crossover(plus_di_14, minus_di_14)

        di14_dir_key = f"di14_dir_{interval}"
        alerted_combo_key = f"alerted_combo_{interval}"

        if new_trigger is not None:
            state[di14_dir_key] = new_trigger

        last_di14_dir = state.get(di14_dir_key)
        combo = f"{last_di14_dir}_{trend_state}"

        aligned = last_di14_dir is not None and last_di14_dir == trend_state

        if aligned and state.get(alerted_combo_key) != combo:
            direction_fa = "صعودی" if trend_state == "bullish" else "نزولی"
            messages.append(
                f"طلا (XAU/USD) - تایم‌فریم {interval}\n"
                f"تأیید روند {direction_fa}: DI{DI_TRIGGER_LENGTH} قبلا {direction_fa} شده بود و حالا DI{DI_TREND_LENGTH} هم {direction_fa} است\n"
                f"قیمت: {latest_close:.2f}\n"
                f"زمان کندل: {latest_time}"
            )
            state[alerted_combo_key] = combo
        elif not aligned:
            state[alerted_combo_key] = None

        time.sleep(1)

    save_state(state)

    if messages:
        send_telegram_message("\n\n---\n\n".join(messages))
    else:
        print("No new signal.")


if __name__ == "__main__":
    main()
