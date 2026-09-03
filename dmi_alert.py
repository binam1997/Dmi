import os
import requests
import pandas as pd
import numpy as np
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

BB_PERIOD = 30
BB_MULT = 2.0
BB_MA_TYPE = "SMA"

OUTPUT_SIZE = 200

MIN_WARMUP = BB_PERIOD + 10


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
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def get_ma(series, length, ma_type):

    if ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    elif ma_type == "RMA":
        return series.ewm(alpha=1 / length, adjust=False).mean()
    else:  # SMA
        return series.rolling(length).mean()


def calculate_bb(df):

    basis = get_ma(df["close"], BB_PERIOD, BB_MA_TYPE)
    dev = BB_MULT * df["close"].rolling(BB_PERIOD).std(ddof=0)

    df["basis"] = basis
    df["upper"] = basis + dev
    df["lower"] = basis - dev

    return df


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload, timeout=20).raise_for_status()


def send_error_alert(error_text):

    try:
        send_telegram(f"⚠️ ربات BB Middle Touch خطا داد:\n\n{error_text}")
    except Exception:
        pass


def build_message(price, basis_val, upper_val, lower_val, high_val, low_val, time_str):

    return f"""
🚨 برخورد قیمت با باند وسط بولینجر ({SYMBOL})

💰 قیمت کلوز: {price:.2f}
📈 های کندل: {high_val:.2f}
📉 لو کندل: {low_val:.2f}

📊 باند وسط: {basis_val:.2f}
⬆️ باند بالا: {upper_val:.2f}
⬇️ باند پایین: {lower_val:.2f}

━━━━━━━━━━━━━━
⏱ زمان: {time_str}
""".strip()


# =========================================================
# MAIN
# =========================================================

def main():

    print("Getting market data...")
    df = get_data()

    if len(df) < MIN_WARMUP:
        print(f"Not enough candles yet ({len(df)} < {MIN_WARMUP}).")
        return

    print("Calculating Bollinger Bands (30)...")
    df = calculate_bb(df)

    curr = df.iloc[-1]

    close_val = curr["close"]
    high_val = curr["high"]
    low_val = curr["low"]
    basis_val = curr["basis"]
    upper_val = curr["upper"]
    lower_val = curr["lower"]
    time_str = curr["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    if pd.isna(basis_val) or pd.isna(upper_val) or pd.isna(lower_val):
        print("Indicators not ready yet (NaN). Skipping.")
        return

    print(f"Checking candle: {time_str} (live)")
    print(f"High: {high_val} | Low: {low_val} | Basis: {basis_val}")

    # برخورد = بازه high/low کندل، باند وسط رو پوشش داده باشه
    touched_middle = low_val <= basis_val <= high_val

    if touched_middle:

        message = build_message(
            close_val, basis_val, upper_val, lower_val,
            high_val, low_val, time_str
        )

        print(f"Sending Alert:\n{message}")
        send_telegram(message)

    else:
        print("No touch on middle band this run.")

    print("Execution completed.")


# =========================================================

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        send_error_alert(str(e))
        raise
