import os
import json
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
BB_MA_TYPE = "SMA"   # نوع میانگین باند وسط (basis)

OUTPUT_SIZE = 200

STATE_FILE = "state_bb_cross.json"

MIN_WARMUP = BB_PERIOD + 10


# ---------------------------------------------------------
# موقعیت قیمت نسبت به باند وسط
# ---------------------------------------------------------

POS_ABOVE = "ABOVE"
POS_BELOW = "BELOW"
POS_NONE = "NONE"


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
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True
        )

    elif ma_type == "RMA":

        return series.ewm(alpha=1 / length, adjust=False).mean()

    else:  # SMA

        return series.rolling(length).mean()


# =========================================================
# Bollinger Bands (Period=30) — محاسبه باند وسط، بالا، پایین
# =========================================================

def calculate_bb(df):

    basis = get_ma(df["close"], BB_PERIOD, BB_MA_TYPE)

    # ta.stdev در Pine جمعیتی است (ddof=0)
    dev = BB_MULT * df["close"].rolling(BB_PERIOD).std(ddof=0)

    df["basis"] = basis
    df["upper"] = basis + dev
    df["lower"] = basis - dev

    return df


# =========================================================
# STATE
# =========================================================

DEFAULT_STATE = {
    "last_position": POS_NONE,
    "last_candle_time": None
}


def load_state():

    if not os.path.exists(STATE_FILE):

        return dict(DEFAULT_STATE)

    try:

        with open(STATE_FILE, "r", encoding="utf-8") as f:

            state = json.load(f)

        for key in DEFAULT_STATE:

            if key not in state:

                state[key] = DEFAULT_STATE[key]

        return state

    except Exception:

        return dict(DEFAULT_STATE)


def save_state(state):

    with open(STATE_FILE, "w", encoding="utf-8") as f:

        json.dump(state, f, ensure_ascii=False, indent=2)


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    requests.post(url, json=payload, timeout=20).raise_for_status()


def send_error_alert(error_text):

    try:

        send_telegram(
            f"⚠️ ربات BB Middle Cross خطا داد:\n\n{error_text}"
        )

    except Exception:

        pass


def build_message(direction, price, basis_val, upper_val, lower_val, time_str):

    return f"""
🚨 BB MIDDLE CROSS ALERT ({SYMBOL})

🧭 جهت: {direction}

💰 قیمت کلوز: {price:.2f}
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

        print(
            f"Not enough candles yet "
            f"({len(df)} < {MIN_WARMUP})."
        )

        return

    print("Calculating Bollinger Bands (30)...")

    df = calculate_bb(df)

    # -----------------------------------------------------
    # آخرین کندل (زنده / درحال شکل‌گیری)
    # -----------------------------------------------------

    curr = df.iloc[len(df) - 1]

    time_str = curr["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    state = load_state()

    print(f"Checking candle: {time_str} (live)")

    close_val = curr["close"]
    basis_val = curr["basis"]
    upper_val = curr["upper"]
    lower_val = curr["lower"]

    if pd.isna(basis_val) or pd.isna(upper_val) or pd.isna(lower_val):

        print("Indicators not ready yet (NaN). Skipping.")

        return

    # موقعیت فعلی قیمت نسبت به باند وسط
    if close_val > basis_val:
        current_position = POS_ABOVE
    elif close_val < basis_val:
        current_position = POS_BELOW
    else:
        current_position = state["last_position"]  # دقیقاً روی خط -> بدون تغییر

    last_position = state["last_position"]

    # -----------------------------------------------------
    # فقط وقتی موقعیت واقعاً عوض شده آلارم بده (کراس جدید)
    # اولین اجرا (last_position == NONE) فقط ثبت می‌شود، آلارم نمی‌دهد
    # -----------------------------------------------------

    if last_position != POS_NONE and current_position != last_position:

        if current_position == POS_ABOVE:

            message = build_message(
                "قیمت از باند وسط به سمت بالا رد شد 🔼",
                close_val,
                basis_val,
                upper_val,
                lower_val,
                time_str
            )

        else:

            message = build_message(
                "قیمت از باند وسط به سمت پایین رد شد 🔽",
                close_val,
                basis_val,
                upper_val,
                lower_val,
                time_str
            )

        print(f"Sending Alert:\n{message}")

        send_telegram(message)

    state["last_position"] = current_position
    state["last_candle_time"] = time_str

    save_state(state)

    print("Execution completed.")


# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(f"FATAL ERROR: {e}")

        send_error_alert(str(e))

        raise
