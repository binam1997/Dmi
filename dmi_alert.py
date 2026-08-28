import os
import json
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo


IRAN_TZ = ZoneInfo("Asia/Tehran")


# =========================================================
# SETTINGS (طبق تنظیمات WPRBB [Loxx])
# =========================================================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "5min"

WPR_PERIOD = 200
WPR_MA_PERIOD = 10
WPR_MA_TYPE = "EMA"

BB_PERIOD = 100
BB_MULT = 1.0
BB_MA_TYPE = "EMA"

FLAT_THRESHOLD = 0.5

OUTPUT_SIZE = 400

STATE_FILE = "state.json"

# حداقل تعداد کندل لازم تا WPR + BB روی آن هر دو معتبر باشند
MIN_WARMUP = WPR_PERIOD + BB_PERIOD + 10


# ---------------------------------------------------------
# رنگ خط WPR
# ---------------------------------------------------------

COLOR_GREEN = "green"
COLOR_RED = "red"
COLOR_YELLOW = "yellow"

# ---------------------------------------------------------
# وضعیت "آماده‌باش" هر سناریو
# ---------------------------------------------------------

ARM_NONE = "NONE"
ARM_ARMED = "ARMED"


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
# WPRBB [Loxx] — دقیقاً طبق کد Pine
# =========================================================

def calculate_wprbb(df):

    highest_high = df["high"].rolling(WPR_PERIOD).max()
    lowest_low = df["low"].rolling(WPR_PERIOD).min()

    wpr_raw = 100 * (df["close"] - highest_high) / (highest_high - lowest_low)

    out_raw = (wpr_raw + 50.0) * 2.0

    df["out"] = get_ma(out_raw, WPR_MA_PERIOD, WPR_MA_TYPE)

    basis = get_ma(df["out"], BB_PERIOD, BB_MA_TYPE)

    # ta.stdev در Pine جمعیتی است (ddof=0)
    dev = BB_MULT * df["out"].rolling(BB_PERIOD).std(ddof=0)

    df["upper"] = basis + dev
    df["lower"] = basis - dev

    out_diff = df["out"] - df["out"].shift(1)

    df["color"] = np.where(
        out_diff > FLAT_THRESHOLD,
        COLOR_GREEN,
        np.where(
            out_diff < -FLAT_THRESHOLD,
            COLOR_RED,
            COLOR_YELLOW
        )
    )

    return df


# =========================================================
# STATE
# =========================================================

DEFAULT_STATE = {
    "bearish_arm": ARM_NONE,
    "bullish_arm": ARM_NONE,
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
            f"⚠️ ربات WPRBB Reversal خطا داد:\n\n{error_text}"
        )

    except Exception:

        pass


def build_message(title, direction, price, out_val, upper_val, lower_val, time_str):

    return f"""
🚨 WPRBB REVERSAL ALERT ({SYMBOL})

{title}
🧭 جهت: {direction}

💰 قیمت کلوز: {price:.2f}
📈 خط WPR: {out_val:.2f}
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

    print("Calculating WPRBB...")

    df = calculate_wprbb(df)

    # -----------------------------------------------------
    # آخرین کندل (زنده / درحال شکل‌گیری) — طبق درخواست شما
    # هر بار اجرا، همین کندل با آخرین قیمت لحظه‌ای چک می‌شود
    # -----------------------------------------------------

    curr = df.iloc[len(df) - 1]

    time_str = curr["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    state = load_state()

    print(f"Checking candle: {time_str} (live)")

    out_val = curr["out"]
    upper_val = curr["upper"]
    lower_val = curr["lower"]
    color = curr["color"]

    if pd.isna(out_val) or pd.isna(upper_val) or pd.isna(lower_val):

        print("Indicators not ready yet (NaN). Skipping.")

        return

    above_upper = out_val > upper_val
    below_lower = out_val < lower_val

    bearish_arm = state["bearish_arm"]
    bullish_arm = state["bullish_arm"]

    # -----------------------------------------------------
    # سناریوی نزولی: عبور از باند بالا + رنگ سبز -> آماده‌باش
    # بعدش هر تغییر رنگ (به زرد یا قرمز) -> آلارم
    # -----------------------------------------------------

    if bearish_arm == ARM_NONE:

        if above_upper and color == COLOR_GREEN:

            bearish_arm = ARM_ARMED

            print("Bearish scenario armed (above upper + green).")

    elif bearish_arm == ARM_ARMED:

        if color != COLOR_GREEN:

            message = build_message(
                "🔴 برگشت نزولی — WPR از سبز خارج شد",
                "Short 🔻",
                curr["close"],
                out_val,
                upper_val,
                lower_val,
                time_str
            )

            print(f"Sending Alert:\n{message}")

            send_telegram(message)

            bearish_arm = ARM_NONE

    # -----------------------------------------------------
    # سناریوی صعودی: عبور از باند پایین + رنگ قرمز -> آماده‌باش
    # بعدش هر تغییر رنگ (به زرد یا سبز) -> آلارم
    # -----------------------------------------------------

    if bullish_arm == ARM_NONE:

        if below_lower and color == COLOR_RED:

            bullish_arm = ARM_ARMED

            print("Bullish scenario armed (below lower + red).")

    elif bullish_arm == ARM_ARMED:

        if color != COLOR_RED:

            message = build_message(
                "🟢 برگشت صعودی — WPR از قرمز خارج شد",
                "Long 🚀",
                curr["close"],
                out_val,
                upper_val,
                lower_val,
                time_str
            )

            print(f"Sending Alert:\n{message}")

            send_telegram(message)

            bullish_arm = ARM_NONE

    state["bearish_arm"] = bearish_arm
    state["bullish_arm"] = bullish_arm
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
        
