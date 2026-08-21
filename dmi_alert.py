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
INTERVAL = "1min"

# تنظیمات WPR و Bollinger Bands
WPR_PERIOD = 200
WPR_MA_PERIOD = 3
WPR_MA_TYPE = "EMA"

BB_PERIOD = 50
BB_STD = 1.0
BB_MA_TYPE = "EMA"

OUTPUT_SIZE = 300

STATE_FILE = "state.json"

# حداقل تعداد کندل لازم تا WPR + Bollinger روی آن هر دو معتبر باشند
MIN_WARMUP = WPR_PERIOD + BB_PERIOD + 5


# =========================================================
# STATE & CONSTANTS
# =========================================================

STATE_NONE = "NONE"
STATE_ABOVE_UPPER = "ABOVE_UPPER"
STATE_INSIDE = "INSIDE"
STATE_BELOW_LOWER = "BELOW_LOWER"


def load_state():

    if not os.path.exists(STATE_FILE):

        return {
            "last_position": STATE_NONE
        }

    try:

        with open(STATE_FILE, "r", encoding="utf-8") as f:

            state = json.load(f)

        if "last_position" not in state:

            state["last_position"] = STATE_NONE

        return state

    except Exception:

        return {
            "last_position": STATE_NONE
        }


def save_state(state):

    with open(STATE_FILE, "w", encoding="utf-8") as f:

        json.dump(state, f, ensure_ascii=False, indent=2)


# =========================================================
# DATA & INDICATORS
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


def calculate_wpr_bb(df):

    # ۱. محاسبه Williams %R پایه
    highest_high = df["high"].rolling(WPR_PERIOD).max()
    lowest_low = df["low"].rolling(WPR_PERIOD).min()

    wpr_raw = 100 * (df["close"] - highest_high) / (highest_high - lowest_low)

    # ۲. نرمالسازی بازه به [-100, 100] و اعمال MA
    wpr_scaled = (wpr_raw + 50.0) * 2.0

    df["signal_line"] = get_ma(wpr_scaled, WPR_MA_PERIOD, WPR_MA_TYPE)

    # ۳. محاسبه باند بولینگر روی خط سیگنال
    basis = get_ma(df["signal_line"], BB_PERIOD, BB_MA_TYPE)

    dev = BB_STD * df["signal_line"].rolling(BB_PERIOD).std(ddof=1)

    df["bb_upper"] = basis + dev
    df["bb_lower"] = basis - dev

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

        send_telegram(
            f"⚠️ ربات WPRBB خطا داد:\n\n{error_text}"
        )

    except Exception:

        # اگر ارسال خطا هم شکست خورد، دیگر کاری نمی شود کرد
        pass


def build_alert_message(title, direction, signal_val, upper_val, lower_val, price, time_str):

    return f"""
🚨 WPRBB SIGNAL ALERT ({SYMBOL})

{title}
🧭 جهت: {direction}

💰 قیمت کلوز: {price:.2f}
📈 خط سیگنال: {signal_val:.2f}
⬆️ باند بالا: {upper_val:.2f}
⬇️ باند پایین: {lower_val:.2f}

━━━━━━━━━━━━━━
⏱ زمان: {time_str}
""".strip()


# =========================================================
# نگاشت تغییر وضعیت -> متن پیام
#
# کلید: (وضعیت قبلی, وضعیت جدید)
# =========================================================

TRANSITION_MESSAGES = {

    (STATE_INSIDE, STATE_BELOW_LOWER): (
        "🔴 برخورد از بالا به باند پایین (ادامه روند نزولی)",
        "Short 🔻"
    ),

    (STATE_BELOW_LOWER, STATE_INSIDE): (
        "🟢 برخورد از پایین به باند پایین (برگشت صعودی)",
        "Long 🟢"
    ),

    (STATE_INSIDE, STATE_ABOVE_UPPER): (
        "🟢 برخورد از پایین به باند بالا (ادامه روند صعودی)",
        "Long 🚀"
    ),

    (STATE_ABOVE_UPPER, STATE_INSIDE): (
        "🔴 برخورد از بالا به باند بالا (برگشت نزولی)",
        "Short 🔻"
    ),

    (STATE_BELOW_LOWER, STATE_ABOVE_UPPER): (
        "🟢 پرش مستقیم از باند پایین به باند بالا",
        "Long 🚀"
    ),

    (STATE_ABOVE_UPPER, STATE_BELOW_LOWER): (
        "🔴 پرش مستقیم از باند بالا به باند پایین",
        "Short 🔻"
    ),
}


# =========================================================
# MAIN LOGIC
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

    print("Calculating WPR + Bollinger...")

    df = calculate_wpr_bb(df)

    # -----------------------------------------------------
    # آخرین کندل (زنده / درحال شکل گیری)
    # -----------------------------------------------------

    curr = df.iloc[len(df) - 1]

    time_str = curr["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    print(f"Checking candle: {time_str}")

    signal_val = curr["signal_line"]
    upper_val = curr["bb_upper"]
    lower_val = curr["bb_lower"]

    # -----------------------------------------------------
    # اگر داده هنوز کامل نشده (NaN)، این اجرا را رد کن
    # بدون این‌که state را دستکاری کنیم
    # -----------------------------------------------------

    if pd.isna(signal_val) or pd.isna(upper_val) or pd.isna(lower_val):

        print("Indicators not ready yet (NaN). Skipping.")

        return

    # -----------------------------------------------------
    # تعیین وضعیت فعلی
    # -----------------------------------------------------

    if signal_val > upper_val:

        current_position = STATE_ABOVE_UPPER

    elif signal_val < lower_val:

        current_position = STATE_BELOW_LOWER

    else:

        current_position = STATE_INSIDE

    state = load_state()

    last_position = state.get("last_position", STATE_NONE)

    # -----------------------------------------------------
    # آلارم فقط روی تغییر وضعیت واقعی نسبت به اجرای قبلی
    # (نه هر بار که همان وضعیت تکرار شود، نه بر اساس زمان)
    # -----------------------------------------------------

    if (
        last_position != STATE_NONE
        and last_position != current_position
    ):

        transition_key = (last_position, current_position)

        transition = TRANSITION_MESSAGES.get(transition_key)

        if transition is not None:

            title, direction = transition

            message = build_alert_message(
                title,
                direction,
                signal_val,
                upper_val,
                lower_val,
                curr["close"],
                time_str
            )

            print(f"Sending Alert:\n{message}")

            send_telegram(message)

            print("Alert sent.")

    else:

        print(
            f"No transition. Position: {current_position}"
        )

    state["last_position"] = current_position

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
        
