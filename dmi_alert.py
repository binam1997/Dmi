import os
import json
import requests
import pandas as pd
import numpy as np

# =========================
# SETTINGS
# =========================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "1min"

# Bollinger اصلی برای سیگنال
BB_LENGTH = 50
BB_STD = 2

# BBW مطابق تنظیمات تصویر
BBW_LENGTH = 20
BBW_STD = 2

# ADX
ADX_LENGTH = 14

# تعداد کندل دریافت شده
OUTPUT_SIZE = 150

# فایل جلوگیری از اسپم
STATE_FILE = "alert_state.json"


# =========================
# GET MARKET DATA
# =========================

def get_data():
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
        "timezone": "UTC"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise Exception(f"TwelveData Error: {data}")

    df = pd.DataFrame(data["values"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"])

    # از قدیمی به جدید
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# =========================
# EMA
# =========================

def ema(series, length):
    return series.ewm(
        span=length,
        adjust=False
    ).mean()


# =========================
# BOLLINGER 50 EMA
# =========================

def calculate_bollinger(df):

    basis = ema(df["close"], BB_LENGTH)

    std = df["close"].rolling(BB_LENGTH).std(ddof=0)

    upper = basis + BB_STD * std
    lower = basis - BB_STD * std

    df["bb_basis"] = basis
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    return df


# =========================
# BBW
# =========================

def calculate_bbw(df):

    # BBW استاندارد:
    # (Upper - Lower) / Middle * 100

    basis = df["close"].rolling(BBW_LENGTH).mean()

    std = df["close"].rolling(BBW_LENGTH).std(ddof=0)

    upper = basis + BBW_STD * std
    lower = basis - BBW_STD * std

    df["bbw"] = ((upper - lower) / basis) * 100

    return df


# =========================
# ADX 14 + DI
# =========================

def calculate_adx(df):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    )

    # Wilder smoothing
    atr = tr.ewm(
        alpha=1 / ADX_LENGTH,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / ADX_LENGTH,
            adjust=False
        ).mean()
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    )

    adx = dx.ewm(
        alpha=1 / ADX_LENGTH,
        adjust=False
    ).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    return df


# =========================
# STATE / ANTI SPAM
# =========================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {
            "last_alert_key": None
        }

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception:
        return {
            "last_alert_key": None
        }


def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# TELEGRAM
# =========================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    response = requests.post(
        url,
        json=payload,
        timeout=20
    )

    response.raise_for_status()


# =========================
# FIND SIGNAL
# =========================

def check_signal(df):

    # آخرین کندل بسته شده
    # چون داده TwelveData ممکن است کندل جاری را هم بدهد،
    # آخرین کندل را کنار می‌گذاریم و کندل قبلی را بررسی می‌کنیم.

    if len(df) < 60:
        return None

    index = len(df) - 2

    candle = df.iloc[index]

    # =========================
    # STEP 1
    # BOLLINGER TOUCH
    # =========================

    upper_touch = (
        candle["high"] >= candle["bb_upper"]
    )

    lower_touch = (
        candle["low"] <= candle["bb_lower"]
    )

    if not upper_touch and not lower_touch:
        return None

    # =========================
    # STEP 2
    # THREE CANDLES BEFORE TOUCH
    # =========================

    if index < 3:
        return None

    c3 = df.iloc[index - 3]
    c2 = df.iloc[index - 2]
    c1 = df.iloc[index - 1]

    bbw_3 = c3["bbw"]
    bbw_2 = c2["bbw"]
    bbw_1 = c1["bbw"]

    if pd.isna(bbw_3) or pd.isna(bbw_1):
        return None

    # =========================
    # BBW TREND
    # =========================

    # مثال:
    # 19 -> 18.5 -> 20
    #
    # چون ابتدا 19 بوده
    # و انتها 20 شده
    # روند را صعودی در نظر می گیریم.

    bbw_rising = bbw_1 > bbw_3

    if not bbw_rising:
        return None

    # =========================
    # ADX / DI
    # =========================

    adx = candle["adx"]
    plus_di = candle["plus_di"]
    minus_di = candle["minus_di"]

    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
        return None

    if plus_di > minus_di:
        di_status = "DI+ بالاتر از DI- است 🟢"
        direction = "صعودی"
    elif minus_di > plus_di:
        di_status = "DI- بالاتر از DI+ است 🔴"
        direction = "نزولی"
    else:
        di_status = "DI+ و DI- برابر هستند ⚪"
        direction = "خنثی"

    # =========================
    # TOUCH TYPE
    # =========================

    if upper_touch and lower_touch:
        touch_type = "همزمان Upper و Lower"
        emoji = "⚠️"

    elif upper_touch:
        touch_type = "Upper Band"
        emoji = "🔴"

    else:
        touch_type = "Lower Band"
        emoji = "🟢"

    # =========================
    # ALERT KEY
    # =========================

    candle_time = str(
        candle["datetime"]
    )

    alert_key = (
        f"{candle_time}_{touch_type}"
    )

    return {
        "alert_key": alert_key,
        "time": candle_time,
        "touch_type": touch_type,
        "emoji": emoji,
        "price": candle["close"],
        "upper": candle["bb_upper"],
        "lower": candle["bb_lower"],

        "bbw_3": bbw_3,
        "bbw_2": bbw_2,
        "bbw_1": bbw_1,

        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,

        "di_status": di_status,
        "direction": direction
    }


# =========================
# MAIN
# =========================

def main():

    print("Getting market data...")

    df = get_data()

    print("Calculating indicators...")

    df = calculate_bollinger(df)
    df = calculate_bbw(df)
    df = calculate_adx(df)

    signal = check_signal(df)

    if signal is None:

        print("No valid signal.")
        return

    state = load_state()

    # جلوگیری از ارسال دوباره همان سیگنال
    if state.get("last_alert_key") == signal["alert_key"]:

        print(
            "Signal already sent. "
            "Skipping duplicate alert."
        )

        return

    # =========================
    # TELEGRAM MESSAGE
    # =========================

    message = f"""
🚨 XAU/USD ALERT

{signal["emoji"]} برخورد با:
{signal["touch_type"]}

💰 قیمت:
{signal["price"]:.3f}

📊 Bollinger 50 EMA
Upper: {signal["upper"]:.3f}
Lower: {signal["lower"]:.3f}

📈 BBW — 3 کندل قبل:
-3 : {signal["bbw_3"]:.4f}
-2 : {signal["bbw_2"]:.4f}
-1 : {signal["bbw_1"]:.4f}

✅ BBW صعودی است

━━━━━━━━━━━━━━

📐 ADX 14:
{signal["adx"]:.2f}

DI+:
{signal["plus_di"]:.2f}

DI-:
{signal["minus_di"]:.2f}

{signal["di_status"]}

📌 وضعیت DI:
{signal["direction"]}

⏱ زمان:
{signal["time"]}
""".strip()

    print(message)

    # اول تلگرام
    send_telegram(message)

    # فقط بعد از موفقیت تلگرام state ذخیره می‌شود
    state["last_alert_key"] = signal["alert_key"]

    save_state(state)

    print("Alert sent successfully.")


if __name__ == "__main__":
    main()
