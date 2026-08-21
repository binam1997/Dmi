import os
import json
import requests
import pandas as pd
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "1min"


# ---------------------------------------------------------
# Strategy 1
# Bollinger 50
# ---------------------------------------------------------

BB50_LENGTH = 50
BB50_STD = 2


# ---------------------------------------------------------
# Strategy 2
# Bollinger 100
# ---------------------------------------------------------

BB100_LENGTH = 100
BB100_STD = 2


# ---------------------------------------------------------
# BBW
# ---------------------------------------------------------

BBW_LENGTH = 20
BBW_STD = 2


# ---------------------------------------------------------
# ADX
# ---------------------------------------------------------

ADX_LENGTH = 14


# ---------------------------------------------------------

OUTPUT_SIZE = 200

STATE_FILE = "state.json"

# حداقل تعداد کندل لازم برای اینکه همه اندیکاتورها معتبر باشند
MIN_WARMUP = max(
    BB100_LENGTH,
    BBW_LENGTH,
    ADX_LENGTH
) + 5


# ---------------------------------------------------------
# وضعیت های ممکن برای هر باند
# ---------------------------------------------------------

STATUS_NONE = "none"
STATUS_RIDE = "ride"
STATUS_REJECT = "reject"


# =========================================================
# GET MARKET DATA
# =========================================================

def get_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVEDATA_API_KEY,
        "timezone": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise Exception(
            f"TwelveData Error: {data}"
        )

    df = pd.DataFrame(data["values"])

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df


# =========================================================
# EMA
# =========================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False
    ).mean()


# =========================================================
# BOLLINGER 50 EMA
# =========================================================

def calculate_bb50(df):

    basis = ema(
        df["close"],
        BB50_LENGTH
    )

    std = df["close"].rolling(
        BB50_LENGTH
    ).std(ddof=1)

    df["bb50_mid"] = basis

    df["bb50_upper"] = (
        basis + BB50_STD * std
    )

    df["bb50_lower"] = (
        basis - BB50_STD * std
    )

    return df


# =========================================================
# BOLLINGER 100 EMA
# =========================================================

def calculate_bb100(df):

    basis = ema(
        df["close"],
        BB100_LENGTH
    )

    std = df["close"].rolling(
        BB100_LENGTH
    ).std(ddof=1)

    df["bb100_mid"] = basis

    df["bb100_upper"] = (
        basis + BB100_STD * std
    )

    df["bb100_lower"] = (
        basis - BB100_STD * std
    )

    return df


# =========================================================
# BBW 20
# =========================================================

def calculate_bbw(df):

    basis = df["close"].rolling(
        BBW_LENGTH
    ).mean()

    std = df["close"].rolling(
        BBW_LENGTH
    ).std(ddof=1)

    upper = (
        basis + BBW_STD * std
    )

    lower = (
        basis - BBW_STD * std
    )

    df["bbw20"] = (
        (upper - lower)
        / basis
        * 100
    )

    return df


# =========================================================
# ADX 14 + DI
# =========================================================

def calculate_adx(df):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    # -----------------------------------------------------
    # True Range
    # -----------------------------------------------------

    tr1 = high - low

    tr2 = (
        high - prev_close
    ).abs()

    tr3 = (
        low - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # -----------------------------------------------------
    # Directional Movement
    # -----------------------------------------------------

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move)
        & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move)
        & (down_move > 0),
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

    # -----------------------------------------------------
    # Wilder smoothing
    # -----------------------------------------------------

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

    di_sum = plus_di + minus_di

    dx = np.where(
        di_sum == 0,
        0,
        100 * (plus_di - minus_di).abs() / di_sum
    )

    dx = pd.Series(
        dx,
        index=df.index
    )

    adx = dx.ewm(
        alpha=1 / ADX_LENGTH,
        adjust=False
    ).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    return df


# =========================================================
# STATE
#
# ساختار:
#
# {
#   "last_candle_time": "...",
#   "band_status": {
#       "bb50_upper_status": "none" | "ride" | "reject",
#       "bb50_lower_status": "none" | "ride" | "reject",
#       "bb100_upper_status": "none" | "ride" | "reject",
#       "bb100_lower_status": "none" | "ride" | "reject"
#   }
# }
#
# آلارم فقط وقتی ارسال می شود که وضعیت یک باند
# نسبت به کندل قبلی عوض شده باشد (نه هر بار که همان
# وضعیت تکرار شود).
# =========================================================

DEFAULT_BAND_STATUS = {
    "bb50_upper_status": STATUS_NONE,
    "bb50_lower_status": STATUS_NONE,
    "bb100_upper_status": STATUS_NONE,
    "bb100_lower_status": STATUS_NONE
}


def load_state():

    if not os.path.exists(
        STATE_FILE
    ):

        return {
            "last_candle_time": None,
            "band_status": dict(
                DEFAULT_BAND_STATUS
            )
        }

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        if "band_status" not in state:

            state["band_status"] = dict(
                DEFAULT_BAND_STATUS
            )

        for key in DEFAULT_BAND_STATUS:

            if key not in state["band_status"]:

                state["band_status"][key] = STATUS_NONE

        if "last_candle_time" not in state:

            state["last_candle_time"] = None

        return state

    except Exception:

        return {
            "last_candle_time": None,
            "band_status": dict(
                DEFAULT_BAND_STATUS
            )
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


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    url = (
        "https://api.telegram.org/"
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


def send_error_alert(error_text):

    try:

        send_telegram(
            f"⚠️ ربات XAU/USD خطا داد:\n\n{error_text}"
        )

    except Exception:

        # اگر ارسال خطا هم شکست خورد، دیگر کاری نمی شود کرد
        pass


# =========================================================
# BBW TREND
#
# سه کندل قبل از سیگنال را بررسی می کنیم و باید
# به صورت پیوسته صعودی باشد: old < mid < new
# =========================================================

def bbw_is_rising(df, index):

    if index < 3:
        return False

    bbw_old = df.iloc[
        index - 3
    ]["bbw20"]

    bbw_mid = df.iloc[
        index - 2
    ]["bbw20"]

    bbw_new = df.iloc[
        index - 1
    ]["bbw20"]

    if (
        pd.isna(bbw_old)
        or pd.isna(bbw_mid)
        or pd.isna(bbw_new)
    ):

        return False

    return (
        bbw_old < bbw_mid
        and bbw_mid < bbw_new
    )


# =========================================================
# DI STATUS TEXT
# =========================================================

def di_status_text(plus_di, minus_di):

    if plus_di > minus_di:

        return "DI+ بالاتر از DI- است 🟢"

    elif minus_di > plus_di:

        return "DI- بالاتر از DI+ است 🔴"

    return "DI+ و DI- برابر هستند ⚪"


# =========================================================
# طبقه بندی برخورد یک باند
#
# none    -> اصلا لمس نشده
# ride    -> لمس شده و بادی کندل هم بیرون از باند کلوز کرده
#            (احتمال ادامه ی روند، قیمت دارد "سوار" باند می شود)
# reject  -> فقط فتیله لمس کرده، کلوز داخل باند برگشته
#            (احتمال برگشت روند)
# =========================================================

def classify_band(high, low, close, band_value, side):

    if side == "upper":

        touched = high >= band_value
        closed_beyond = close > band_value

    else:

        touched = low <= band_value
        closed_beyond = close < band_value

    if not touched:
        return STATUS_NONE

    if closed_beyond:
        return STATUS_RIDE

    return STATUS_REJECT


# =========================================================
# STRATEGY CHECK (مشترک برای BB50 و BB100)
# =========================================================

def check_band_touch(df, index, upper_col, lower_col, label):

    candle = df.iloc[index]

    upper_status = classify_band(
        candle["high"],
        candle["low"],
        candle["close"],
        candle[upper_col],
        "upper"
    )

    lower_status = classify_band(
        candle["high"],
        candle["low"],
        candle["close"],
        candle[lower_col],
        "lower"
    )

    adx = candle["adx"]
    plus_di = candle["plus_di"]
    minus_di = candle["minus_di"]

    bbw_ok = bbw_is_rising(
        df,
        index
    )

    indicators_ok = (
        bbw_ok
        and not pd.isna(adx)
        and not pd.isna(plus_di)
        and not pd.isna(minus_di)
    )

    return {
        "type": label,
        "upper_status": upper_status,
        "lower_status": lower_status,
        "indicators_ok": indicators_ok,
        "price": candle["close"],
        "time": str(candle["datetime"]),
        "bbw_old": df.iloc[index - 3]["bbw20"] if index >= 3 else np.nan,
        "bbw_mid": df.iloc[index - 2]["bbw20"] if index >= 2 else np.nan,
        "bbw_new": df.iloc[index - 1]["bbw20"] if index >= 1 else np.nan,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "di_status": di_status_text(plus_di, minus_di)
    }


# =========================================================
# متن پیام بر اساس side ("upper"/"lower") و status ("ride"/"reject")
# =========================================================

def band_headline(side, status):

    if side == "upper" and status == STATUS_RIDE:

        return (
            "🟢🚀 سوار باند بالا شد",
            "احتمال ادامه ی روند صعودی"
        )

    if side == "upper" and status == STATUS_REJECT:

        return (
            "🔴↩️ رد شد از باند بالا",
            "احتمال برگشت روند"
        )

    if side == "lower" and status == STATUS_RIDE:

        return (
            "🔴🚀 سوار باند پایین شد",
            "احتمال ادامه ی روند نزولی"
        )

    # side == "lower" and status == STATUS_REJECT

    return (
        "🟢↩️ رد شد از باند پایین",
        "احتمال برگشت روند"
    )


def build_message(signal, side, status):

    band_name = (
        "BB50 EMA"
        if signal["type"] == "BB50 TOUCH"
        else "BB100 EMA"
    )

    title, note = band_headline(
        side,
        status
    )

    return f"""
🚨 XAU/USD ALERT

{title}
{note}

📊 Strategy:
{signal["type"]}

💰 قیمت:
{signal["price"]:.3f}

━━━━━━━━━━━━━━

📈 Bollinger Band

{band_name}

📊 BBW20 — سه کندل قبل:

-3 : {signal["bbw_old"]:.4f}
-2 : {signal["bbw_mid"]:.4f}
-1 : {signal["bbw_new"]:.4f}

✅ BBW صعودی است

━━━━━━━━━━━━━━

📐 ADX 14:
{signal["adx"]:.2f}

DI+:
{signal["plus_di"]:.2f}

DI-:
{signal["minus_di"]:.2f}

{signal["di_status"]}

━━━━━━━━━━━━━━

⏱ {signal["time"]}
""".strip()


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Getting market data..."
    )

    df = get_data()

    if len(df) < MIN_WARMUP:

        print(
            f"Not enough candles yet "
            f"({len(df)} < {MIN_WARMUP})."
        )

        return

    print(
        "Calculating indicators..."
    )

    df = calculate_bb50(df)
    df = calculate_bb100(df)
    df = calculate_bbw(df)
    df = calculate_adx(df)

    # -----------------------------------------------------
    # آخرین کندل بسته شده
    # (فرض: آخرین ردیف = کندل درحال شکل گیری)
    # -----------------------------------------------------

    index = len(df) - 2

    candle_time = str(
        df.iloc[index]["datetime"]
    )

    print(
        f"Checking candle: {candle_time}"
    )

    state = load_state()

    # -----------------------------------------------------
    # اگر این کندل قبلاً پردازش شده، دوباره پردازش نکن
    # -----------------------------------------------------

    if state["last_candle_time"] == candle_time:

        print(
            "This candle was already processed. Skipping."
        )

        return

    band_status = state["band_status"]

    strategies = [
        (
            "bb50_upper_status",
            "bb50_lower_status",
            "bb50_upper",
            "bb50_lower",
            "BB50 TOUCH"
        ),
        (
            "bb100_upper_status",
            "bb100_lower_status",
            "bb100_upper",
            "bb100_lower",
            "BB100 TOUCH"
        )
    ]

    messages_to_send = []

    for (
        upper_status_key,
        lower_status_key,
        upper_col,
        lower_col,
        label
    ) in strategies:

        result = check_band_touch(
            df,
            index,
            upper_col,
            lower_col,
            label
        )

        old_upper_status = band_status[upper_status_key]
        old_lower_status = band_status[lower_status_key]

        new_upper_status = result["upper_status"]
        new_lower_status = result["lower_status"]

        # ---------------------------------------------------
        # آلارم فقط روی تغییر وضعیت، و فقط اگر وضعیت جدید
        # none نباشد (خروج از باند بی صدا ثبت می شود)
        # ---------------------------------------------------

        if (
            new_upper_status != old_upper_status
            and new_upper_status != STATUS_NONE
            and result["indicators_ok"]
        ):

            messages_to_send.append(
                build_message(
                    result,
                    "upper",
                    new_upper_status
                )
            )

        if (
            new_lower_status != old_lower_status
            and new_lower_status != STATUS_NONE
            and result["indicators_ok"]
        ):

            messages_to_send.append(
                build_message(
                    result,
                    "lower",
                    new_lower_status
                )
            )

        # ---------------------------------------------------
        # پرچم وضعیت را همیشه به‌روز کن، چه آلارم بفرستیم چه نه
        # ---------------------------------------------------

        band_status[upper_status_key] = new_upper_status
        band_status[lower_status_key] = new_lower_status

    # -----------------------------------------------------
    # ارسال
    # -----------------------------------------------------

    if not messages_to_send:

        print(
            "No new signal."
        )

    for message in messages_to_send:

        print(
            message
        )

        send_telegram(
            message
        )

        print(
            "Alert sent."
        )

    state["last_candle_time"] = candle_time
    state["band_status"] = band_status

    save_state(
        state
    )

    print(
        "State saved."
    )


# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            f"FATAL ERROR: {e}"
        )

        send_error_alert(
            str(e)
        )

        raise
