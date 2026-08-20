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
# Bollinger 25
# ---------------------------------------------------------

BB25_LENGTH = 25
BB25_STD = 2

# تعداد کندل برای پیدا کردن کمترین عرض
SQUEEZE_LOOKBACK = 20

# تعداد کندل افزایش بعد از کمترین عرض
EXPANSION_CONFIRM_CANDLES = 2

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
    ).std(ddof=0)

    df["bb50_mid"] = basis

    df["bb50_upper"] = (
        basis + BB50_STD * std
    )

    df["bb50_lower"] = (
        basis - BB50_STD * std
    )

    return df


# =========================================================
# BOLLINGER 25 EMA
# =========================================================

def calculate_bb25(df):

    basis = ema(
        df["close"],
        BB25_LENGTH
    )

    std = df["close"].rolling(
        BB25_LENGTH
    ).std(ddof=0)

    df["bb25_mid"] = basis

    df["bb25_upper"] = (
        basis + BB25_STD * std
    )

    df["bb25_lower"] = (
        basis - BB25_STD * std
    )

    # عرض باند
    df["bb25_width"] = (
        df["bb25_upper"]
        - df["bb25_lower"]
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
    ).std(ddof=0)

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

    # True Range
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

    # Directional Movement

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


# =========================================================
# STATE
# =========================================================

def load_state():

    if not os.path.exists(
        STATE_FILE
    ):

        return {
            "last_alert_keys": []
        }

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        if "last_alert_keys" not in state:
            state["last_alert_keys"] = []

        return state

    except Exception:

        return {
            "last_alert_keys": []
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


# =========================================================
# BBW TREND
# =========================================================
#
# سه کندل قبل از سیگنال را بررسی می کنیم.
#
# مثال:
#
# 19 -> 18.5 -> 20
#
# چون 20 > 19 است،
# روند BBW صعودی محسوب می شود.
#
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

    return bbw_new > bbw_old


# =========================================================
# STRATEGY 1
#
# BB50 TOUCH
# +
# BBW20 RISING
# =========================================================

def check_bb50_touch(df, index):

    if index < 5:
        return None

    candle = df.iloc[index]

    upper_touch = (
        candle["high"]
        >= candle["bb50_upper"]
    )

    lower_touch = (
        candle["low"]
        <= candle["bb50_lower"]
    )

    if not upper_touch and not lower_touch:
        return None

    # BBW باید قبل از برخورد صعودی باشد

    if not bbw_is_rising(
        df,
        index
    ):

        return None

    adx = candle["adx"]
    plus_di = candle["plus_di"]
    minus_di = candle["minus_di"]

    if (
        pd.isna(adx)
        or pd.isna(plus_di)
        or pd.isna(minus_di)
    ):

        return None

    if plus_di > minus_di:

        di_status = (
            "DI+ بالاتر از DI- است 🟢"
        )

    elif minus_di > plus_di:

        di_status = (
            "DI- بالاتر از DI+ است 🔴"
        )

    else:

        di_status = (
            "DI+ و DI- برابر هستند ⚪"
        )

    if upper_touch and lower_touch:

        touch = "Upper + Lower"

        emoji = "⚠️"

    elif upper_touch:

        touch = "Upper Band"

        emoji = "🔴"

    else:

        touch = "Lower Band"

        emoji = "🟢"

    candle_time = str(
        candle["datetime"]
    )

    alert_key = (
        f"BB50_{candle_time}_{touch}"
    )

    return {

        "type": "BB50 TOUCH",

        "alert_key": alert_key,

        "time": candle_time,

        "touch": touch,

        "emoji": emoji,

        "price": candle["close"],

        "bbw_old": df.iloc[
            index - 3
        ]["bbw20"],

        "bbw_mid": df.iloc[
            index - 2
        ]["bbw20"],

        "bbw_new": df.iloc[
            index - 1
        ]["bbw20"],

        "adx": adx,

        "plus_di": plus_di,

        "minus_di": minus_di,

        "di_status": di_status
    }


# =========================================================
# STRATEGY 2
#
# BB25 SQUEEZE -> EXPANSION
#
# 1. 20 کندل قبلی
# 2. پیدا کردن کمترین Width
# 3. بعد از Minimum حداقل 2 کندل افزایش
# 4. Price از Midline عبور کرده
# 5. DI جهت حرکت را تایید می کند
# 6. BBW20 صعودی است
#
# =========================================================

def check_bb25_expansion(
    df,
    index
):

    # حداقل داده مورد نیاز

    required = (
        SQUEEZE_LOOKBACK
        + EXPANSION_CONFIRM_CANDLES
        + 5
    )

    if index < required:
        return None

    candle = df.iloc[index]

    previous = df.iloc[
        index - 1
    ]

    # -----------------------------------------------------
    # 1. بیست کندل قبل از کندل فعلی
    # -----------------------------------------------------

    start = (
        index - SQUEEZE_LOOKBACK
    )

    end = index

    widths = df.iloc[
        start:end
    ]["bb25_width"].copy()

    if widths.isna().any():
        return None

    # -----------------------------------------------------
    # 2. کمترین عرض
    # -----------------------------------------------------

    min_position = widths.idxmin()

    min_width = widths.loc[
        min_position
    ]

    # موقعیت کمترین عرض نسبت به بازه
    min_index = df.index.get_loc(
        min_position
    )

    # -----------------------------------------------------
    # 3. بررسی Expansion
    # -----------------------------------------------------

    # باید بعد از Minimum حداقل
    # 2 کندل افزایش داشته باشیم.

    expansion_count = 0

    current_position = index - 1

    while current_position > min_position:

        current_width = df.iloc[
            current_position
        ]["bb25_width"]

        previous_width = df.iloc[
            current_position - 1
        ]["bb25_width"]

        if current_width > previous_width:

            expansion_count += 1

            current_position -= 1

        else:

            break

    if expansion_count < EXPANSION_CONFIRM_CANDLES:

        return None

    # اطمینان از اینکه Expansion
    # واقعاً بعد از Minimum اتفاق افتاده

    if min_position >= index:

        return None

    # -----------------------------------------------------
    # 4. Cross Midline
    # -----------------------------------------------------

    bullish_cross = (
        previous["close"]
        <= previous["bb25_mid"]
        and
        candle["close"]
        > candle["bb25_mid"]
    )

    bearish_cross = (
        previous["close"]
        >= previous["bb25_mid"]
        and
        candle["close"]
        < candle["bb25_mid"]
    )

    if not bullish_cross and not bearish_cross:
        return None

    # -----------------------------------------------------
    # 5. ADX / DI
    # -----------------------------------------------------

    adx = candle["adx"]

    plus_di = candle["plus_di"]

    minus_di = candle["minus_di"]

    if (
        pd.isna(adx)
        or pd.isna(plus_di)
        or pd.isna(minus_di)
    ):

        return None

    # -----------------------------------------------------
    # جهت سیگنال
    # -----------------------------------------------------

    if bullish_cross:

        if plus_di <= minus_di:
            return None

        direction = "صعودی 🟢"

        di_status = (
            "DI+ بالاتر از DI- است"
        )

    else:

        if minus_di <= plus_di:
            return None

        direction = "نزولی 🔴"

        di_status = (
            "DI- بالاتر از DI+ است"
        )

    # -----------------------------------------------------
    # 6. BBW20 Rising
    # -----------------------------------------------------

    if not bbw_is_rising(
        df,
        index
    ):

        return None

    bbw_old = df.iloc[
        index - 3
    ]["bbw20"]

    bbw_mid = df.iloc[
        index - 2
    ]["bbw20"]

    bbw_new = df.iloc[
        index - 1
    ]["bbw20"]

    # -----------------------------------------------------
    # Alert Key
    # -----------------------------------------------------

    candle_time = str(
        candle["datetime"]
    )

    alert_key = (
        f"BB25_EXPANSION_"
        f"{candle_time}_"
        f"{direction}"
    )

    return {

        "type": "BB25 EXPANSION",

        "alert_key": alert_key,

        "time": candle_time,

        "direction": direction,

        "price": candle["close"],

        "bb25_mid": candle["bb25_mid"],

        "bb25_width": candle["bb25_width"],

        "min_width": min_width,

        "min_width_time": str(
            df.loc[
                min_position,
                "datetime"
            ]
        ),

        "expansion_count":
            expansion_count,

        "bbw_old": bbw_old,

        "bbw_mid": bbw_mid,

        "bbw_new": bbw_new,

        "adx": adx,

        "plus_di": plus_di,

        "minus_di": minus_di,

        "di_status": di_status
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Getting market data..."
    )

    df = get_data()

    print(
        "Calculating indicators..."
    )

    df = calculate_bb50(df)

    df = calculate_bb25(df)

    df = calculate_bbw(df)

    df = calculate_adx(df)

    # -----------------------------------------------------
    # آخرین کندل بسته شده
    # -----------------------------------------------------

    index = len(df) - 2

    print(
        f"Checking candle: "
        f"{df.iloc[index]['datetime']}"
    )

    # -----------------------------------------------------
    # Strategy 1
    # -----------------------------------------------------

    signal_1 = check_bb50_touch(
        df,
        index
    )

    # -----------------------------------------------------
    # Strategy 2
    # -----------------------------------------------------

    signal_2 = check_bb25_expansion(
        df,
        index
    )

    # -----------------------------------------------------
    # جمع آلارم ها
    # -----------------------------------------------------

    signals = []

    if signal_1 is not None:
        signals.append(signal_1)

    if signal_2 is not None:
        signals.append(signal_2)

    if not signals:

        print(
            "No valid signal."
        )

        return

    # -----------------------------------------------------
    # State
    # -----------------------------------------------------

    state = load_state()

    sent_keys = set(
        state.get(
            "last_alert_keys",
            []
        )
    )

    for signal in signals:

        alert_key = signal[
            "alert_key"
        ]

        # جلوگیری از اسپم

        if alert_key in sent_keys:

            print(
                f"Already sent: "
                f"{alert_key}"
            )

            continue

        # =================================================
        # MESSAGE - STRATEGY 1
        # =================================================

        if signal["type"] == "BB50 TOUCH":

            message = f"""
🚨 XAU/USD ALERT

{signal["emoji"]} برخورد قیمت با:
{signal["touch"]}

💰 قیمت:
{signal["price"]:.3f}

━━━━━━━━━━━━━━

📊 BB50 EMA

📈 BBW20 — سه کندل قبل:
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

⏱ {signal["time"]}
""".strip()

        # =================================================
        # MESSAGE - STRATEGY 2
        # =================================================

        else:

            message = f"""
🚨 XAU/USD EXPANSION ALERT

🔥 Bollinger 25
SQUEEZE → EXPANSION

📌 جهت:
{signal["direction"]}

💰 قیمت:
{signal["price"]:.3f}

━━━━━━━━━━━━━━

📊 BB25 EMA

Midline:
{signal["bb25_mid"]:.3f}

Current Width:
{signal["bb25_width"]:.4f}

Minimum Width:
{signal["min_width"]:.4f}

Minimum Width Time:
{signal["min_width_time"]}

📈 Expansion:
{signal["expansion_count"]} کندل متوالی

━━━━━━━━━━━━━━

📍 BBW20 — سه کندل قبل:

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

⏱ {signal["time"]}
""".strip()

        # -------------------------------------------------
        # Send Telegram
        # -------------------------------------------------

        print(message)

        send_telegram(
            message
        )

        # -------------------------------------------------
        # Save state
        # -------------------------------------------------

        sent_keys.add(
            alert_key
        )

        print(
            f"Alert sent: "
            f"{alert_key}"
        )

    # -----------------------------------------------------
    # نگهداری آخرین 100 آلارم
    # -----------------------------------------------------

    state["last_alert_keys"] = list(
        sent_keys
    )[-100:]

    save_state(
        state
    )

    print(
        "State saved."
    )


# =========================================================

if __name__ == "__main__":
    main()
