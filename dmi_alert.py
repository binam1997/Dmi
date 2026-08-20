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
        "timezone": "Asia/Tehran"
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
# BOLLINGER 100 EMA
# =========================================================

def calculate_bb100(df):

    basis = ema(
        df["close"],
        BB100_LENGTH
    )

    std = df["close"].rolling(
        BB100_LENGTH
    ).std(ddof=0)

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

    # -----------------------------------------------------
    # برخورد با باند
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # BBW باید صعودی باشد
    # -----------------------------------------------------

    if not bbw_is_rising(
        df,
        index
    ):

        return None

    # -----------------------------------------------------
    # ADX / DI
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

    # -----------------------------------------------------
    # نوع برخورد
    # -----------------------------------------------------

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
# BB100 TOUCH
# +
# BBW20 RISING
#
# دقیقاً مشابه Strategy 1
# =========================================================

def check_bb100_touch(df, index):

    if index < 5:
        return None

    candle = df.iloc[index]

    # -----------------------------------------------------
    # برخورد با باند
    # -----------------------------------------------------

    upper_touch = (
        candle["high"]
        >= candle["bb100_upper"]
    )

    lower_touch = (
        candle["low"]
        <= candle["bb100_lower"]
    )

    if not upper_touch and not lower_touch:
        return None

    # -----------------------------------------------------
    # BBW باید صعودی باشد
    # -----------------------------------------------------

    if not bbw_is_rising(
        df,
        index
    ):

        return None

    # -----------------------------------------------------
    # ADX / DI
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

    # -----------------------------------------------------
    # نوع برخورد
    # -----------------------------------------------------

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
        f"BB100_{candle_time}_{touch}"
    )

    return {

        "type": "BB100 TOUCH",

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

    # -----------------------------------------------------
    # Bollinger 50
    # -----------------------------------------------------

    df = calculate_bb50(df)

    # -----------------------------------------------------
    # Bollinger 100
    # -----------------------------------------------------

    df = calculate_bb100(df)

    # -----------------------------------------------------
    # BBW20
    # -----------------------------------------------------

    df = calculate_bbw(df)

    # -----------------------------------------------------
    # ADX14
    # -----------------------------------------------------

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

    signal_2 = check_bb100_touch(
        df,
        index
    )

    # -----------------------------------------------------
    # جمع آلارم ها
    # -----------------------------------------------------

    signals = []

    if signal_1 is not None:

        signals.append(
            signal_1
        )

    if signal_2 is not None:

        signals.append(
            signal_2
        )

    # -----------------------------------------------------
    # هیچ سیگنالی وجود ندارد
    # -----------------------------------------------------

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

    # =====================================================
    # SEND SIGNALS
    # =====================================================

    for signal in signals:

        alert_key = signal[
            "alert_key"
        ]

        # -------------------------------------------------
        # جلوگیری از اسپم
        # -------------------------------------------------

        if alert_key in sent_keys:

            print(
                f"Already sent: "
                f"{alert_key}"
            )

            continue

        # =================================================
        # MESSAGE
        # =================================================

        message = f"""
🚨 XAU/USD ALERT

{signal["emoji"]} برخورد قیمت با:
{signal["touch"]}

📊 Strategy:
{signal["type"]}

💰 قیمت:
{signal["price"]:.3f}

━━━━━━━━━━━━━━

📈 Bollinger Band

{"BB50 EMA" if signal["type"] == "BB50 TOUCH" else "BB100 EMA"}

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

        # -------------------------------------------------
        # چاپ در Console
        # -------------------------------------------------

        print(message)

        # -------------------------------------------------
        # ارسال تلگرام
        # -------------------------------------------------

        send_telegram(
            message
        )

        # -------------------------------------------------
        # ذخیره Alert Key
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
