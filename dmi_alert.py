import json
import os

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


IRAN_TZ = ZoneInfo("Asia/Tehran")


# =========================================================
# SETTINGS (same values as the TradingView strategy "WPR DT/DB")
# =========================================================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOL = "XAU/USD"
INTERVAL = "5min"
BAR_MINUTES = 5
OUTPUT_SIZE = 1000
WARMUP_BARS = 300
MIN_CANDLES = WARMUP_BARS + 50

# signals older than this many closed candles are ignored (protects from late runs)
MAX_AGE_BARS = 3

STATE_FILE = "wpr_state.json"
ERROR_ALERT_MINUTES = 60

# --- general ---
ENTRY_MODE = "all"                # "all" = on every DT/DB diamond, "main" = after neckline break
LOC_MODE = "mean_reversion"       # "mean_reversion", "with_trend", "off"
REV_OK = True                     # opposite signal may reverse an open trade
BB_LEN = 100
BB_TYPE = "SMA"                   # "SMA" or "EMA"

# --- WPR ---
W_LEN = 30
W_SM = 20
MID_LEN = 100
W_HL = True

# --- %B ---
PB_LEN = 25
PB_MULT = 2.0

# --- 5m phase filter ---
PHASE_MODE = "off"                # "off", "block", "require"
WALK_N = 12
WALK_LO = 0.5
TURN_LVL = 0.5

# --- 15m %B filter ---
H15_MODE = "direction"            # "off", "direction", "side"
H15_LVL = 0.5

# --- pattern ---
P_L = 3
P_R = 3
MIN_GAP = 4
MAX_GAP = 40
ALLOW_FS = True                   # Allow Divergence / Failure Swing
TOL = 8.0
MIN_DEPTH = 10.0
REQ_SIDE = True
TRIG_WIN = 30

# --- stop and target ---
STOP_MODE = "swing"               # "atr" = distance from entry, "swing" = swing extreme +- ATR buffer

SWING_LEN = 30
ATR_LEN = 14
ATR_TYPE = "rma"                  # "rma" or "sma"
ATR_BUF = 2.0
ATR_MULT = 1.5
MAX_RISK_PTS = 0.0                # 0 = off
RISK_MODE = "skip"                # "skip" or "cap"
TP_MODE = "rr"                    # "rr" or "bbmid"
RR = 1.0                          # Changed to 1.0
MIN_TPR = 0.5

# --- exit mode ---
X_MODE = "fixed"                  # "fixed", "pctb", "pctb_tp"
X_LVL = 0.5

ENTRY_ALL = ENTRY_MODE == "all"
MAIN_MODE = not ENTRY_ALL
USE_BX = X_MODE != "fixed"
TP_ON = X_MODE != "pctb"


# =========================================================
# HELPERS
# =========================================================

def safe_text(text):

    text = str(text)

    for secret in (TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN):
        if secret:
            text = text.replace(secret, "***")

    return text


def load_state():

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


def save_state(state):

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


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
        "timezone": "UTC",
        "order": "asc"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise Exception(f"TwelveData Error: {data}")

    df = pd.DataFrame(data["values"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # keep only fully closed candles (datetime = candle open time)
    now = pd.Timestamp.now(tz="UTC")
    closed = (df["datetime"] + pd.Timedelta(minutes=BAR_MINUTES)) <= now
    df = df[closed].reset_index(drop=True)

    return df


# =========================================================
# INDICATORS
# =========================================================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def add_indicators(df):

    close = df["close"]

    # WPR
    hi = df["high"] if W_HL else close
    lo = df["low"] if W_HL else close
    highest = hi.rolling(W_LEN).max()
    lowest = lo.rolling(W_LEN).min()
    raw = (close - lowest) / (highest - lowest) * 200 - 100
    df["wpr"] = ema(raw, W_SM)
    df["wmid"] = ema(df["wpr"], MID_LEN)

    # price Bollinger midline
    df["bbmid"] = ema(close, BB_LEN) if BB_TYPE == "EMA" else close.rolling(BB_LEN).mean()

    # %B on the chart timeframe
    pb_mid = close.rolling(PB_LEN).mean()
    pb_dev = PB_MULT * close.rolling(PB_LEN).std(ddof=0)
    df["pb"] = (close - (pb_mid - pb_dev)) / (2 * pb_dev)

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean() if ATR_TYPE == "sma" else rma(tr, ATR_LEN)

    df["swing_hi"] = df["high"].rolling(SWING_LEN).max()
    df["swing_lo"] = df["low"].rolling(SWING_LEN).min()

    return df

def add_h15(df):

    # %B of the last two CLOSED 15m bars before the 15m bar that contains each 5m bar
    s = df.set_index("datetime")["close"]
    c15 = s.resample("15min", label="left", closed="left").last().dropna()
    mid = c15.rolling(PB_LEN).mean()
    dev = PB_MULT * c15.rolling(PB_LEN).std(ddof=0)
    pb15 = (c15 - (mid - dev)) / (2 * dev)

    last1 = pb15.shift(1)
    last2 = pb15.shift(2)

    start = df["datetime"].dt.floor("15min")
    df["hb_now"] = last1.reindex(start).to_numpy()
    df["hb_prev"] = last2.reindex(start).to_numpy()

    return df


# =========================================================
# SIGNAL LOGIC (port of the Pine strategy)
# =========================================================

def run_logic(df):

    n = len(df)
    t = df["datetime"].tolist()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    w = df["wpr"].to_numpy()
    wm = df["wmid"].to_numpy()
    bb = df["bbmid"].to_numpy()
    pb = df["pb"].to_numpy()
    at = df["atr"].to_numpy()
    sh = df["swing_hi"].to_numpy()
    sl = df["swing_lo"].to_numpy()
    hbn = df["hb_now"].to_numpy()
    hbp = df["hb_prev"].to_numpy()

    signals = []

    pos = 0
    stop_lv = None
    tp_lv = None
    x_arm = False

    # double top state
    ph_l = None
    ph_lix = None
    ph_lmid = None
    neck_t = None
    arm_top = False
    arm_t_neck = None
    arm_t_lvl = None
    arm_t_bar = None

    # double bottom state
    pl_l = None
    pl_lix = None
    pl_lmid = None
    neck_b = None
    arm_bot = False
    arm_b_neck = None
    arm_b_lvl = None
    arm_b_bar = None

    for i in range(1, n):

        if i < WARMUP_BARS:
            continue

        # ---------- stop / take profit hit during this bar ----------
        if pos != 0 and stop_lv is not None:
            if pos == 1:
                hit_stop = l[i] <= stop_lv
                hit_tp = TP_ON and tp_lv is not None and h[i] >= tp_lv
            else:
                hit_stop = h[i] >= stop_lv
                hit_tp = TP_ON and tp_lv is not None and l[i] <= tp_lv
            if hit_stop or hit_tp:
                pos = 0
                stop_lv = None
                tp_lv = None

        # ---------- pivots on WPR ----------
        cidx = i - P_R
        new_ph = False
        new_pl = False
        ph = None
        pl = None
        if cidx - P_L >= 0:
            center = w[cidx]
            seg_l = w[cidx - P_L:cidx]
            seg_r = w[cidx + 1:i + 1]
            if not np.isnan(center) and not np.isnan(seg_l).any() and not np.isnan(seg_r).any():
                if center > seg_l.max() and center > seg_r.max():
                    new_ph = True
                    ph = float(center)
                elif center < seg_l.min() and center < seg_r.min():
                    new_pl = True
                    pl = float(center)
        m_at_p = wm[cidx] if cidx >= 0 else np.nan

        # ---------- double top ----------
        valid_t = False
        if new_ph and ph_l is not None and neck_t is not None:
            gap = cidx - ph_lix
            wpr_top_cond = (ph <= ph_l + TOL) if ALLOW_FS else (abs(ph - ph_l) <= TOL)
            
            if MIN_GAP <= gap <= MAX_GAP and wpr_top_cond and (min(ph, ph_l) - neck_t) >= MIN_DEPTH:
                if (not REQ_SIDE) or (ph > m_at_p and ph_l > ph_lmid):
                    valid_t = True

        if valid_t:
            arm_top = MAIN_MODE
            arm_t_neck = neck_t
            arm_t_lvl = max(ph, ph_l)
            arm_t_bar = i
        elif new_ph:
            arm_top = False

        if new_ph:
            neck_t = None
        elif new_pl:
            neck_t = pl if neck_t is None else min(neck_t, pl)

        if new_ph:
            ph_l = ph
            ph_lix = cidx
            ph_lmid = m_at_p

        exp_t = arm_top and (i - arm_t_bar) > TRIG_WIN
        inv_t = arm_top and w[i] > arm_t_lvl + TOL
        brk_t = arm_top and w[i] < arm_t_neck
        raw_sell = valid_t if ENTRY_ALL else brk_t
        if brk_t or exp_t or inv_t:
            arm_top = False

        # ---------- double bottom ----------
        valid_b = False
        if new_pl and pl_l is not None and neck_b is not None:
            gap = cidx - pl_lix
            wpr_bot_cond = (pl >= pl_l - TOL) if ALLOW_FS else (abs(pl - pl_l) <= TOL)
            
            if MIN_GAP <= gap <= MAX_GAP and wpr_bot_cond and (neck_b - max(pl, pl_l)) >= MIN_DEPTH:
                if (not REQ_SIDE) or (pl < m_at_p and pl_l < pl_lmid):
                    valid_b = True

        if valid_b:
            arm_bot = MAIN_MODE
            arm_b_neck = neck_b
            arm_b_lvl = min(pl, pl_l)
            arm_b_bar = i
        elif new_pl:
            arm_bot = False

        if new_pl:
            neck_b = None
        elif new_ph:
            neck_b = ph if neck_b is None else max(neck_b, ph)

        if new_pl:
            pl_l = pl
            pl_lix = cidx
            pl_lmid = m_at_p

        exp_b = arm_bot and (i - arm_b_bar) > TRIG_WIN
        inv_b = arm_bot and w[i] < arm_b_lvl - TOL
        brk_b = arm_bot and w[i] > arm_b_neck
        raw_buy = valid_b if ENTRY_ALL else brk_b
        if brk_b or exp_b or inv_b:
            arm_bot = False

        # ---------- location filter ----------
        if LOC_MODE == "off":
            loc_ok_s = True
            loc_ok_b = True
        elif LOC_MODE == "mean_reversion":
            loc_ok_s = bool(c[i] > bb[i])
            loc_ok_b = bool(c[i] < bb[i])
        else:
            loc_ok_s = bool(c[i] < bb[i])
            loc_ok_b = bool(c[i] > bb[i])

        # ---------- %B 5m phase filter ----------
        win = pb[max(0, i - WALK_N + 1):i + 1]
        dn_walk = bool(win.max() < TURN_LVL and pb[i] <= WALK_LO)
        up_walk = bool(win.min() > TURN_LVL and pb[i] >= 1 - WALK_LO)
        if PHASE_MODE == "block":
            blk_buy_p = dn_walk
            blk_sell_p = up_walk
        elif PHASE_MODE == "require":
            blk_buy_p = not (pb[i] > TURN_LVL)
            blk_sell_p = not (pb[i] < TURN_LVL)
        else:
            blk_buy_p = False
            blk_sell_p = False

        # ---------- %B 15m filter ----------
        hb_rise = bool(hbn[i] > hbp[i])
        hb_fall = bool(hbn[i] < hbp[i])
        if H15_MODE == "off":
            h15_buy_ok = True
            h15_sell_ok = True
        else:
            h15_buy_ok = hb_rise and (H15_MODE == "direction" or hbn[i] > H15_LVL)
            h15_sell_ok = hb_fall and (H15_MODE == "direction" or hbn[i] < H15_LVL)

        buy_pass = (not blk_buy_p) and h15_buy_ok
        sell_pass = (not blk_sell_p) and h15_sell_ok

        # ---------- stop and target ----------
        atr_i = at[i]
        if STOP_MODE == "swing":
            sell_stop_raw = sh[i] + ATR_BUF * atr_i
            buy_stop_raw = sl[i] - ATR_BUF * atr_i
        else:
            sell_stop_raw = c[i] + ATR_MULT * atr_i
            buy_stop_raw = c[i] - ATR_MULT * atr_i

        cap = RISK_MODE == "cap"
        too_big_s = MAX_RISK_PTS > 0 and (sell_stop_raw - c[i]) > MAX_RISK_PTS
        too_big_b = MAX_RISK_PTS > 0 and (c[i] - buy_stop_raw) > MAX_RISK_PTS
        sell_stop = c[i] + MAX_RISK_PTS if (too_big_s and cap) else sell_stop_raw
        buy_stop = c[i] - MAX_RISK_PTS if (too_big_b and cap) else buy_stop_raw
        sell_risk = sell_stop - c[i]
        buy_risk = c[i] - buy_stop

        mid_s = TP_MODE == "bbmid" and (c[i] - bb[i]) >= MIN_TPR * sell_risk
        mid_b = TP_MODE == "bbmid" and (bb[i] - c[i]) >= MIN_TPR * buy_risk
        sell_tp = bb[i] if mid_s else c[i] - RR * sell_risk
        buy_tp = bb[i] if mid_b else c[i] + RR * buy_risk
        tp_ok_s = not (too_big_s and not cap)
        tp_ok_b = not (too_big_b and not cap)

        # ---------- SIGNAL EVALUATION AND ALERTS ----------
        direction = 0
        is_action = False
        block_reason = None

        if raw_sell:
            direction = -1
            if not loc_ok_s:
                block_reason = "محل قیمت (زیر میانگین بولینجر)"
            elif not sell_pass:
                block_reason = f"فاز %B 5m یا 15m"
            elif not tp_ok_s:
                block_reason = "ریسک فراتر از حد مجاز"
            elif pos < 0 or (pos > 0 and not REV_OK):
                block_reason = "یک موقعیت فروش از قبل باز است"
            else:
                is_action = True

        elif raw_buy:
            direction = 1
            if not loc_ok_b:
                block_reason = "محل قیمت (بالای میانگین بولینجر)"
            elif not buy_pass:
                block_reason = f"فاز %B 5m یا 15m"
            elif not tp_ok_b:
                block_reason = "ریسک فراتر از حد مجاز"
            elif pos > 0 or (pos < 0 and not REV_OK):
                block_reason = "یک موقعیت خرید از قبل باز است"
            else:
                is_action = True

        # ---------- %B exit ----------
        bx_long = USE_BX and pos > 0 and x_arm and pb[i] <= X_LVL
        bx_short = USE_BX and pos < 0 and x_arm and pb[i] >= X_LVL
        if pos == 0:
            x_arm = False
        else:
            x_arm = x_arm or ((pb[i] > X_LVL) if pos > 0 else (pb[i] < X_LVL))

        pos_before = pos

        # Add Signal / Blocked Event to List
        if direction != 0:
            signals.append({
                "i": i,
                "time": t[i],
                "dir": direction,
                "is_action": is_action,
                "block_reason": block_reason,
                "entry": float(c[i]),
                "stop": float(sell_stop if direction == -1 else buy_stop) if is_action else None,
                "tp": float(sell_tp if direction == -1 else buy_tp) if TP_ON and is_action else None,
                "wpr": float(w[i]),
                "pb5": float(pb[i]),
                "pb15": float(hbn[i]) if not np.isnan(hbn[i]) else None,
                "pb15_up": hb_rise,
                "reversal": (pos_before != 0) if is_action else False,
            })
            
            # Execute trade if conditions met
            if is_action:
                pos = direction
                stop_lv = float(sell_stop if direction == -1 else buy_stop)
                tp_lv = float(sell_tp if direction == -1 else buy_tp) if TP_ON else None
                x_arm = False

        # Add Exit Event to List
        elif (bx_long and pos == 1) or (bx_short and pos == -1):
            signals.append({
                "i": i,
                "time": t[i],
                "dir": 0, # 0 means Exit
                "is_action": True,
                "block_reason": None,
                "entry": float(c[i]),
                "stop": None,
                "tp": None,
                "wpr": float(w[i]),
                "pb5": float(pb[i]),
                "pb15": None,
                "pb15_up": None,
                "reversal": False,
            })
            pos = 0
            stop_lv = None
            tp_lv = None

    return signals


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload, timeout=20).raise_for_status()


def send_error_alert(error_text):
    try:
        state = load_state()
        now = pd.Timestamp.now(tz=IRAN_TZ)

        last = state.get("last_error_time")
        if last:
            elapsed = (now - pd.Timestamp(last)).total_seconds()
            if elapsed < ERROR_ALERT_MINUTES * 60:
                return

        send_telegram(f"⚠️ ربات سیگنال WPR خطا داد:\n\n{error_text}")

        state["last_error_time"] = now.isoformat()
        save_state(state)
    except Exception:
        pass


def build_message(s, last_idx, last_close):

    close_time = (s["time"] + pd.Timedelta(minutes=BAR_MINUTES)).tz_convert(IRAN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    age = last_idx - s["i"]
    age_txt = f"\n⏱ این پیام {age} کندل پیش صادر شده | قیمت فعلی: {last_close:.2f}" if age >= 1 else ""

    # پیام خروج
    if s["dir"] == 0:
        return f"""
⚪️ خروج از معامله (رسیدن به سطح %B)

💰 قیمت خروج: {s['entry']:.2f}
━━━━━━━━━━━━━━
⏱ زمان: {close_time}{age_txt}
""".strip()

    is_buy = s["dir"] == 1
    side = "خرید (کف دوقلو)" if is_buy else "فروش (سقف دوقلو)"
    mode = "روی لوزی" if ENTRY_ALL else "شکست خط گردن"

    # پیام مسدود شدن الگو
    if not s["is_action"]:
        icon = "⚠️"
        return f"""
{icon} الگوی {side} مسدود شد!
نماد: {SYMBOL} - ۵ دقیقه

❌ علت مسدود شدن: {s['block_reason']}
📊 مقدار %B پنج دقیقه: {s['pb5']:.2f}
💰 قیمت فعلی: {s['entry']:.2f}
━━━━━━━━━━━━━━
⏱ زمان: {close_time}{age_txt}
""".strip()

    # پیام سیگنال قطعی
    icon = "🟢" if is_buy else "🔴"
    risk = abs(s["entry"] - s["stop"])

    if s["tp"] is None:
        tp_line = "🎯 حد سود: ندارد (خروج با %B)"
    else:
        reward = abs(s["tp"] - s["entry"])
        rr_val = reward / risk if risk > 0 else 0
        tp_line = f"🎯 حد سود: {s['tp']:.2f} (ریوارد {rr_val:.1f} برابر)"

    pb15_txt = "-"
    if s["pb15"] is not None:
        pb15_txt = f"{s['pb15']:.2f} " + ("صعودی" if s["pb15_up"] else "نزولی")

    extra = ""
    if s["reversal"]:
        extra += "\n🔁 این سیگنال معامله باز قبلی را معکوس می کند"

    return f"""
{icon} سیگنال قطعی {side} ({SYMBOL} - ۵ دقیقه)

💰 ورود (کلوز کندل): {s['entry']:.2f}
🛑 استاپ: {s['stop']:.2f} (ریسک {risk:.2f})
{tp_line}

📊 WPR: {s['wpr']:.1f} | %B پنج دقیقه: {s['pb5']:.2f}
📈 %B پانزده دقیقه: {pb15_txt}
🧩 حالت ورود: {mode}

━━━━━━━━━━━━━━
⏱ زمان بسته شدن کندل: {close_time}{extra}{age_txt}
""".strip()


# =========================================================
# MAIN
# =========================================================

def main():

    if os.environ.get("TEST_MESSAGE") == "1":
        send_telegram("✅ ربات سیگنال WPR فعال است")
        print("Test message sent.")
        return

    print("Getting market data...")
    df = get_data()

    if len(df) < MIN_CANDLES:
        print(f"Not enough candles yet ({len(df)} < {MIN_CANDLES}).")
        return

    print("Calculating indicators...")
    df = add_indicators(df)
    df = add_h15(df)

    signals = run_logic(df)

    state = load_state()
    last_alert = None
    if state.get("last_alert_time"):
          last_alert = pd.Timestamp(state["last_alert_time"])

    last_idx = len(df) - 1
    last_row = df.iloc[-1]
    last_close = float(last_row["close"])

    print(f"Last closed candle: {last_row['datetime']} | close: {last_close:.2f}")
    print(
        f"WPR: {last_row['wpr']:.2f} | WPR mid: {last_row['wmid']:.2f} | "
        f"%B 5m: {last_row['pb']:.2f} | %B 15m: {last_row['hb_now']:.2f} (prev {last_row['hb_prev']:.2f})"
    )

    pending = []
    for s in signals:
        fresh = (last_idx - s["i"]) < MAX_AGE_BARS
        is_new = last_alert is None or s["time"] > last_alert
        if fresh and is_new:
            pending.append(s)

    print(f"Signals found: {len(signals)} | pending: {len(pending)}")

    for s in pending:

        message = build_message(s, last_idx, last_close)

        print(f"Sending Alert:\n{message}")
        send_telegram(message)

        # save right after each successful send to avoid duplicates
        state["last_alert_time"] = s["time"].isoformat()
        save_state(state)

    save_state(state)

    print("Execution completed.")


# =========================================================

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {safe_text(e)}")
        send_error_alert(safe_text(e))
        raise
        
