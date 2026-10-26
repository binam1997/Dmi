import json
import os
import requests
import pandas as pd
from zoneinfo import ZoneInfo

# تنظیم منطقه زمانی ایران برای نمایش در پیام‌ها
IRAN_TZ = ZoneInfo("Asia/Tehran")

# =========================================================
# تنظیمات اصلی
# =========================================================

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOL = "XAU/USD"
INTERVAL = "30min"
OUTPUT_SIZE = 150     # دریافت ۱۵۰ کندل برای عبور از تعطیلات آخر هفته

# تنظیمات بولینجر باند
BB_LENGTH = 20
BB_MULT = 2.5

# نام فایلی که در فایل YAML شما تنظیم شده است
STATE_FILE = "state.json"

# =========================================================
# توابع کمکی
# =========================================================

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

# =========================================================
# دریافت داده و محاسبه اندیکاتور
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
    
    # فیلتر حیاتی: حذف کندل‌های روز شنبه و یکشنبه برای همخوانی با تریدینگ‌ویو
    df = df[df['datetime'].dt.dayofweek < 5]
    
    # مرتب‌سازی از قدیمی به جدید
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def add_bollinger_bands(df):
    # میانگین متحرک ساده (SMA)
    df['sma'] = df['close'].rolling(window=BB_LENGTH).mean()
    
    # انحراف معیار
    df['std'] = df['close'].rolling(window=BB_LENGTH).std(ddof=0)
    
    # محاسبه باندهای بالا و پایین
    df['upper_band'] = df['sma'] + (BB_MULT * df['std'])
    df['lower_band'] = df['sma'] - (BB_MULT * df['std'])
    
    return df

# =========================================================
# اجرای اصلی
# =========================================================

def main():
    # اگر دکمه اجرای تست در گیت‌هاب زده شده باشد
    if os.environ.get("TEST_MESSAGE") == "1":
        send_telegram("✅ ربات سیگنال بولینجر باند فعال است (پیام تستی)")
        print("Test message sent.")
        return

    print(f"Fetching data for {SYMBOL} ({INTERVAL})...")
    df = get_data()
    
    if len(df) < BB_LENGTH:
        print("داده کافی برای محاسبه بولینجر وجود ندارد.")
        return

    df = add_bollinger_bands(df)
    
    # گرفتن آخرین کندل تشکیل‌شده (یا در حال تشکیل)
    last_candle = df.iloc[-1]
    
    # تبدیل زمان به فرمت خوانا در ساعت ایران
    candle_time_str = last_candle['datetime'].tz_convert(IRAN_TZ).strftime("%Y-%m-%d %H:00")
    
    high = last_candle['high']
    low = last_candle['low']
    close = last_candle['close']
    upper = last_candle['upper_band']
    lower = last_candle['lower_band']

    print(f"Current Candle: {candle_time_str}")
    print(f"H: {high:.2f} | L: {low:.2f} | UB: {upper:.2f} | LB: {lower:.2f}")

    # لود کردن وضعیت قبلی تا در یک ساعت مشخص، دوبار پیام ندهد
    state = load_state()
    last_alerted_candle = state.get("last_alerted_candle", "")

    if last_alerted_candle == candle_time_str:
        print("آلارم برای این کندل ۱ ساعته قبلاً ارسال شده است.")
        return

    alert_message = ""

    # بررسی شرایط برخورد
    if high >= upper:
        alert_message = (
            f"🔴 برخورد به سقف بولینجر باند!\n"
            f"نماد: {SYMBOL} (تایم‌فریم ۱ ساعته)\n"
            f"💰 قیمت فعلی: {close:.2f}\n"
            f"📈 بالاترین قیمت (High): {high:.2f}\n"
            f"📊 باند بالا: {upper:.2f}\n\n"
            f"⏱ زمان کندل: {candle_time_str}"
        )
    elif low <= lower:
        alert_message = (
            f"🟢 برخورد به کف بولینجر باند!\n"
            f"نماد: {SYMBOL} (تایم‌فریم ۱ ساعته)\n"
            f"💰 قیمت فعلی: {close:.2f}\n"
            f"📉 پایین‌ترین قیمت (Low): {low:.2f}\n"
            f"📊 باند پایین: {lower:.2f}\n\n"
            f"⏱ زمان کندل: {candle_time_str}"
        )

    # اگر برخوردی وجود داشت، پیام بفرست و state را آپدیت کن
    if alert_message:
        print("برخورد تشخیص داده شد! ارسال پیام به تلگرام...")
        send_telegram(alert_message)
        
        # ثبت زمان برای جلوگیری از پیام تکراری در این ساعت
        state["last_alerted_candle"] = candle_time_str
        save_state(state)
    else:
        print("برخوردی با باندهای بولینجر صورت نگرفته است.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        
