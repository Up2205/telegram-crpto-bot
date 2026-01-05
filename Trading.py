"""
بوت تيليجرام للتداول الذكي - نسخة محسنة
"""
import os
import logging
import psycopg2
from functools import lru_cache
from typing import List, Tuple, Optional
import ccxt
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
from dataclasses import dataclass
from enum import Enum
from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ✅ إعدادات Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ✅ إعدادات الأدمن وقاعدة البيانات
ADMIN_IDS = [5389040264]
DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ إعدادات تيليجرام - استخدام متغير بيئة
TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    logger.error("⚠️ BOT_TOKEN غير موجود! ضع التوكن في متغير البيئة.")

# ✅ إعداد قاعدة البيانات
def get_db():
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL غير موجود!")
        return None
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    """إنشاء الجداول تلقائيًا إذا لم تكن موجودة"""
    if not DATABASE_URL:
        return
        
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS authorized_users (
                        user_id BIGINT PRIMARY KEY,
                        username TEXT,
                        added_by BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            conn.commit()
        logger.info("✅ تم تهيئة قاعدة البيانات بنجاح")
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة قاعدة البيانات: {e}")

def is_authorized(user_id: int) -> bool:
    """التحقق من صلاحية المستخدم"""
    # الأدمن دائماً مصرح له
    if user_id in ADMIN_IDS:
        return True
        
    if not DATABASE_URL:
        return False
        
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM authorized_users WHERE user_id = %s", (user_id,))
                return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"خطأ في التحقق من الصلاحية: {e}")
        return False

async def check_auth(update: Update) -> bool:
    """دالة مساعدة للتحقق والرد"""
    user = update.effective_user
    if is_authorized(user.id):
        # ✅ تحديث اسم المستخدم عند الاستخدام
        try:
            if user.username and DATABASE_URL:
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE authorized_users SET username = %s WHERE user_id = %s", (user.username, user.id))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating username: {e}")
            
        return True
        
    await update.message.reply_text(
        "⛔ *عذراً، هذا البوت خاص.*\n"
        "يرجى التواصل مع الأدمن (المسؤول) للتفعيل: @Up2205",
        parse_mode='Markdown'  # parse_mode: يسمح بتنسيق النص (مثل استخدام * للخط العريض)
    )
    return False

# ✅ أوامر إدارة المستخدمين (للأدمن فقط)
async def auth_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تفعيل مستخدم جديد"""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("استخدم الأمر هكذا:\n/auth 123456789")
        return

    try:
        new_user_id = int(context.args[0])
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO authorized_users (user_id, added_by)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                """, (new_user_id, user_id))
            conn.commit()
        await update.message.reply_text(f"✅ تم تفعيل المستخدم: `{new_user_id}`", parse_mode='Markdown')
    except ValueError:
        await update.message.reply_text("❌ تأكد من كتابة ID صحيح (أرقام فقط).")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {e}")

async def unauth_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إلغاء تفعيل مستخدم"""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("استخدم الأمر هكذا:\n/unauth 123456789")
        return

    try:
        target_id = int(context.args[0])
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM authorized_users WHERE user_id = %s", (target_id,))
            conn.commit()
        await update.message.reply_text(f"⛔ تم إلغاء تفعيل المستخدم: `{target_id}`", parse_mode='Markdown')
    except ValueError:
        await update.message.reply_text("❌ تأكد من كتابة ID صحيح.")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {e}")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض قائمة المستخدمين المصرح لهم (للأدمن فقط)"""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ *عذراً، هذا الأمر للأدمن فقط.*", parse_mode='Markdown')
        return
    await update.message.reply_text("جاري جلب المستخدمين👥...")
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, username, created_at FROM authorized_users ORDER BY created_at DESC")
                rows = cur.fetchall()

        if not rows:
            await update.message.reply_text("📭 لا يوجد مستخدمين مصرح لهم.")
            return
        
        msg = "📋 قائمة المستخدمين المصرح لهم: \n\n"
        for uid, username, created_at in rows:
            # ✅ محاولة جلب الاسم من تيليجرام إذا كان غير موجود
            if not username:
                try:
                    chat = await context.bot.get_chat(uid)
                    if chat.username:
                        username = chat.username
                        # تحديث القاعدة
                        with get_db() as conn:
                            with conn.cursor() as cur:
                                cur.execute("UPDATE authorized_users SET username = %s WHERE user_id = %s", (username, uid))
                            conn.commit()
                except Exception as e:
                    logger.warning(f"Could not fetch info for {uid}: {e}")

            # ✅ معالجة الاسم لتجنب أخطاء Markdown
            if username:
                safe_username = username.replace("_", "\\_")
                user_link = f"@{safe_username}"
            else:
                user_link = "بدون معرف"

            date_str = created_at.strftime("%Y-%m-%d") if created_at else "?"
            msg += f"👤 `{uid}` - {user_link}\n📅 {date_str}\n\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        await update.message.reply_text("❌ حدث خطأ أثناء جلب القائمة.")


# ✅ إعداد Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 30000,
})

# ✅ قائمة العملات المتابعة
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT", "ADA/USDT"]

# ✅ متغير لتتبع طلبات الإيقاف
STOP_SIGNALS = {}

async def stop_execution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إيقاف العمليات الجارية"""
    user_id = update.effective_user.id
    STOP_SIGNALS[user_id] = True
    await update.message.reply_text("🛑 تم طلب إيقاف العمليات الجارية...")

# ✅ Cache للأسواق (يتم تحديثه كل 5 دقائق)
@lru_cache(maxsize=1)
def get_symbols_cached() -> List[str]:
    """جلب قائمة العملات مع caching"""
    try:
        markets = exchange.load_markets()
        symbols = [
            s['symbol'] for s in markets.values()
            if s['quote'] == 'USDT' and s['spot'] and s['active']
        ]
        logger.info(f"تم تحميل {len(symbols)} عملة")
        return symbols
    except Exception as e:
        logger.error(f"خطأ في جلب الأسواق: {e}")
        return []

def get_symbols() -> List[str]:
    """جلب قائمة العملات (بدون cache للتوافق)"""
    return get_symbols_cached()

def get_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 25) -> pd.DataFrame:
    """جلب بيانات OHLCV"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        logger.error(f"خطأ في جلب بيانات {symbol}: {e}")
        raise

def validate_symbol(symbol: str) -> bool:
    """التحقق من صحة اسم العملة"""
    if not symbol or '/' not in symbol:
        return False
    return symbol.upper() in get_symbols()

# ========== ✅ نظام الإشارات الاحترافي ==========

class SignalType(Enum):
    """أنواع الإشارات"""
    STRONG_BUY = "شراء قوي 🟢🟢🟢"
    BUY = "شراء 🟢"
    WEAK_BUY = "شراء ضعيف 🟡"
    HOLD = "احتفظ ⚪"
    WEAK_SELL = "بيع ضعيف 🟡"
    SELL = "بيع 🔴"
    STRONG_SELL = "بيع قوي 🔴🔴🔴"

@dataclass
class TradingSignal:
    """إشارة تداول احترافية"""
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    reasoning: List[str]
    indicators: dict

def calculate_advanced_indicators(df: pd.DataFrame) -> dict:
    """حساب جميع المؤشرات الفنية المتقدمة"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    indicators = {}
    
    # ✅ مؤشرات الزخم
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    indicators['RSI'] = rsi.rsi().iloc[-1]
    indicators['RSI_prev'] = rsi.rsi().iloc[-2] if len(df) > 1 else indicators['RSI']
    
    # ✅ Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
    indicators['Stoch_K'] = stoch.stoch().iloc[-1]
    indicators['Stoch_D'] = stoch.stoch_signal().iloc[-1]
    
    # ✅ Williams %R
    williams = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close)
    indicators['Williams_R'] = williams.williams_r().iloc[-1]
    
    # ✅ MACD
    macd = ta.trend.MACD(close=close)
    indicators['MACD'] = macd.macd().iloc[-1]
    indicators['MACD_Signal'] = macd.macd_signal().iloc[-1]
    indicators['MACD_Hist'] = macd.macd_diff().iloc[-1]
    indicators['MACD_Hist_prev'] = macd.macd_diff().iloc[-2] if len(df) > 1 else indicators['MACD_Hist']
    
    # ✅ مؤشرات الاتجاه - EMA
    ema_9 = ta.trend.EMAIndicator(close=close, window=9)
    ema_21 = ta.trend.EMAIndicator(close=close, window=21)
    ema_50 = ta.trend.EMAIndicator(close=close, window=50)
    ema_200 = ta.trend.EMAIndicator(close=close, window=200)
    indicators['EMA_9'] = ema_9.ema_indicator().iloc[-1]
    indicators['EMA_21'] = ema_21.ema_indicator().iloc[-1]
    indicators['EMA_50'] = ema_50.ema_indicator().iloc[-1]
    indicators['EMA_200'] = ema_200.ema_indicator().iloc[-1]
    
    # ✅ مؤشرات الاتجاه - SMA
    sma_20 = ta.trend.SMAIndicator(close=close, window=20)
    sma_50 = ta.trend.SMAIndicator(close=close, window=50)
    indicators['SMA_20'] = sma_20.sma_indicator().iloc[-1]
    indicators['SMA_50'] = sma_50.sma_indicator().iloc[-1]
    
    # ✅ ADX (قوة الاتجاه)
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close)
    indicators['ADX'] = adx.adx().iloc[-1]
    indicators['ADX_Pos'] = adx.adx_pos().iloc[-1]
    indicators['ADX_Neg'] = adx.adx_neg().iloc[-1]
    
    # ✅ Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    indicators['BB_Upper'] = bollinger.bollinger_hband().iloc[-1]
    indicators['BB_Middle'] = bollinger.bollinger_mavg().iloc[-1]
    indicators['BB_Lower'] = bollinger.bollinger_lband().iloc[-1]
    indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / indicators['BB_Middle'] * 100
    
    # ✅ ATR (متوسط المدى الحقيقي)
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close)
    indicators['ATR'] = atr.average_true_range().iloc[-1]
    
    # ✅ مؤشرات الحجم (حساب يدوي)
    volume_sma = volume.rolling(window=20).mean().iloc[-1]
    indicators['Volume_SMA'] = volume_sma if not pd.isna(volume_sma) else volume.iloc[-1]
    indicators['Volume_Ratio'] = volume.iloc[-1] / indicators['Volume_SMA'] if indicators['Volume_SMA'] > 0 else 1
    
    # ✅ السعر الحالي والتغيرات
    indicators['Price'] = close.iloc[-1]
    indicators['Price_Change_1h'] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(df) > 1 else 0
    indicators['Price_Change_24h'] = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100) if len(df) > 24 else 0
    
    # ✅ Support & Resistance
    indicators['Support'] = low.tail(20).min()
    indicators['Resistance'] = high.tail(20).max()
    
    # ✅ حجم التداول 24 ساعة بالدولار (سيتم تحديثه من ticker في الدوال)
    indicators['Volume_24h_USD'] = 0  # سيتم تحديثه لاحقاً
    
    return indicators

def is_good_trading_opportunity(signal: TradingSignal, indicators: dict, min_volume_ratio: float = 1.0, min_risk_reward: float = 1.2, max_volatility: float = 7.0, min_volume_24h: float = 500000) -> Tuple[bool, List[str]]:
    """
    التحقق من جودة فرصة التداول
    
    Parameters:
    - min_volume_ratio: الحد الأدنى لنسبة الحجم (Volume_Ratio)
    - min_risk_reward: الحد الأدنى لـ Risk/Reward
    - max_volatility: الحد الأقصى للتقلبات (BB_Width) - القيم الأعلى = تقلبات أكبر
    - min_volume_24h: الحد الأدنى لحجم التداول 24 ساعة بالدولار (افتراضي: $500K)
    """
    warnings = []
    is_good = True
    
    # ✅ التحقق من حجم التداول (Volume_Ratio)
    volume_ratio = indicators.get('Volume_Ratio', 1.0)
    if volume_ratio < min_volume_ratio:
        is_good = False
        warnings.append(f"⚠️ حجم تداول منخفض ({volume_ratio:.2f}x) - قد تكون السيولة قليلة")
    
    # ✅ التحقق من حجم التداول المطلق (24 ساعة)
    volume_24h = indicators.get('Volume_24h_USD', 0)
    if volume_24h > 0 and volume_24h < min_volume_24h:
        is_good = False
        warnings.append(f"⚠️ حجم تداول مطلق منخفض (${volume_24h:,.0f}) - سيولة قليلة، قد يكون صعب البيع/الشراء")
    
    # ✅ التحقق من Risk/Reward
    if signal.risk_reward < min_risk_reward:
        is_good = False
        warnings.append(f"⚠️ Risk/Reward ضعيف (1:{signal.risk_reward:.2f}) - يجب أن يكون على الأقل 1:{min_risk_reward}")
    
    # ✅ التحقق من الاتجاه العام (لإشارات الشراء)
    if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
        ema_200 = indicators.get('EMA_200', indicators['Price'])
        if indicators['Price'] < ema_200:
            warnings.append("🟡 تحذير: السعر تحت EMA200 (اتجاه هابط عام) - كن حذراً")
    
    # ✅ التحقق من التقلبات (BB_Width) - الأهم!
    bb_width = indicators.get('BB_Width', 0)
    if bb_width > max_volatility:
        is_good = False
        warnings.append(f"❌ تقلبات عالية جداً ({bb_width:.2f}%) - مخاطرة عالية جداً، تم استبعادها")
    
    # ✅ التحقق من قوة الاتجاه
    adx = indicators.get('ADX', 0)
    if adx < 20 and signal.signal_type not in [SignalType.HOLD]:
        warnings.append("🟡 ADX < 20 - الاتجاه ضعيف، قد يكون هناك تقلبات")
    
    return is_good, warnings

def calculate_guarantee_level(signal: TradingSignal, indicators: dict) -> Tuple[str, float]:
    """
    حساب مستوى الضمان للفرصة (من 0 إلى 100)
    Returns: (level_name, score)
    """
    score = 0
    max_score = 100
    
    # ✅ الثقة الأساسية (30 نقطة)
    score += (signal.confidence / 100) * 30
    
    # ✅ Risk/Reward (25 نقطة)
    if signal.risk_reward >= 3.0:
        score += 25
    elif signal.risk_reward >= 2.0:
        score += 20
    elif signal.risk_reward >= 1.5:
        score += 15
    elif signal.risk_reward >= 1.2:
        score += 10
    else:
        score += 5
    
    # ✅ حجم التداول (15 نقطة)
    volume_ratio = indicators.get('Volume_Ratio', 1.0)
    if volume_ratio >= 2.0:
        score += 15
    elif volume_ratio >= 1.5:
        score += 12
    elif volume_ratio >= 1.0:
        score += 8
    else:
        score += 3
    
    # ✅ قوة الاتجاه - ADX (15 نقطة)
    adx = indicators.get('ADX', 0)
    if adx >= 40:
        score += 15
    elif adx >= 30:
        score += 12
    elif adx >= 25:
        score += 8
    elif adx >= 20:
        score += 5
    else:
        score += 2
    
    # ✅ الاتجاه العام - EMA200 (10 نقطة)
    ema_200 = indicators.get('EMA_200', indicators['Price'])
    if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
        if indicators['Price'] > ema_200:
            score += 10
        else:
            score += 3
    elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
        if indicators['Price'] < ema_200:
            score += 10
        else:
            score += 3
    
    # ✅ التقلبات - BB Width (5 نقطة)
    bb_width = indicators.get('BB_Width', 0)
    if bb_width <= 3:
        score += 5
    elif bb_width <= 5:
        score += 3
    elif bb_width <= 8:
        score += 1
    # إذا كان > 8، لا نضيف نقاط
    
    # ✅ نوع الإشارة (10 نقطة)
    if signal.signal_type == SignalType.STRONG_BUY or signal.signal_type == SignalType.STRONG_SELL:
        score += 10
    elif signal.signal_type == SignalType.BUY or signal.signal_type == SignalType.SELL:
        score += 7
    elif signal.signal_type == SignalType.WEAK_BUY or signal.signal_type == SignalType.WEAK_SELL:
        score += 3
    
    # ✅ تحديد المستوى
    if score >= 80:
        level = "🟢 مضمون جداً"
    elif score >= 65:
        level = "🟢 مضمون"
    elif score >= 50:
        level = "🟡 جيد"
    elif score >= 35:
        level = "🟡 متوسط"
    else:
        level = "🔴 محفوف بالمخاطر"
    
    return level, min(100, score)

def analyze_professional_signal(df: pd.DataFrame, indicators: dict) -> TradingSignal:
    """تحليل احترافي وإنتاج إشارة تداول"""
    price = indicators['Price']
    reasoning = []
    buy_points = 0
    sell_points = 0
    
    # تحليل RSI
    rsi = indicators['RSI']
    rsi_prev = indicators['RSI_prev']
    if rsi < 30 and rsi > rsi_prev:
        buy_points += 3
        reasoning.append("✅ RSI في ذروة البيع + ارتفاع")
    elif rsi < 40:
        buy_points += 1
    elif rsi > 70 and rsi < rsi_prev:
        sell_points += 3
        reasoning.append("❌ RSI في ذروة الشراء + انخفاض")
    elif rsi > 60:
        sell_points += 1
    
    # تحليل MACD
    macd = indicators['MACD']
    macd_signal = indicators['MACD_Signal']
    macd_hist = indicators['MACD_Hist']
    macd_hist_prev = indicators['MACD_Hist_prev']
    if macd > macd_signal and macd_hist > 0 and macd_hist > macd_hist_prev:
        buy_points += 3
        reasoning.append("✅ MACD إشارة شراء قوية")
    elif macd > macd_signal:
        buy_points += 1
    elif macd < macd_signal and macd_hist < 0 and macd_hist < macd_hist_prev:
        sell_points += 3
        reasoning.append("❌ MACD إشارة بيع قوية")
    elif macd < macd_signal:
        sell_points += 1
    
    # تحليل EMA
    ema_9 = indicators['EMA_9']
    ema_21 = indicators['EMA_21']
    ema_50 = indicators['EMA_50']
    if ema_9 > ema_21 > ema_50 and price > ema_9:
        buy_points += 3
        reasoning.append("✅ Golden Cross + السعر فوق EMA9")
    elif ema_9 > ema_21:
        buy_points += 1
    elif ema_9 < ema_21 < ema_50 and price < ema_9:
        sell_points += 3
        reasoning.append("❌ Death Cross + السعر تحت EMA9")
    elif ema_9 < ema_21:
        sell_points += 1
    
    # تحليل ADX
    adx = indicators['ADX']
    if adx > 25:
        if indicators['ADX_Pos'] > indicators['ADX_Neg']:
            buy_points += 2
            reasoning.append("✅ ADX > 25 + اتجاه صاعد")
        else:
            sell_points += 2
            reasoning.append("❌ ADX > 25 + اتجاه هابط")
    
    # تحليل Bollinger Bands
    bb_upper = indicators['BB_Upper']
    bb_lower = indicators['BB_Lower']
    bb_middle = indicators.get('BB_Middle', (bb_upper + bb_lower) / 2)
    bb_width = indicators.get('BB_Width', 0)
    
    if price <= bb_lower:
        buy_points += 2
        reasoning.append("✅ السعر عند الحد الأدنى لـ Bollinger (فرصة شراء)")
    elif price >= bb_upper:
        sell_points += 2
        reasoning.append("❌ السعر عند الحد الأعلى لـ Bollinger (فرصة بيع)")
    
    if bb_width > 5:  # تقلبات عالية
        reasoning.append("⚠️ تقلبات عالية (BB Width > 5%)")
    
    # تحليل الحجم
    volume_ratio = indicators['Volume_Ratio']
    if volume_ratio > 2:
        if buy_points > sell_points:
            buy_points += 2
            reasoning.append("✅ حجم تداول عالي جداً + إشارات شراء (تأكيد قوي)")
        elif sell_points > buy_points:
            sell_points += 2
            reasoning.append("❌ حجم تداول عالي جداً + إشارات بيع (تأكيد قوي)")
    elif volume_ratio > 1.5:
        reasoning.append("🟡 حجم تداول أعلى من المتوسط")
    
    # تحليل Support/Resistance
    support = indicators['Support']
    resistance = indicators['Resistance']
    distance_to_support = ((price - support) / price) * 100
    distance_to_resistance = ((resistance - price) / price) * 100
    
    if distance_to_support < 2:
        buy_points += 2
        reasoning.append("✅ السعر قريب من Support (فرصة شراء)")
    elif distance_to_resistance < 2:
        sell_points += 1
        reasoning.append("🟡 السعر قريب من Resistance (احذر)")
    
    # تحليل Stochastic
    stoch_k = indicators.get('Stoch_K', 50)
    stoch_d = indicators.get('Stoch_D', 50)
    if stoch_k < 20 and stoch_k > stoch_d:
        buy_points += 1
        reasoning.append("🟡 Stochastic في منطقة ذروة البيع")
    elif stoch_k > 80 and stoch_k < stoch_d:
        sell_points += 1
        reasoning.append("🟡 Stochastic في منطقة ذروة الشراء")
    
    # السعر مقابل المتوسطات الطويلة
    ema_200 = indicators.get('EMA_200', price)
    sma_50 = indicators.get('SMA_50', price)
    if price > ema_200 and price > sma_50:
        buy_points += 2
        reasoning.append("✅ السعر فوق EMA200 و SMA50 (اتجاه صاعد قوي)")
    elif price < ema_200 and price < sma_50:
        sell_points += 2
        reasoning.append("❌ السعر تحت EMA200 و SMA50 (اتجاه هابط قوي)")
    
    # تحديد الإشارة
    confidence = min(100, ((buy_points + sell_points) / 20) * 100)
    
    if buy_points - sell_points >= 8:
        signal_type = SignalType.STRONG_BUY
    elif buy_points - sell_points >= 4:
        signal_type = SignalType.BUY
    elif buy_points - sell_points >= 1:
        signal_type = SignalType.WEAK_BUY
    elif sell_points - buy_points >= 8:
        signal_type = SignalType.STRONG_SELL
    elif sell_points - buy_points >= 4:
        signal_type = SignalType.SELL
    elif sell_points - buy_points >= 1:
        signal_type = SignalType.WEAK_SELL
    else:
        signal_type = SignalType.HOLD
        reasoning.append("⚪ إشارات متضاربة")
    
    # حساب نقاط الدخول والخروج
    atr = indicators['ATR']
    if signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
        entry_price = price
        stop_loss = price - (atr * 2)
        take_profit_1 = price + (atr * 1.5)
        take_profit_2 = price + (atr * 2.5)
        take_profit_3 = price + (atr * 4)
    elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL, SignalType.WEAK_SELL]:
        entry_price = price
        stop_loss = price + (atr * 2)
        take_profit_1 = price - (atr * 1.5)
        take_profit_2 = price - (atr * 2.5)
        take_profit_3 = price - (atr * 4)
    else:
        entry_price = stop_loss = take_profit_1 = take_profit_2 = take_profit_3 = price
    
    risk = abs(price - stop_loss)
    reward = abs(take_profit_2 - price)
    risk_reward = reward / risk if risk > 0 else 0
    
    return TradingSignal(
        signal_type=signal_type,
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        take_profit_3=take_profit_3,
        risk_reward=risk_reward,
        reasoning=reasoning,
        indicators=indicators
    )

# ✅ /start - ترحيب
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة ترحيب"""
    if not update.message:
        logger.warning("تحديث بدون رسالة في /start")
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    

    # ... الكود السابق ...
    
    msg = """<b>🤖 أهلاً بك في بوت التداول الذكي!</b>

━━━━━━━━━━━━━━━━━━━━
<b>📊 الأوامر المتاحة:</b>

<b>🔍 فحص وتحليل:</b>
/scan - فحص السوق لاكتشاف فرص الشراء
/analyze [العملة] - تحليل عملة محددة
/top - أكثر العملات تحركاً خلال 24 ساعة
/silent_moves - ضخ سيولة بدون حركة سعر

<b>🎯 إشارات التداول:</b>
/signal [العملة] - إشارة تداول احترافية
/signals_scan - فحص السوق لأفضل إشارات الشراء

<b>📋 قوائم:</b>
/watchlist - تحليل قائمة العملات المتابعة

<b>❓ مساعدة:</b>
/help - عرض دليل الاستخدام الكامل

━━━━━━━━━━━━━━━━━━━━
💡 مثال: <code>/signal BTC</code>
⚠️ <i>تحذير: تحليل تقني فقط وليس نصيحة استثمارية</i>

━━━━━━━━━━━━━━━━━━━━
المطور : @Up2205"""

    await update.message.reply_text(msg, parse_mode='HTML') # تغيير النوع هنا

# ✅ /analyze - تحليل يدوي لأي عملة
async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تحليل عملة محددة"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    if len(context.args) == 0:  # type: ignore
        await update.message.reply_text("اكتب اسم العملة بعد الأمر. مثال: /analyze BTC")
        return

    symbol = context.args[0].upper()  # type: ignore
    
    # ✅ إضافة USDT تلقائياً إذا لم يكتبها المستخدم
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    
    if not validate_symbol(symbol):
        await update.message.reply_text(f"⚠️ العملة {symbol} غير موجودة أو غير مدعومة.")
        return
    
    try:
        df = get_ohlcv(symbol)
        price_now = df['close'].iloc[-1]
        df['value'] = df['close'] * df['volume']
        volume_24h = df['value'][:-1].sum()
        
        # ✅ جلب التغير اليومي من Binance
        ticker = exchange.fetch_ticker(symbol)
        change_24h = ticker['percentage']
        highest_price = ticker['high']
        
        rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]

        msg = f"""📊 تحليل {symbol}:

💸 السعر الحالي: {price_now:,.4f}  

📈 أعلى سعر: {highest_price:.4f}

🧮 إجمالي حجم 24س: {volume_24h:,.2f}

📉 تغير السعر 24س: {change_24h:.2f}%  

📉 RSI: {rsi:.2f}  

"""
        await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"خطأ في تحليل {symbol}: {e}")
        await update.message.reply_text(f"⚠️ خطأ في تحليل {symbol}: {str(e)}")

# ✅ /top - العملات الأكثر تحركًا
async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """العملات الأكثر تحركاً"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    await update.message.reply_text("🚀 جاري تحليل أكثر العملات تحركاً خلال 24 ساعة...")

    user_id = update.effective_user.id
    STOP_SIGNALS[user_id] = False

    movers: List[Tuple[str, float]] = []
    symbols = get_symbols()

    # ✅ جلب البيانات مرة واحدة فقط
    for symbol in symbols:
        # ✅ التحقق من طلب الإيقاف
        if STOP_SIGNALS.get(user_id, False):
            await update.message.reply_text("🛑 تم إيقاف التحليل.")
            return

        try:
            df = get_ohlcv(symbol)
            change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            if abs(change) > 3:
                movers.append((symbol, change))
        except Exception as e:
            logger.debug(f"خطأ في {symbol}: {e}")
            continue

    movers = sorted(movers, key=lambda x: abs(x[1]), reverse=True)[:10]

    if not movers:
        await update.message.reply_text("❌ لا توجد عملات متحركة حالياً.")
        return

    # ✅ تجميع الرسائل لتجنب spam
    messages = []
    for symbol, change in movers:
        try:
            df = get_ohlcv(symbol)

            price_now = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-2]
            price_change_1h = ((price_now - price_prev) / price_prev) * 100

            df['value'] = df['close'] * df['volume']
            usd_volume_24h = df['value'][:-1].sum()
            volume_now = df['volume'].iloc[-1]
            volume_24h = df['volume'][:-1].sum()
            
            ticker = exchange.fetch_ticker(symbol)
            change_24h = ticker['percentage']
            highest_price = ticker['high']

            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]

            msg = f"""📊 تحليل {symbol} (Top Mover):
💸 السعر الحالي: {price_now:.4f}
💸 أعلى سعر: {highest_price:.4f}
📈 تغير 24 ساعة: {change_24h:.2f}%
📉 تغير آخر ساعة: {price_change_1h:.2f}%
📊 حجم التداول (ساعة): {volume_now:.2f}
🧮 إجمالي حجم التداول 24h: {volume_24h:.2f}
💰 القيمة بالدولار: ${usd_volume_24h:,.2f}

📉 RSI: {rsi:.2f}
"""
            messages.append(msg)

        except Exception as e:
            logger.error(f"خطأ في تحليل {symbol}: {e}")
            continue

    # ✅ إرسال الرسائل
    for msg in messages:
        await update.message.reply_text(msg)

# ✅ /silent_moves - عملات فيها ضخ بدون تحرك سعري
async def silent_moves(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """العملات التي فيها ضخ سيولة بدون تحرك سعري"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    await update.message.reply_text("🔍 نبحث عن ضخ سيولة بدون حركة سعر...")

    user_id = update.effective_user.id
    STOP_SIGNALS[user_id] = False

    matches = False
    symbols = get_symbols()

    for symbol in symbols:
        # ✅ التحقق من طلب الإيقاف
        if STOP_SIGNALS.get(user_id, False):
            await update.message.reply_text("🛑 تم إيقاف البحث.")
            return

        try:
            df = get_ohlcv(symbol)

            volume_now = df['volume'].iloc[-1]
            volume_avg = df['volume'][:-1].mean()
            volume_change = ((volume_now - volume_avg) / volume_avg) * 100

            price_now = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-2]
            price_change = ((price_now - price_prev) / price_prev) * 100

            if volume_change > 80 and abs(price_change) < 1:
                df['value'] = df['close'] * df['volume']
                usd_volume_24h = df['value'][:-1].sum()
                rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
                ticker = exchange.fetch_ticker(symbol)
                change_24h = ticker['percentage']
                
                msg = f"""🕵️ {symbol} - سيولة بدون تحرك واضح
💸 السعر الحالي: {price_now:.4f}
📉 تغير آخر ساعة: {price_change:.2f}%
📈 تغير الحجم: {volume_change:.2f}%
📊 حجم الساعة: {volume_now:.2f}
🧮 حجم 24h: {df['volume'][:-1].sum():.2f}
💰 القيمة بالدولار 24h: ${usd_volume_24h:,.2f}
📉 RSI: {rsi:.2f}
"""
                await update.message.reply_text(msg)
                matches = True

        except Exception as e:
            logger.debug(f"خطأ في {symbol}: {e}")
            continue

    if not matches:
        await update.message.reply_text("❌ لا توجد عملات فيها ضخ سيولة بدون تحرك سعري.")

# ✅ /watchlist - تحليل قائمة العملات المخصصة
async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تحليل قائمة العملات المتابعة"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    await update.message.reply_text("📋 نحلل القائمة الخاصة بك...")
    signals = []

    for symbol in WATCHLIST:
        try:
            df = get_ohlcv(symbol)
            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
            volume_now = df['volume'].iloc[-1]
            volume_avg = df['volume'][:-1].mean()

            if rsi < 30 or volume_now > volume_avg * 2:
                signals.append(f"✅ {symbol}: RSI {rsi:.1f}, حجم {volume_now:.0f}")
        except Exception as e:
            logger.debug(f"خطأ في {symbol}: {e}")
            continue

    msg = "📡 إشارات من قائمتك:\n\n" + "\n".join(signals) if signals else "📭 لا توجد إشارات حالياً."
    await update.message.reply_text(msg)

# ✅ /scan - فحص السوق للفرص (محسّن)
async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """فحص السوق للفرص باستخدام نظام الإشارات الاحترافي"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    await update.message.reply_text("🔎 جاري فحص السوق بحثًا عن أفضل الفرص...\n⏳ قد يستغرق هذا بضع دقائق...")

    user_id = update.effective_user.id
    STOP_SIGNALS[user_id] = False

    try:
        symbols = get_symbols()[:150]  # فحص أول 150 عملة (للأداء)
        
        signals_found = []
        processed = 0
        
        for symbol in symbols:
            # ✅ التحقق من طلب الإيقاف
            if STOP_SIGNALS.get(user_id, False):
                await update.message.reply_text("🛑 تم إيقاف الفحص بناءً على طلبك.")
                return

            try:
                # ✅ استخدام نظام الإشارات الاحترافي
                df = get_ohlcv(symbol, timeframe='1h', limit=200)
                indicators = calculate_advanced_indicators(df)
                signal = analyze_professional_signal(df, indicators)
                
                # ✅ فقط الإشارات القوية والمتوسطة (ليس WEAK أو HOLD)
                if signal.signal_type in [
                    SignalType.STRONG_BUY, SignalType.BUY,
                    SignalType.STRONG_SELL, SignalType.SELL
                ]:
                    # ✅ جلب حجم التداول 24 ساعة
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        volume_24h_usd = ticker.get('quoteVolume', 0)
                        indicators['Volume_24h_USD'] = volume_24h_usd
                    except:
                        indicators['Volume_24h_USD'] = 0
                    
                    # ✅ التحقق من جودة الفرصة (مع فلترة التقلبات)
                    is_good, warnings = is_good_trading_opportunity(
                        signal, indicators,
                        min_volume_ratio=0.6,      # أقل صرامة للفحص الشامل
                        min_risk_reward=0.9,       # أقل صرامة للفحص الشامل
                        max_volatility=7.0,        # استبعاد العملات بتقلبات > 7%
                        min_volume_24h=500000      # حد أدنى $500K حجم تداول
                    )
                    
                    # ✅ حساب مستوى الضمان
                    guarantee_level, guarantee_score = calculate_guarantee_level(signal, indicators)
                    
                    # ✅ فلترة: فقط الفرص التي مستوى ضمانها جيد أو أفضل (score >= 45)
                    if guarantee_score >= 45:
                        # ✅ تحديد نوع الفرصة
                        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                            opportunity_type = "✅ إشارة مؤكدة"
                        elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                            opportunity_type = "📊 إشارة جيدة"
                        else:
                            opportunity_type = "📢 إشارة مبكرة"
                        
                        # ✅ جلب معلومات إضافية
                        ticker = exchange.fetch_ticker(symbol)
                        change_24h = ticker['percentage']
                        volume_24h = ticker.get('quoteVolume', 0)
                        
                        # ✅ حساب نقاط الجودة (Quality Score)
                        quality_score = guarantee_score
                        if signal.risk_reward >= 2.0:
                            quality_score += 5
                        if indicators.get('Volume_Ratio', 1) >= 1.5:
                            quality_score += 3
                        if is_good:
                            quality_score += 5
                        
                        signals_found.append((
                            symbol, signal, indicators, quality_score,
                            is_good, warnings, guarantee_level, guarantee_score,
                            opportunity_type, change_24h, volume_24h
                        ))
                        
            except Exception as e:
                logger.debug(f"خطأ في {symbol}: {e}")
                continue
            
            processed += 1
            if processed % 30 == 0:
                await update.message.reply_text(f"⏳ تم فحص {processed} عملة...")
        
        if not signals_found:
            await update.message.reply_text("📭 لا توجد إشارات قوية حالياً.\n💡 جرب الأمر /signals_scan للبحث عن فرص أخرى")
            return
        
        # ✅ ترتيب حسب نقاط الجودة (الأفضل أولاً)
        signals_found.sort(key=lambda x: x[3], reverse=True)
        signals_found = signals_found[:25]  # أفضل 25
        
        # ✅ فصل إشارات الشراء والبيع
        buy_signals = [s for s in signals_found if s[1].signal_type in [SignalType.STRONG_BUY, SignalType.BUY]]
        sell_signals = [s for s in signals_found if s[1].signal_type in [SignalType.STRONG_SELL, SignalType.SELL]]
        
        # ✅ بناء الرسالة
        msg = "🎯 *نتائج فحص السوق:*\n\n"
        msg += f"📊 تم فحص {processed} عملة ووجدنا {len(signals_found)} فرصة جيدة\n"
        msg += f"🟢 فرص شراء: {len(buy_signals)} | 🔴 فرص بيع: {len(sell_signals)}\n\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # ✅ عرض فرص الشراء
        if buy_signals:
            msg += "🟢 *أفضل فرص الشراء:*\n\n"
            for i, (symbol, signal, indicators, quality_score, is_good, warnings, 
                    guarantee_level, guarantee_score, opportunity_type, change_24h, volume_24h) in enumerate(buy_signals[:12], 1):
                
                msg += f"{guarantee_level} *{i}. {symbol}*\n"
                msg += f"   {opportunity_type} - {signal.signal_type.value}\n"
                msg += f"   🎯 مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg += f"   💰 السعر: {signal.entry_price:,.4f} USDT\n"
                msg += f"   📈 تغير 24س: {change_24h:.2f}%\n"
                msg += f"   📊 Risk/Reward: 1:{signal.risk_reward:.2f} "
                msg += f"{'✅ ممتاز' if signal.risk_reward >= 2.0 else '✅ جيد' if signal.risk_reward >= 1.5 else '⚠️ متوسط'}\n"
                msg += f"   🛑 SL: {signal.stop_loss:,.4f} | ✅ TP2: {signal.take_profit_2:,.4f}\n"
                msg += f"   📊 RSI: {indicators['RSI']:.2f} | ADX: {indicators['ADX']:.2f}\n\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # ✅ عرض فرص البيع
        if sell_signals:
            msg += "🔴 *أفضل فرص البيع:*\n\n"
            for i, (symbol, signal, indicators, quality_score, is_good, warnings,
                    guarantee_level, guarantee_score, opportunity_type, change_24h, volume_24h) in enumerate(sell_signals[:12], 1):
                
                msg += f"{guarantee_level} *{i}. {symbol}*\n"
                msg += f"   {opportunity_type} - {signal.signal_type.value}\n"
                msg += f"   🎯 مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg += f"   💰 السعر: {signal.entry_price:,.4f} USDT\n"
                msg += f"   📈 تغير 24س: {change_24h:.2f}%\n"
                msg += f"   📊 Risk/Reward: 1:{signal.risk_reward:.2f} "
                msg += f"{'✅ ممتاز' if signal.risk_reward >= 2.0 else '✅ جيد' if signal.risk_reward >= 1.5 else '⚠️ متوسط'}\n"
                msg += f"   🛑 SL: {signal.stop_loss:,.4f} | ✅ TP2: {signal.take_profit_2:,.4f}\n"
                msg += f"   📊 RSI: {indicators['RSI']:.2f} | ADX: {indicators['ADX']:.2f}\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += "💡 *نصائح للاستخدام:*\n"
        msg += "• استخدم /signal [العملة] للحصول على تحليل مفصل\n"
        msg += "• 🟢 مضمون جداً = فرصة ممتازة (80%+)\n"
        msg += "• 🟢 مضمون = فرصة جيدة جداً (65%+)\n"
        msg += "• 🟡 جيد = فرصة جيدة (50%+)\n"
        msg += "• استخدم Stop Loss دائماً ولا تخاطر بأكثر من 2-5%\n"
        msg += "• تحليل تقني فقط وليس نصيحة استثمارية"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
        # ✅ إرسال أفضل الفرص المضمونة بشكل مفصل
        high_guarantee = [s for s in signals_found if s[7] >= 75]  # مستوى ضمان 75%+
        if high_guarantee:
            msg_detail = "🟢 *أفضل الفرص المضمونة جداً:*\n\n"
            for symbol, signal, indicators, quality_score, is_good, warnings, guarantee_level, guarantee_score, opportunity_type, change_24h, volume_24h in high_guarantee[:5]:
                msg_detail += f"✅ *{symbol}* - {signal.signal_type.value}\n"
                msg_detail += f"   مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg_detail += f"   السعر: {signal.entry_price:,.4f} | R/R: 1:{signal.risk_reward:.2f}\n"
                msg_detail += f"   SL: {signal.stop_loss:,.4f} | TP2: {signal.take_profit_2:,.4f}\n\n"
            msg_detail += "💡 استخدم /signal [العملة] للحصول على تحليل مفصل"
            await update.message.reply_text(msg_detail, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"خطأ في /scan: {e}")
        await update.message.reply_text(f"⚠️ خطأ: {str(e)}")

# ✅ /signal - إشارة تداول احترافية مباشرة
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إشارة تداول احترافية: متى تشتري ومتى تبيع"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    if len(context.args) == 0:  # type: ignore
        await update.message.reply_text("📊 استخدم: /signal BTC")
        return
    
    symbol = context.args[0].upper()  # type: ignore
    
    # ✅ إضافة USDT تلقائياً إذا لم يكتبها المستخدم
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    
    if not validate_symbol(symbol):
        await update.message.reply_text(f"⚠️ العملة {symbol} غير موجودة.")
        return

    try:
        await update.message.reply_text(f"🔍 جاري تحليل {symbol}...")
        
        df = get_ohlcv(symbol, timeframe='1h', limit=200)  # زيادة البيانات لتحليل أفضل
        indicators = calculate_advanced_indicators(df)
        signal = analyze_professional_signal(df, indicators)
        
        # ✅ جلب حجم التداول 24 ساعة
        ticker = exchange.fetch_ticker(symbol)
        change_24h = ticker['percentage']
        volume_24h = ticker.get('quoteVolume', 0)
        indicators['Volume_24h_USD'] = volume_24h
        
        # ✅ التحقق من جودة الفرصة (مع فلترة التقلبات)
        is_good, warnings = is_good_trading_opportunity(
            signal, indicators,
            min_volume_ratio=0.8,
            min_risk_reward=1.2,
            max_volatility=7.0,        # استبعاد العملات بتقلبات > 7%
            min_volume_24h=500000      # حد أدنى $500K حجم تداول
        )
        
        # ✅ حساب مستوى الضمان
        guarantee_level, guarantee_score = calculate_guarantee_level(signal, indicators)
        
        # ✅ تحديد متى يشتري/يبيع بشكل واضح
        action_advice = ""
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            action_advice = f"""
🟢 *متى تشتري:*
• ✅ اشترِ الآن عند السعر: {signal.entry_price:,.4f} USDT
• ✅ أو انتظر انخفاض بسيط إلى: {signal.entry_price * 0.995:,.4f} USDT (شراء أفضل)
• ⚠️ لا تشتري إذا ارتفع السعر أكثر من 2% من سعر الدخول

🛑 *متى تبيع (Stop Loss):*
• ❌ بيع فوري إذا وصل السعر إلى: {signal.stop_loss:,.4f} USDT
• 💰 هذا يحدد خسارتك القصوى

✅ *متى تأخذ الربح:*
• 🎯 TP1: {signal.take_profit_1:,.4f} USDT - بيع 30% من الكمية
• 🎯 TP2: {signal.take_profit_2:,.4f} USDT - بيع 50% من الكمية  
• 🎯 TP3: {signal.take_profit_3:,.4f} USDT - بيع 20% المتبقية"""
        elif signal.signal_type == SignalType.WEAK_BUY:
            action_advice = f"""
🟡 *إشارة شراء ضعيفة:*
• ⚠️ انتظر تأكيد إضافي قبل الشراء
• 🔍 راقب السعر عند: {signal.entry_price:,.4f} USDT
• ✅ اشترِ فقط إذا تحسنت المؤشرات"""
        elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            action_advice = f"""
🔴 *متى تبيع:*
• ✅ بيع الآن عند السعر: {signal.entry_price:,.4f} USDT
• ✅ أو انتظر ارتفاع بسيط إلى: {signal.entry_price * 1.005:,.4f} USDT (بيع أفضل)
• ⚠️ لا تبيع إذا انخفض السعر أكثر من 2% من سعر الدخول

🛑 *متى تغلق (Stop Loss):*
• ❌ إغلاق فوري إذا وصل السعر إلى: {signal.stop_loss:,.4f} USDT
• 💰 هذا يحدد خسارتك القصوى"""
        elif signal.signal_type == SignalType.WEAK_SELL:
            action_advice = f"""
🟡 *إشارة بيع ضعيفة:*
• ⚠️ انتظر تأكيد إضافي قبل البيع
• 🔍 راقب السعر عند: {signal.entry_price:,.4f} USDT"""
        else:
            action_advice = """
⚪ *لا توجد إشارة واضحة:*
• ⏸️ انتظر حتى تظهر إشارة أوضح
• 🔍 راقب السوق ولا تتداول الآن"""
        
        # ✅ تحليل الاتجاه العام
        ema_200 = indicators.get('EMA_200', indicators['Price'])
        trend_analysis = ""
        if indicators['Price'] > ema_200:
            trend_analysis = "📈 *الاتجاه العام: صاعد* (السعر فوق EMA200)"
        else:
            trend_analysis = "📉 *الاتجاه العام: هابط* (السعر تحت EMA200)"
        
        msg = f"""
🎯 *إشارة تداول احترافية - {symbol}*

━━━━━━━━━━━━━━━━━━━━
📊 *الإشارة: {signal.signal_type.value}*
🎯 *مستوى الثقة: {signal.confidence:.1f}%*
🛡️ *مستوى الضمان: {guarantee_level} ({guarantee_score:.0f}/100)*
{'✅ فرصة جيدة' if is_good else '⚠️ فرصة تحتاج حذر'}

━━━━━━━━━━━━━━━━━━━━
💰 *السعر الحالي:* {indicators['Price']:,.4f} USDT
📈 *تغير 24س:* {change_24h:.2f}%
💵 *حجم التداول 24س:* ${volume_24h:,.0f} USDT
{trend_analysis}

━━━━━━━━━━━━━━━━━━━━
{action_advice}

━━━━━━━━━━━━━━━━━━━━
📊 *المؤشرات الفنية:*
• RSI: {indicators['RSI']:.2f} {'(ذروة بيع)' if indicators['RSI'] < 30 else '(ذروة شراء)' if indicators['RSI'] > 70 else ''}
• MACD: {indicators['MACD']:.4f} | Signal: {indicators['MACD_Signal']:.4f}
• ADX: {indicators['ADX']:.2f} {'(اتجاه قوي)' if indicators['ADX'] > 25 else '(اتجاه ضعيف)'}
• EMA9: {indicators['EMA_9']:,.4f}
• EMA21: {indicators['EMA_21']:,.4f}
• EMA50: {indicators['EMA_50']:,.4f}
• EMA200: {indicators['EMA_200']:,.4f}
• Volume Ratio: {indicators['Volume_Ratio']:.2f}x {'(حجم عالي)' if indicators['Volume_Ratio'] > 1.5 else '(حجم عادي)'}
• Support: {indicators['Support']:,.4f} | Resistance: {indicators['Resistance']:,.4f}

━━━━━━━━━━━━━━━━━━━━
📊 *نقاط الدخول والخروج:*
📍 *سعر الدخول:* {signal.entry_price:,.4f} USDT
🛑 *Stop Loss:* {signal.stop_loss:,.4f} USDT
   (خسارة محتملة: {abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):.2f}%)

✅ *Take Profit 1:* {signal.take_profit_1:,.4f} USDT
   (ربح: +{abs((signal.take_profit_1 - signal.entry_price) / signal.entry_price * 100):.2f}%)

✅ *Take Profit 2:* {signal.take_profit_2:,.4f} USDT
   (ربح: +{abs((signal.take_profit_2 - signal.entry_price) / signal.entry_price * 100):.2f}%)

✅ *Take Profit 3:* {signal.take_profit_3:,.4f} USDT
   (ربح: +{abs((signal.take_profit_3 - signal.entry_price) / signal.entry_price * 100):.2f}%)

📊 *Risk/Reward Ratio:* 1:{signal.risk_reward:.2f}
{'✅ جيد' if signal.risk_reward >= 1.5 else '⚠️ متوسط' if signal.risk_reward >= 1.2 else '❌ ضعيف'}

━━━━━━━━━━━━━━━━━━━━
💡 *التحليل التفصيلي:*
"""
        for reason in signal.reasoning:
            msg += f"{reason}\n"
        
        # ✅ إضافة التحذيرات
        if warnings:
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n⚠️ *تحذيرات مهمة:*\n"
            for warning in warnings:
                msg += f"{warning}\n"
        
        msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += "⚠️ *تحذير:* تحليل تقني فقط وليس نصيحة استثمارية\n"
        msg += "💡 *نصيحة:* استخدم Stop Loss دائماً ولا تخاطر بأكثر من 2-5% من رأس المال"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"خطأ في /signal {symbol}: {e}")
        await update.message.reply_text(f"⚠️ خطأ: {str(e)}")

# ✅ /signals_scan - فحص السوق لأفضل إشارات الشراء
async def signals_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """فحص جميع العملات وإيجاد أفضل الإشارات"""
    if not update.message:
        return

    # ✅ التحقق من الصلاحية
    if not await check_auth(update):
        return
    
    await update.message.reply_text("🔍 جاري فحص السوق للبحث عن أفضل فرص الشراء والبيع...\n⏳ قد يستغرق هذا بضع دقائق...")
    
    user_id = update.effective_user.id
    STOP_SIGNALS[user_id] = False

    try:
        markets = exchange.load_markets()
        # ✅ فلترة أفضل: العملات بحجم تداول عالي أولاً
        all_symbols = [
            s for s in markets.values() 
            if s['quote'] == 'USDT' and s['spot'] and s['active']
        ]
        
        # ✅ ترتيب حسب حجم التداول (إن أمكن) أو أخذ أول 100 عملة
        symbols = [s['symbol'] for s in all_symbols[:100]]
        
        signals_found = []
        processed = 0
        
        for symbol in symbols:
            # ✅ التحقق من طلب الإيقاف
            if STOP_SIGNALS.get(user_id, False):
                await update.message.reply_text("🛑 تم إيقاف فحص الإشارات.")
                return

            try:
                df = get_ohlcv(symbol, timeframe='1h', limit=200)
                indicators = calculate_advanced_indicators(df)
                signal = analyze_professional_signal(df, indicators)
                
                # ✅ إشارات شراء وبيع قوية أو متوسطة
                if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.STRONG_SELL, SignalType.SELL]:
                    # ✅ جلب حجم التداول 24 ساعة
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        volume_24h_usd = ticker.get('quoteVolume', 0)
                        indicators['Volume_24h_USD'] = volume_24h_usd
                    except:
                        indicators['Volume_24h_USD'] = 0
                    
                    # ✅ التحقق من جودة الفرصة (مع فلترة التقلبات)
                    is_good, warnings = is_good_trading_opportunity(
                        signal, indicators, 
                        min_volume_ratio=0.7,      # أقل صرامة للفحص الشامل
                        min_risk_reward=1.0,       # أقل صرامة للفحص الشامل
                        max_volatility=7.0,        # استبعاد العملات بتقلبات > 7%
                        min_volume_24h=500000      # حد أدنى $500K حجم تداول
                    )
                    
                    # ✅ حساب مستوى الضمان
                    guarantee_level, guarantee_score = calculate_guarantee_level(signal, indicators)
                    
                    # ✅ فلترة: فقط الفرص التي مستوى ضمانها جيد أو أفضل (score >= 50)
                    if guarantee_score >= 50:
                        # ✅ حساب نقاط الجودة (Score) - استخدام مستوى الضمان
                        quality_score = guarantee_score
                        
                        # ✅ إضافة نقاط إضافية للجودة
                        if signal.risk_reward >= 2.0:
                            quality_score += 5
                        
                        if indicators.get('Volume_Ratio', 1) >= 1.5:
                            quality_score += 3
                        
                        if indicators['Price'] > indicators.get('EMA_200', indicators['Price']) and signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                            quality_score += 3
                        elif indicators['Price'] < indicators.get('EMA_200', indicators['Price']) and signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                            quality_score += 3
                        
                        if is_good:
                            quality_score += 5
                        
                        signals_found.append((symbol, signal, indicators, quality_score, is_good, warnings, guarantee_level, guarantee_score))
                    
            except Exception as e:
                logger.debug(f"خطأ في {symbol}: {e}")
                continue
        
            processed += 1
            if processed % 20 == 0:
                await update.message.reply_text(f"⏳ تم فحص {processed} عملة...")
        
        if not signals_found:
            await update.message.reply_text("❌ لا توجد إشارات قوية حالياً.\n💡 جرب الأمر /scan للبحث عن فرص أخرى")
            return
        
        # ✅ ترتيب حسب نقاط الجودة (Quality Score) - الأفضل أولاً
        signals_found.sort(key=lambda x: x[3], reverse=True)
        signals_found = signals_found[:20]  # أفضل 20
        
        # ✅ فصل إشارات الشراء والبيع
        buy_signals = [s for s in signals_found if s[1].signal_type in [SignalType.STRONG_BUY, SignalType.BUY]]
        sell_signals = [s for s in signals_found if s[1].signal_type in [SignalType.STRONG_SELL, SignalType.SELL]]
        
        # ✅ تقسيم الرسائل
        msg = "🎯 *أفضل فرص التداول في السوق:*\n\n"
        msg += f"📊 تم فحص {processed} عملة ووجدنا {len(signals_found)} فرصة جيدة\n"
        msg += f"🟢 فرص شراء: {len(buy_signals)} | 🔴 فرص بيع: {len(sell_signals)}\n\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # ✅ عرض فرص الشراء أولاً
        if buy_signals:
            msg += "🟢 *أفضل فرص الشراء:*\n\n"
            for i, (symbol, signal, indicators, quality_score, is_good, warnings, guarantee_level, guarantee_score) in enumerate(buy_signals[:8], 1):
                msg += f"{guarantee_level} *{i}. {symbol}*\n"
                msg += f"   📊 الإشارة: {signal.signal_type.value}\n"
                msg += f"   🎯 مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg += f"   💰 السعر: {signal.entry_price:,.4f} USDT\n"
                msg += f"   📈 تغير 24س: {indicators.get('Price_Change_24h', 0):.2f}%\n"
                msg += f"   📊 Risk/Reward: 1:{signal.risk_reward:.2f} "
                msg += f"{'✅ ممتاز' if signal.risk_reward >= 2.0 else '✅ جيد' if signal.risk_reward >= 1.5 else '⚠️ متوسط'}\n"
                msg += f"   🛑 SL: {signal.stop_loss:,.4f} | ✅ TP2: {signal.take_profit_2:,.4f}\n\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # ✅ عرض فرص البيع
        if sell_signals:
            msg += "🔴 *أفضل فرص البيع:*\n\n"
            for i, (symbol, signal, indicators, quality_score, is_good, warnings, guarantee_level, guarantee_score) in enumerate(sell_signals[:8], 1):
                msg += f"{guarantee_level} *{i}. {symbol}*\n"
                msg += f"   📊 الإشارة: {signal.signal_type.value}\n"
                msg += f"   🎯 مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg += f"   💰 السعر: {signal.entry_price:,.4f} USDT\n"
                msg += f"   📈 تغير 24س: {indicators.get('Price_Change_24h', 0):.2f}%\n"
                msg += f"   📊 Risk/Reward: 1:{signal.risk_reward:.2f} "
                msg += f"{'✅ ممتاز' if signal.risk_reward >= 2.0 else '✅ جيد' if signal.risk_reward >= 1.5 else '⚠️ متوسط'}\n"
                msg += f"   🛑 SL: {signal.stop_loss:,.4f} | ✅ TP2: {signal.take_profit_2:,.4f}\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += "💡 *نصائح للاستخدام:*\n"
        msg += "• استخدم /signal [العملة] للحصول على تحليل مفصل\n"
        msg += "• 🟢 مضمون جداً = فرصة ممتازة (80%+)\n"
        msg += "• 🟢 مضمون = فرصة جيدة جداً (65%+)\n"
        msg += "• 🟡 جيد = فرصة جيدة (50%+)\n"
        msg += "• استخدم Stop Loss دائماً ولا تخاطر بأكثر من 2-5%\n"
        msg += "• تحليل تقني فقط وليس نصيحة استثمارية"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
        # ✅ إرسال أفضل الفرص المضمونة بشكل مفصل
        high_guarantee = [s for s in signals_found if s[7] >= 80]  # مستوى ضمان 80%+
        if high_guarantee:
            msg_detail = "🟢 *أفضل الفرص المضمونة جداً:*\n\n"
            for symbol, signal, indicators, quality_score, is_good, warnings, guarantee_level, guarantee_score in high_guarantee[:5]:
                msg_detail += f"✅ *{symbol}* - {signal.signal_type.value}\n"
                msg_detail += f"   مستوى الضمان: {guarantee_score:.0f}/100\n"
                msg_detail += f"   السعر: {signal.entry_price:,.4f} | R/R: 1:{signal.risk_reward:.2f}\n\n"
            msg_detail += "💡 استخدم /signal [العملة] للحصول على تحليل مفصل"
            await update.message.reply_text(msg_detail, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"خطأ في /signals_scan: {e}")
        await update.message.reply_text(f"⚠️ خطأ: {str(e)}")

# ✅ /help - مساعدة
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض المساعدة"""
    if not update.message:
        return
    
    msg = """🆘 مساعدة البوت - دليل الاستخدام الكامل 🤖

━━━━━━━━━━━━━━━━━━━━
📋 الأوامر الأساسية:

/start - عرض رسالة ترحيب وشرح عام
/help - عرض قائمة الأوامر (هذه الرسالة)

━━━━━━━━━━━━━━━━━━━━
🔍 أوامر الفحص والتحليل:

/scan
فحص السوق الآن لاكتشاف فرص الشراء
• يبحث عن إشارات مؤكدة ومبكرة
• يعرض أفضل الفرص في السوق

/analyze <العملة>
تحليل عملة يدويًا
• مثال: /analyze BTC
• يعرض: السعر، الحجم، RSI، التغيرات

/top
أكثر العملات تحركًا خلال 24 ساعة
• يعرض أفضل 10 عملات متحركة
• مع تحليل مفصل لكل عملة

/silent_moves
العملات التي فيها ضخ سيولة بدون تحرك سعري
• يكتشف فرص التجميع الخفية
• حجم عالي بدون حركة سعر واضحة

━━━━━━━━━━━━━━━━━━━━
🎯 أوامر الإشارات الاحترافية:

/signal <العملة>
إشارة تداول احترافية مباشرة
• مثال: /signal BTC
• يعطي: شراء قوي / بيع / احتفظ
• مع: نقاط الدخول، Stop Loss، Take Profit
• مستوى الثقة: 0-100%
• Risk/Reward Ratio

/signals_scan
فحص السوق لأفضل إشارات الشراء
• يفحص 50 عملة تلقائياً
• يعرض أفضل 10 إشارات شراء
• مرتبة حسب مستوى الثقة

━━━━━━━━━━━━━━━━━━━━
📋 أوامر القوائم:

/watchlist
تحليل قائمة العملات المتابعة
• يعرض إشارات من قائمتك الخاصة
• العملات: BTC, ETH, SOL, PEPE, ADA

━━━━━━━━━━━━━━━━━━━━
� أوامر التحكم:

/stop
إيقاف العمليات الجارية فوراً
• يوقف الفحص والتحليل

━━━━━━━━━━━━━━━━━━━━
�💡 معلومات إضافية:

✏️ كل الأرقام تطبع بدقة عالية
💰 مع فواصل عشرية ومبالغ بالدولار
✳️ البوت يعمل فقط على السوق Spot
📊 يستخدم مؤشرات فنية متقدمة:
   • RSI, MACD, EMA, ADX
   • Bollinger Bands, ATR
   • Volume Analysis

━━━━━━━━━━━━━━━━━━━━
⚠️ تحذيرات مهمة:

• هذا تحليل تقني فقط
• ليس نصيحة استثمارية
• لا يضمن الربح
• استخدم Stop Loss دائماً
• لا تخاطر بأكثر من 2-5% في صفقة

━━━━━━━━━━━━━━━━━━━━
🔧 التخصيص:

للتعديل على قائمة المتابعة:
عدّل WATCHLIST داخل الكود

━━━━━━━━━━━━━━━━━━━━
📞 أمثلة على الاستخدام:

/signal BTC
/analyze ETH
/signals_scan
/top


━━━━━━━━━━━━━━━━━━━━
المطور : @Up2205
"""
    await update.message.reply_text(msg)

async def setup_commands(app: Application):
    """إعداد أوامر البوت"""
    await app.bot.set_my_commands([
        BotCommand("start", "بدء استخدام البوت"),
        BotCommand("scan", "فحص السوق الآن لاكتشاف فرص الشراء 🔍"),
        BotCommand("analyze", "تحليل عملة محددة"),
        BotCommand("signal", "إشارة تداول احترافية مباشرة 🎯"),
        BotCommand("signals_scan", "فحص السوق لأفضل إشارات الشراء"),
        BotCommand("top", "أكثر العملات تحركاً"),
        BotCommand("silent_moves", "ضخ سيولة بدون تحرك سعر"),
        BotCommand("watchlist", "تحليل قائمة العملات الخاصة"),
        BotCommand("stop", "إيقاف العمليات الجارية 🛑"),
        BotCommand("help", "عرض المساعدة"),
    ])

def main():
    """تشغيل البوت"""
    if not TOKEN:
        logger.error("❌ TOKEN غير موجود! لا يمكن تشغيل البوت.")
        return
    
    # تهيئة قاعدة البيانات
    init_db()
    
    app = Application.builder().token(TOKEN).post_init(setup_commands).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("signals_scan", signals_scan))
    app.add_handler(CommandHandler("top", top))
    app.add_handler(CommandHandler("silent_moves", silent_moves))
    app.add_handler(CommandHandler("watchlist", watchlist))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("auth", auth_user))
    app.add_handler(CommandHandler("unauth", unauth_user))
    app.add_handler(CommandHandler("users", list_users))
    app.add_handler(CommandHandler("stop", stop_execution))
    app.add_handler(CommandHandler("help", help_command))
    
    logger.info("✅ البوت يعمل الآن...")
    
    # Webhook configuration for Render
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting webhook on port {port}")
    
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TOKEN,
        webhook_url=f"https://telegram-crpto-bot.onrender.com/{TOKEN}"
    )

if __name__ == "__main__":
    main()

