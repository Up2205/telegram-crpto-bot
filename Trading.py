import asyncio
import ccxt
import pandas as pd
import ta
import ta.momentum
import ta.trend
from telegram import BotCommand, Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# ✅ إعدادات تيليجرام
TOKEN = ''
bot = Bot(token=TOKEN)

# ✅ إعداد Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 30000,
})

# ✅ قائمة العملات اللي تبي تراقبها فقط
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT","ADA/USDT"]

# ✅ /start - ترحيب
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """🤖 أهلاً بك في بوت التداول الذكي!  
الأوامر المتاحة:

/scan - فحص السوق الآن لاكتشاف فرص الشراء 🔍   
/analyze BTC/USDT - تحليل عملة  
/top - أكثر العملات تحركاً  
/silent_moves - ضخ سيولة بدون حركة سعر  
/watchlist - تحليل قائمة عملاتك المختارة  
"""
    if update.message:
        await update.message.reply_text(msg)
    else:
        print("❌ لا يوجد رسالة في هذا التحديث")

# ✅ /analyze - تحليل يدوي لأي عملة
async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0: # type: ignore
        await update.message.reply_text("اكتب اسم العملة بعد الأمر. مثال: /analyze BTC/USDT") # type: ignore
        return

    symbol = context.args[0].upper() # type: ignore
    try:
        df = get_ohlcv(symbol)
        price_now = df['close'].iloc[-1]
        df['value'] = df['close'] * df['volume']
        volume_24h = df['value'][:-1].sum()
    # ✅ هنا نجيب التغير اليومي من Binance مباشرة
        ticker = exchange.fetch_ticker(symbol)
        change_24h = ticker['percentage']
        hegerprice=ticker['high']
            
        rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]

        msg = f"""📊 تحليل {symbol}:

💸 السعرالحالي: {price_now:,.4f}  

📈 اعلى سعر: {hegerprice:.2f}

🧮 إجمالي حجم 24س: {volume_24h:,.2f}

📉 تغير الحجم: {change_24h:.2f}٪  

📉 RSI: {rsi:.2f}  

"""
        await update.message.reply_text(msg) # type: ignore

    except Exception as e:
        await update.message.reply_text(f"⚠️ خطأ في تحليل {symbol}: {str(e)}") # type: ignore

# ✅ /top - العملات الأكثر تحركًا
async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 جاري تحليل أكثر العملات تحركاً خلال 24 ساعة...") # type: ignore

    movers = []

    for symbol in get_symbols():
        try:
            df = get_ohlcv(symbol)
            change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            if abs(change) > 3:
                movers.append((symbol, change))
        except:
            continue

    movers = sorted(movers, key=lambda x: abs(x[1]), reverse=True)[:10]  # خذ أقوى 5 فقط

    if not movers:
        await update.message.reply_text("❌ لا توجد عملات متحركة حالياً.") # type: ignore
        return

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
            hegerprice=ticker['high']

            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]

            msg = f"""📊 تحليل {symbol} (Top Mover):
💸 السعر الحالي: {price_now:.4f}
💸 اعلى سعر: {hegerprice:.4f}
📈 تغير 24 ساعة: {change_24h:.2f}%
📉 تغير آخر ساعة: {price_change_1h:.2f}%
📊 حجم التداول (ساعة): {hegerprice:.2f}
🧮 إجمالي حجم التداول 24h: {volume_24h:.2f}
💰 القيمة بالدولار: ${usd_volume_24h:,.2f}

📉 RSI: {rsi:.2f}
"""
            await update.message.reply_text(msg) # type: ignore

        except Exception as e:
            await update.message.reply_text(f"⚠️ خطأ في تحليل {symbol}: {str(e)}") # type: ignore


# ✅ /silent_moves - عملات فيها ضخ بدون تحرك سعري
async def silent_moves(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 نبحث عن ضخ سيولة بدون حركة سعر...") # type: ignore

    matches = False

    for symbol in get_symbols():
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
📈 تغير الحجم: {change_24h:.2f}%
📊 حجم الساعة: {volume_now:.2f}
🧮 حجم 24h: {df['volume'][:-1].sum():.2f}
💰 القيمة بالدولار 24h: ${usd_volume_24h:,.2f}
📉 RSI: {rsi:.2f}
"""
                await update.message.reply_text(msg) # type: ignore
                matches = True

        except:
            continue

    if not matches:
        await update.message.reply_text("❌ لا توجد عملات فيها ضخ سيولة بدون تحرك سعري.") # type: ignore


# ✅ /watchlist - تحليل قائمة العملات المخصصة
async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📋 نحلل القائمة الخاصة بك...") # type: ignore
    signals = []

    for symbol in WATCHLIST:
        try:
            df = get_ohlcv(symbol)
            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
            volume_now = df['volume'].iloc[-1]
            volume_avg = df['volume'][:-1].mean()

            if rsi < 30 or volume_now > volume_avg * 2:
                signals.append(f"✅ {symbol}: RSI {rsi:.1f}, حجم {volume_now:.0f}")
        except:
            continue

    msg = "📡 إشارات من قائمتك:\n\n" + "\n".join(signals) if signals else "📭 لا توجد إشارات حالياً."
    await update.message.reply_text(msg) # type: ignore

# ✅ /scan - فحص السوق للفرص
async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 جاري فحص السوق بحثًا عن فرص مؤكدة وغير مؤكدة...") # type: ignore

    found = False  # عشان نعرف إذا في أي إشارة

    for symbol in get_symbols():
        try:
            df = get_ohlcv(symbol)

            # الأسعار
            price_now = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-2]
            price_change = ((price_now - price_prev) / price_prev) * 100

            # الأحجام
            volume_now = df['volume'].iloc[-1]
            volume_avg = df['volume'][:-1].mean()
            volume_change = ((volume_now - volume_avg) / volume_avg) * 100

            # المؤشرات الفنية
            rsi = ta.momentum.RSIIndicator(close=df['close']).rsi().iloc[-1]
            ema9 = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator().iloc[-1]
            ema21 = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator().iloc[-1]
            macd_hist = ta.trend.MACD(close=df['close']).macd_diff().iloc[-1]
            if pd.isna(macd_hist): macd_hist = 0.0
            resistance_broken = price_now > df['high'].iloc[-2]

            # القيمة بالدولار
            df['value'] = df['close'] * df['volume']
            usd_volume_24h = df['value'][:-1].sum()
            ticker = exchange.fetch_ticker(symbol)
            change_24h = ticker['percentage']
            hegerprice=ticker['high']

            # ========== ✅ إشارة مؤكدة ==========
            if (
                volume_now > volume_avg * 3 and
                resistance_broken and
                ema9 > ema21
            ):
                msg = f"""✅ إشارة مؤكدة ({symbol})
💸 السعر: {price_now:.4f}
💸 اعلى سعر: {hegerprice:.4f}
📈 تغير الساعة: {price_change:.2f}%
📊 حجم التغير 24الساعة: {change_24h:.2f}
📉 RSI: {rsi:.2f}
📈 EMA9 > EMA21 ✅
📈 كسر مقاومة ✅
💰 حجم التداول بالدولار (24h): ${usd_volume_24h:,.2f}
"""
                await update.message.reply_text(msg) # type: ignore
                found = True

            # ========== 📢 إشارة مبكرة ==========
            elif price_change > 3 or volume_change > 100:
                msg = f"""📢 تنبيه مبكر ({symbol})
💸 السعر: {price_now:.4f}
💸 اعلى سعر: {hegerprice:.4f}
📈 تغير الساعة: {price_change:.2f}%  
📊 حجم 24الساعة: {change_24h:.2f}
📉 RSI: {rsi:.2f}
📊 MACD Histogram: {macd_hist:.4f}
💰 حجم التداول بالدولار (24h): ${usd_volume_24h:,.2f}
"""
                await update.message.reply_text(msg) # type: ignore
                found = True

        except:
            continue

    if not found:
        await update.message.reply_text("📭 لا توجد إشارات حالياً.") # type: ignore

# ✅ /daily - موجز ذكي
async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📆 موجز ذكي قادم...") # type: ignore
    await top(update, context)
    await silent_moves(update, context)
    await watchlist(update, context)

# ✅ /help - 
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
🆘 *مساعدة البوت - دليل الاستخدام* 🤖

إليك قائمة بالأوامر المتاحة:

/start – عرض رسالة ترحيب وشرح عام  
/help – عرض قائمة الأوامر (هذي الرسالة)  
/scan – فحص السوق الآن لاكتشاف فرص الشراء 🔍  
/analyze <العملة> – تحليل عملة يدويًا (مثال: /analyze BTC/USDT) 📊  
/top – عرض أكثر العملات تحركًا خلال 24 ساعة 🚀  
/silent_moves – العملات اللي زاد فيها حجم التداول بدون تحرك سعري (فرص تجميع خفية) 🕵️  
/watchlist – فحص العملات اللي انت مختارها فقط (قائمة خاصة بك) 📋  
✏️ كل الأرقام تطبع بدقة عالية مع فواصل عشرية ومبالغ بالدولار 💰  
✳️ البوت يعمل فقط على السوق *الحلال (Spot)*

💡 للتعديل على قائمة المتابعة، عدّل `WATCHLIST` داخل الكود.
    """
    await update.message.reply_text(msg) # type: ignore

# ✅ أدوات مساعدة
def get_symbols():
    return [s['symbol'] for s in exchange.load_markets().values() if s['quote'] == 'USDT' and s['spot'] and s['active']]

def get_ohlcv(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=25)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def setup_commands(app):
    app.bot.set_my_commands([
        BotCommand("start", "بدء استخدام البوت"),
        BotCommand("scan", "فحص السوق الآن لاكتشاف فرص الشراء 🔍"),
        BotCommand("analyze", "تحليل عملة محددة"),
        BotCommand("top", "أكثر العملات تحركاً"),
        BotCommand("silent_moves", "ضخ سيولة بدون تحرك سعر"),
        BotCommand("watchlist", "تحليل قائمة العملات الخاصة"),
    ])

app = Application.builder().token(TOKEN).build()
# ✅ تشغيل البوت
def main():
     
    setup_commands(app)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("top", top))
    app.add_handler(CommandHandler("silent_moves", silent_moves))
    app.add_handler(CommandHandler("watchlist", watchlist))
    app.add_handler(CommandHandler("scan", scan))
    # app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("help", help_command))
    print("✅ البوت يعمل الآن...")
    app.run_polling() 

if __name__ == "__main__":
    main()
