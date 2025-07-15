import os
import logging
import argparse
import threading
import optuna
import csv
import asyncio
import ccxt
import concurrent.futures
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import telegram
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
import numpy as np
import psutil
from numba import njit

# --- Configuration & Initialization ---

from keys import api_mainnet, secret_mainnet, telegram_bot_token, telegram_chat_id

# Binance API credentials
BINANCE_API_KEY = api_mainnet
BINANCE_API_SECRET = secret_mainnet

# Telegram Bot token
TELEGRAM_BOT_TOKEN = telegram_bot_token
TELEGRAM_CHAT_ID = telegram_chat_id

# Trading parameters
def get_trading_symbols(csv_path='symbols.csv'):
    """Reads trading symbols from a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"Symbol file not found at {csv_path}. Please create it.")
        return []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            symbols = [row[0] for row in reader]
        if not symbols:
            logger.error(f"Symbol file at {csv_path} is empty.")
            return []
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbol file: {e}")
        return []

RISK_PERCENTAGE = 0.01  # 1% of account balance
LEVERAGE = 10
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
SIGNAL_CONFIDENCE_THRESHOLD = 0.7

# --- Logging Setup ---

def setup_logging():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("smc_bot.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.error("Missing required environment variables. Please set BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID.")
    exit()

logger.info("SMC Bot started. Configuration loaded.")

TRADING_SYMBOLS = get_trading_symbols()

# --- Data Ingestion ---

def get_binance_client(authenticated=True):
    """Initializes and returns the Binance client."""
    try:
        config = {
            'options': {
                'defaultType': 'future',
            },
        }
        if authenticated:
            config['apiKey'] = BINANCE_API_KEY
            config['secret'] = BINANCE_API_SECRET

        binance = ccxt.binance(config)
        binance.load_markets()
        return binance
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication with Binance failed: {e}")
        return None
    except ccxt.NetworkError as e:
        logger.error(f"Network error connecting to Binance: {e}")
        return None

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

def get_progress_bar(*args, **kwargs):
    """
    Returns a configured tqdm progress bar instance.
    This is a centralized place to manage progress bar settings.
    """
    # You can add default settings here, e.g.:
    # kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    return tqdm(*args, **kwargs)

def thread_friendly_progress_bar(fn, items, max_workers, desc):
    """
    A wrapper around tqdm.contrib.concurrent.thread_map for thread-friendly progress bars.
    """
    with get_progress_bar(total=len(items), desc=desc) as pbar:
        def wrapper(item):
            result = fn(item, pbar)
            pbar.update()
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(wrapper, items))
    return results

def fetch_ohlcv(client, symbol, timeframe, since=None, limit=None):
    """
    Fetches historical OHLCV data from Binance.
    """
    try:
        ohlcv = client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except ccxt.BadSymbol:
        logger.error(f"Symbol {symbol} not found on Binance.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

# Example usage:
# binance_client = get_binance_client()
# if binance_client:
#     df_5m = fetch_ohlcv(binance_client, TRADING_SYMBOL, '5m')
#     df_1h = fetch_ohlcv(binance_client, TRADING_SYMBOL, '1h')
#     logger.info(f"Fetched 5m data: {len(df_5m)} rows")
#     logger.info(f"Fetched 1h data: {len(df_1h)} rows")

# --- SMC Pattern Labeling ---

@njit
def is_order_block(high, low, close, open, i):
    """
    Identifies an Order Block.
    A simple definition: a bearish candle followed by a strong bullish move
    that breaks the high of the bearish candle. Vice-versa for bearish OB.
    """
    # Bullish Order Block
    if close[i-1] < open[i-1] and close[i] > open[i]:
        if high[i] > high[i-1]:
            return True
    # Bearish Order Block
    if close[i-1] > open[i-1] and close[i] < open[i]:
        if low[i] < low[i-1]:
            return True
    return False

@njit
def is_fair_value_gap(high, low, i):
    """
    Identifies a Fair Value Gap (FVG) or Imbalance.
    A 3-candle pattern where there is a gap between the wicks of the 1st and 3rd candles.
    """
    # Bullish FVG
    if high[i] < low[i-2] and high[i] < low[i+1]:
        return True
    # Bearish FVG
    if low[i] > high[i-2] and low[i] > high[i+1]:
        return True
    return False

@njit
def is_breaker_block(high, low, close, open, i):
    """
    Identifies a Breaker Block.
    A simplified definition: An order block that is violated.
    """
    if is_order_block(high, low, close, open, i-1):
        # Bullish breaker
        if close[i] < low[i-1]:
            return True
        # Bearish breaker
        if close[i] > high[i-1]:
            return True
    return False

@njit
def is_liquidity_run(high, low, i, recent_swing_high, recent_swing_low):
    """
    Identifies a Liquidity Run (or stop hunt).
    Flags wicks that go beyond recent swing points.
    """
    if high[i] > recent_swing_high or low[i] < recent_swing_low:
        return True
    return False

def check_system_load(cpu_threshold=80, memory_threshold=80):
    """
    Checks CPU and memory usage and logs a warning if they exceed the thresholds.
    """
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    if cpu_usage > cpu_threshold:
        logger.warning(f"High CPU usage detected: {cpu_usage}%")
    if memory_usage > memory_threshold:
        logger.warning(f"High memory usage detected: {memory_usage}%")

@njit
def label_smc_patterns_chunk_numba(high, low, close, open, order_block, fair_value_gap, breaker_block, liquidity_run):
    for i in range(2, len(high) - 1):
        if is_order_block(high, low, close, open, i):
            order_block[i] = True

        if is_fair_value_gap(high, low, i):
            fair_value_gap[i] = True

        if is_breaker_block(high, low, close, open, i):
            breaker_block[i] = True

        if i > 10:
            recent_swing_high = np.max(high[i-11:i-1])
            recent_swing_low = np.min(low[i-11:i-1])
            if is_liquidity_run(high, low, i, recent_swing_high, recent_swing_low):
                liquidity_run[i] = True
    return order_block, fair_value_gap, breaker_block, liquidity_run

def label_smc_patterns_chunk(chunk_data, pbar=None):
    """
    Applies SMC pattern labeling to a chunk of the dataframe.
    """
    chunk_idx, df_chunk_in = chunk_data
    df_chunk = df_chunk_in.copy()
    if pbar:
        pbar.set_description(f"Labeling chunk {chunk_idx} ({len(df_chunk)} rows)")
    check_system_load()
    try:
        high = df_chunk['high'].values
        low = df_chunk['low'].values
        close = df_chunk['close'].values
        open_ = df_chunk['open'].values
        
        order_block = np.full(len(df_chunk), False)
        fair_value_gap = np.full(len(df_chunk), False)
        breaker_block = np.full(len(df_chunk), False)
        liquidity_run = np.full(len(df_chunk), False)

        order_block, fair_value_gap, breaker_block, liquidity_run = label_smc_patterns_chunk_numba(
            high, low, close, open_, order_block, fair_value_gap, breaker_block, liquidity_run
        )

        df_chunk['order_block'] = order_block
        df_chunk['fair_value_gap'] = fair_value_gap
        df_chunk['breaker_block'] = breaker_block
        df_chunk['liquidity_run'] = liquidity_run
        
        with open("label_heartbeat.txt", "a") as hb:
            hb.write(f"{datetime.now()}: finished chunk {chunk_idx}\n")
        
        return df_chunk
    except Exception as e:
        logger.error(f"An error occurred in label_smc_patterns_chunk {chunk_idx}: {e}")
        return pd.DataFrame()

def process_symbol(symbol):
    """
    Downloads and labels data for a single symbol.
    """
    labeled_data_dir = 'labeled_data'
    if not os.path.exists(labeled_data_dir):
        os.makedirs(labeled_data_dir)

    safe_symbol = symbol.replace('/', '_')
    labeled_cache_file_5m = f"{labeled_data_dir}/{safe_symbol}_5m_labeled.csv"
    labeled_cache_file_1h = f"{labeled_data_dir}/{safe_symbol}_1h_labeled.csv"

    if os.path.exists(labeled_cache_file_5m) and os.path.exists(labeled_cache_file_1h):
        logger.info(f"Loading cached labeled data for {symbol}")
        df_5m = pd.read_csv(labeled_cache_file_5m, index_col='timestamp', parse_dates=True)
        df_1h = pd.read_csv(labeled_cache_file_1h, index_col='timestamp', parse_dates=True)
        return df_5m, df_1h

    try:
        unauthenticated_client = get_binance_client(authenticated=False)
        logger.info(f"Processing symbol: {symbol}")
        
        logger.info(f"Fetching 5m data for {symbol}...")
        df_5m = fetch_ohlcv(unauthenticated_client, symbol, '5m')
        
        logger.info(f"Fetching 1h data for {symbol}...")
        df_1h = fetch_ohlcv(unauthenticated_client, symbol, '1h')

        if df_5m.empty or df_1h.empty:
            logger.warning(f"Could not fetch sufficient data for {symbol}. Skipping...")
            return None

        logger.info(f"Labeling 5m SMC patterns for {symbol}...")
        df_5m = label_smc_patterns(df_5m, args.debug)
        
        logger.info(f"Labeling 1h SMC patterns for {symbol}...")
        df_1h = label_smc_patterns(df_1h, args.debug)
        
        df_5m.to_csv(labeled_cache_file_5m)
        df_1h.to_csv(labeled_cache_file_1h)
        logger.info(f"Cached labeled data for {symbol}")
        
        with open("last_symbol.txt", "w") as f:
            f.write(symbol)
            
        logger.info(f"Finished processing symbol: {symbol}")
        return df_5m, df_1h
    except Exception as e:
        logger.error(f"An error occurred while processing symbol {symbol}: {e}")
        return None

def label_smc_patterns(df, debug=False):
    """
    Applies SMC pattern labeling to the dataframe using multithreading.
    """
    df['order_block'] = False
    df['fair_value_gap'] = False
    df['breaker_block'] = False
    df['liquidity_run'] = False
    df['mitigation_zone'] = False
    df['smc_labels'] = [[] for _ in range(len(df))]

    num_processes = 4  # Limit the number of threads
    chunk_size = max(1000, len(df) // num_processes)  # Increase chunk size
    chunks = [(i, df.iloc[i:i + chunk_size]) for i in range(0, len(df), chunk_size)]

    if debug:
        results = []
        with get_progress_bar(total=len(chunks), desc="Labeling SMC Patterns (Debug)") as pbar:
            for chunk in chunks:
                results.append(label_smc_patterns_chunk(chunk, pbar))
                pbar.update()
    else:
        results = thread_friendly_progress_bar(label_smc_patterns_chunk, chunks, max_workers=num_processes, desc="Labeling SMC Patterns")
        
    return pd.concat([res for res in results if res is not None and not res.empty])

# --- Feature Engineering ---

def create_features(df_main, df_higher_tf):
    """
    Merges multi-timeframe data and creates features.
    """
    # Resample higher timeframe data to match the main dataframe
    df_main = df_main[~df_main.index.duplicated(keep='first')]
    df_higher_tf = df_higher_tf[~df_higher_tf.index.duplicated(keep='first')]
    df_main = df_main.resample('5T').ffill()
    df_higher_tf = df_higher_tf.resample('5T').ffill()
    df = pd.merge(df_main, df_higher_tf, left_index=True, right_index=True, suffixes=('', '_htf'))

    # ATR for stop loss and take profit
    df.ta.atr(length=14, append=True)

    # Other indicators
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=50, append=True)

    # --- New Feature Engineering ---

    # Zone Distance Metrics
    for zone_type in ['order_block', 'fair_value_gap', 'breaker_block']:
        zone_indices = np.where(df[zone_type])[0]
        if len(zone_indices) > 0:
            # Get the index of the last zone for each row
            last_zone_indices = np.searchsorted(zone_indices, np.arange(len(df)), side='right') - 1
            # Get the 'close' price of the last zone
            last_zone_close = df['close'].values[zone_indices[last_zone_indices]]
            df[f'dist_to_{zone_type}'] = np.abs(df['close'].values - last_zone_close)
            df[f'{zone_type}_width'] = df['high'].values[zone_indices[last_zone_indices]] - df['low'].values[zone_indices[last_zone_indices]]
        else:
            df[f'dist_to_{zone_type}'] = 9999 # Use a large number to indicate no zone
            df[f'{zone_type}_width'] = 0

    # Time-of-Day & Session Flags
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek

    df.fillna(0, inplace=True) # Fill any other NaNs with 0
    
    # Multi-TF Context Ratios
    df['htf_trend_slope'] = get_trend_slope(df['close_htf'])

    return df

def scale_features(df):
    """
    Scales the features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    # Select only numerical columns for scaling
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target variables
    targets = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run']
    for target in targets:
        if target in feature_cols:
            feature_cols.remove(target)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler, feature_cols

def get_trend_slope(series, window=20):
    """
    Calculates the slope of a series using linear regression.
    """
    slopes = [0] * (window - 1)
    for i in range(window - 1, len(series)):
        y = series[i - window + 1:i + 1]
        x = list(range(len(y)))
        model = np.polyfit(x, y, 1)
        slopes.append(model[0])
    return slopes

# --- Model Training & Artifact Management ---

def objective(trial, df, features):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Define hyperparameters to tune
    solver = trial.suggest_categorical('solver', ['lbfgs'])
    c = trial.suggest_float('C', 1e-4, 1e4, log=True)

    params = {'solver': solver, 'C': c, 'max_iter': 2000, 'class_weight': 'balanced'}

    targets = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run']

    X = df[features]
    y = df[targets]

    total_accuracy = 0
    num_models = 0

    for target in targets:
        if len(y[target].unique()) < 2:
            continue

        num_models += 1
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index][target], y.iloc[test_index][target]

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        total_accuracy += np.mean(accuracies)

    return total_accuracy / num_models if num_models > 0 else 0

def train_model(df, features, params=None):
    """
    Trains a multi-output model using time-series cross-validation, handling single-class data.
    """
    if params is None:
        params = {}
    params.setdefault('solver', 'lbfgs')
    params.setdefault('max_iter', 2000)
    params.setdefault('class_weight', 'balanced')

    targets = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run', 'mitigation_zone']

    X = df[features]
    y = df[targets]

    # --- Data Analysis ---
    logger.info("Target variable distribution:")
    for target in targets:
        logger.info(f"- {target}: {y[target].value_counts(normalize=True).to_dict()}")

    # --- Train Model ---
    estimators = []

    for target in targets:
        if len(y[target].unique()) < 2:
            logger.warning(f"Skipping training for '{target}' due to single-class data.")
            estimators.append(None)
            continue

        # Train final model on all data for this target
        final_model_target = LogisticRegression(**params)
        final_model_target.fit(X, y[target])
        estimators.append(final_model_target)

    return estimators, features

def save_artifacts(models, scaler, features, model_path='smc_model.pkl', scaler_path='scaler.pkl', features_path='features.pkl'):
    """
    Saves the trained models, scaler and features to files.
    """
    joblib.dump(models, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(features, features_path)
    logger.info(f"Models saved to {model_path}, scaler saved to {scaler_path}, features saved to {features_path}")

def load_artifacts(model_path='smc_model.pkl', scaler_path='scaler.pkl', features_path='features.pkl'):
    """
    Loads the models, scaler and features from files.
    """
    try:
        models = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        logger.info(f"Models, scaler and features loaded from {model_path}, {scaler_path} and {features_path}")
        return models, scaler, features
    except FileNotFoundError:
        logger.warning("Model, scaler or features artifacts not found. Need to train first.")
        return None, None, None

# --- Live Inference & Signal Construction ---

def get_signal(df_latest, models, scaler, features, symbol, df_higher_tf):
    """
    Runs inference on the latest data and constructs a signal if multiple conditions are met,
    including a trend filter from a higher timeframe.
    """
    if df_latest.empty:
        return None

    # --- Higher Timeframe Trend Analysis ---
    htf_trend = df_higher_tf['htf_trend_slope'].iloc[-1]
    
    # Prepare features for the latest candle
    latest_features_df = df_latest[features]
    latest_features_df.fillna(0, inplace=True)

    latest_features_scaled = scaler.transform(latest_features_df)

    patterns = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run']
    
    # --- Multi-Condition Logic ---
    met_conditions = []
    confidences = {}

    for i, model in enumerate(models):
        if model is None:
            continue

        pred_proba = model.predict_proba(latest_features_scaled)[:, 1]
        if pred_proba[0] >= SIGNAL_CONFIDENCE_THRESHOLD:
            met_conditions.append(patterns[i])
            confidences[patterns[i]] = pred_proba[0]

    # Require at least 2 conditions to be met
    if len(met_conditions) < 2:
        return None

    # --- Signal Construction with HTF Filter ---
    side = None
    if htf_trend > 0 and 'order_block' in met_conditions and 'fair_value_gap' in met_conditions:
        side = 'buy'
    elif htf_trend < 0 and 'breaker_block' in met_conditions:
        side = 'sell'
    else:
        return None # Conditions not met for a trade
    
    entry_price = df_latest['close'].iloc[0]
    atr = df_latest['ATRr_14'].iloc[0] if 'ATRr_14' in df_latest.columns else 0.001
    
    if side == 'buy':
        stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
        take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
    else: # Sell
        stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
        take_profit = entry_price - (atr * TAKE_PROFIT_ATR_MULTIPLIER)

    signal = {
        'symbol': symbol,
        'side': side,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'quantity': 1,  # This should be calculated based on risk
        'pattern': ", ".join(met_conditions),
        'confidence': np.mean(list(confidences.values())),
        'timeframe': df_latest.index.freqstr if hasattr(df_latest.index, 'freqstr') else '5m',
    }
    logger.info(f"Signal generated: {signal}")
    return signal

# --- Order Execution & Telegram Alerts ---

async def send_telegram_alert(signal):
    """
    Sends a formatted alert to a Telegram channel with an inline keyboard.
    """
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        keyboard = [
            [
                telegram.InlineKeyboardButton("👍 Accept", callback_data=f"accept_{signal['symbol']}"),
                telegram.InlineKeyboardButton("👎 Reject", callback_data=f"reject_{signal['symbol']}"),
            ]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)

        message = (
            f"🚀 New Signal: {signal['symbol']} 🚀\n"
            f"Pattern: {signal['pattern']} ({signal['confidence']:.2f}%)\n"
            f"Timeframe: {signal['timeframe']}\n"
            f"Side: {signal['side'].upper()}\n"
            f"Entry: {signal['entry_price']:.2f}\n"
            f"Stop Loss: {signal['stop_loss']:.2f}\n"
            f"Take Profit: {signal['take_profit']:.2f}\n"
            f"Risked Quantity: {signal['quantity']}"
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, reply_markup=reply_markup)
        logger.info("Telegram alert sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")

def handle_telegram_callback(update, context):
    """
    Handles feedback from the inline keyboard.
    """
    query = update.callback_query
    query.answer()

    action, symbol = query.data.split('_')

    if action == 'accept':
        logger.info(f"User accepted signal for {symbol}")
        # Here you could trigger the order execution
    elif action == 'reject':
        logger.info(f"User rejected signal for {symbol}")
        # Here you could log the rejection for later analysis

def execute_order(client, signal, atr):
    """
    Submits a limit order to Binance with dynamic position sizing.
    """
    if not client:
        return
    try:
        # Dynamic Position Sizing
        account_balance = client.fetch_balance()['free']['USDT']
        risk_amount_per_trade = account_balance * RISK_PERCENTAGE

        # Adjust risk based on volatility (ATR)
        # For higher volatility, risk less
        volatility_adjustment = 1 / atr
        adjusted_risk_amount = risk_amount_per_trade * volatility_adjustment

        entry_price = signal['entry_price']
        stop_loss_price = signal['stop_loss']
        quantity = adjusted_risk_amount / abs(entry_price - stop_loss_price)

        # Create order
        order = client.create_limit_order(
            signal['symbol'],
            signal['side'],
            quantity,
            entry_price
        )
        logger.info(f"Limit order placed for {signal['symbol']} at {entry_price}")
        return order
    except Exception as e:
        logger.error(f"Failed to execute order: {e}")
        return None

def trail_stop_loss(client, order, atr):
    """
    Trails the stop loss for an open position.
    """
    if not client or not order:
        return

    try:
        # Get current position and price
        position = client.fetch_position(order['symbol'])
        current_price = client.fetch_ticker(order['symbol'])['last']

        if position['side'] == 'long':
            # Breakeven
            if current_price > order['price'] * 1.01: # 1% profit
                new_stop_loss = order['price']
                client.edit_order(order['id'], order['symbol'], stopPrice=new_stop_loss)
            # Trailing
            elif current_price > position['entryPrice']:
                new_stop_loss = current_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
                if new_stop_loss > position['stopLossPrice']:
                    client.edit_order(order['id'], order['symbol'], stopPrice=new_stop_loss)

        elif position['side'] == 'short':
            # Breakeven
            if current_price < order['price'] * 0.99: # 1% profit
                new_stop_loss = order['price']
                client.edit_order(order['id'], order['symbol'], stopPrice=new_stop_loss)
            # Trailing
            elif current_price < position['entryPrice']:
                new_stop_loss = current_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
                if new_stop_loss < position['stopLossPrice']:
                    client.edit_order(order['id'], order['symbol'], stopPrice=new_stop_loss)
    except Exception as e:
        logger.error(f"Failed to trail stop loss: {e}")

# --- Main Loop & Scheduling ---

async def main(application):
    """
    Main execution loop for the bot.
    """
    unauthenticated_client = get_binance_client(authenticated=False)
    authenticated_client = get_binance_client(authenticated=True)

    if not unauthenticated_client:
        logger.error("Could not create unauthenticated Binance client. Exiting.")
        return

    if not TRADING_SYMBOLS:
        logger.error("No trading symbols found. Please create a 'symbols.csv' file. Exiting.")
        return

    if not authenticated_client:
        logger.warning("Could not create authenticated Binance client. Trading will be disabled.")

    models, scaler, features = load_artifacts()

    if models and scaler and features:
        user_input = input("Found existing models. Do you want to [r]etrain or [s]tart signal generation? ").lower()
        if user_input == 's':
            logger.info("Starting signal generation with the existing models.")
        elif user_input == 'r':
            models, scaler = None, None # Force retraining
        else:
            logger.info("Invalid input. Exiting.")
            return

    if not models or not scaler:
        train_mode = input("Do you want to perform a [f]ull train or a [q]uick train? ").lower()
        if train_mode == 'q':
            logger.info("Performing quick model training...")
            symbols_to_process = TRADING_SYMBOLS[:1]
            days_to_fetch = 30
        elif train_mode == 'f':
            logger.info("Performing full model training...")
            symbols_to_process = TRADING_SYMBOLS
            days_to_fetch = None # Fetch all data
        else:
            logger.error("Invalid training mode selected. Exiting.")
            return

        all_dfs_5m = []
        all_dfs_1h = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(get_progress_bar(executor.map(process_symbol, symbols_to_process), total=len(symbols_to_process), desc="Processing Symbols"))

        for result in results:
            if result is not None:
                df_5m, df_1h = result
                if days_to_fetch:
                    df_5m = df_5m.last(f'{days_to_fetch}D')
                    df_1h = df_1h.last(f'{days_to_fetch}D')
                all_dfs_5m.append(df_5m)
                all_dfs_1h.append(df_1h)

        if not all_dfs_5m:
            logger.error("No data fetched for any symbol. Exiting.")
            return
            
        # Combine data from all symbols
        combined_df_5m = pd.concat(all_dfs_5m)
        combined_df_1h = pd.concat(all_dfs_1h)
        
        features_cache_path = 'features_df.pkl'
        if os.path.exists(features_cache_path):
            logger.info("Loading cached features...")
            features_df = pd.read_pickle(features_cache_path)
        else:
            logger.info("Creating features for combined data...")
            features_df = create_features(combined_df_5m, combined_df_1h)
            features_df.to_pickle(features_cache_path)

        scaled_df, scaler, features = scale_features(features_df)

        logger.info("Running Optuna hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, scaled_df, features), n_trials=args.optuna_trials, n_jobs=-1, timeout=600)
        best_params = study.best_params
        logger.info(f"Best Optuna params: {best_params}")

        models, features = train_model(scaled_df, features, best_params)

        if models and scaler and features:
            save_artifacts(models, scaler, features)
        else:
            logger.error("Model training failed. Exiting.")
            return

    # --- Live Trading Loop ---
    logger.info("Starting live trading loop...")
    open_orders = []
    total_symbols_processed = 0
    errors_encountered = 0
    while True:
        try:
            if application.bot_data.get('paused', False):
                await asyncio.sleep(60)
                continue

            # Trail stop losses for open orders
            if authenticated_client:
                for order in open_orders:
                    atr = fetch_ohlcv(unauthenticated_client, order['symbol'], '5m')['close'].std() # Simplified ATR
                    trail_stop_loss(authenticated_client, order, atr)

            for symbol in TRADING_SYMBOLS:
                # Fetch latest data
                latest_df_5m = fetch_ohlcv(unauthenticated_client, symbol, '5m', limit=1000)
                latest_df_1h = fetch_ohlcv(unauthenticated_client, symbol, '1h', limit=300)

                if latest_df_5m.empty or latest_df_1h.empty:
                    logger.warning(f"Could not fetch latest data for {symbol}. Skipping...")
                    continue

                # Label and feature engineer the latest data
                latest_df_5m = label_smc_patterns(latest_df_5m, args.debug)
                latest_df_1h = label_smc_patterns(latest_df_1h, args.debug)
                latest_features_df = create_features(latest_df_5m, latest_df_1h)

                if features: # Ensure features are defined
                    # Get the very last row for signal generation
                    last_candle_features = latest_features_df.iloc[[-1]]

                    # Get signal
                    signal = get_signal(last_candle_features, models, scaler, features, symbol, latest_features_df)

                    if signal:
                        await send_telegram_alert(signal)
                        if authenticated_client:
                            atr = latest_features_df['ATRr_14'].iloc[-1]
                            order = execute_order(authenticated_client, signal, atr)
                            if order:
                                open_orders.append(order)

            total_symbols_processed += len(TRADING_SYMBOLS)
            logger.info("--- Progress Summary ---")
            logger.info(f"Total symbols processed in this cycle: {len(TRADING_SYMBOLS)}")
            logger.info(f"Total symbols processed overall: {total_symbols_processed}")
            logger.info(f"Errors encountered in this cycle: {errors_encountered}")
            logger.info("--------------------------")
            errors_encountered = 0  # Reset for the next cycle

            # Wait for a shorter interval for higher frequency
            logger.info("Completed a cycle through all symbols. Waiting for 1 minute...")
            await asyncio.sleep(60) # 1 minute

        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}. Reconnecting...")
            errors_encountered += 1
            await asyncio.sleep(60)
            binance_client = get_binance_client()
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            errors_encountered += 1
            await asyncio.sleep(60)

def start(update, context):
    update.message.reply_text('SMC Bot started!')

def pause(update, context):
    context.bot_data['paused'] = True
    update.message.reply_text('SMC Bot paused.')

def resume(update, context):
    context.bot_data['paused'] = False
    update.message.reply_text('SMC Bot resumed.')

def status(update, context):
    if context.bot_data.get('paused', False):
        update.message.reply_text('SMC Bot is paused.')
    else:
        update.message.reply_text('SMC Bot is running.')

def run_backtest(symbol, days=30):
    """
    Runs a backtest on historical data.
    """
    logger.info(f"Running backtest for {symbol} for the last {days} days...")

    binance_client = get_binance_client()
    if not binance_client:
        return

    # Load data
    df_5m = fetch_ohlcv(binance_client, symbol, '5m', since=binance_client.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat()))
    df_1h = fetch_ohlcv(binance_client, symbol, '1h', since=binance_client.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat()))

    if df_5m.empty or df_1h.empty:
        logger.error(f"Could not fetch data for {symbol}. Exiting backtest.")
        return

    # Prepare data and model
    df_5m = label_smc_patterns(df_5m, args.debug)
    df_1h = label_smc_patterns(df_1h, args.debug)
    features_df = create_features(df_5m, df_1h)
    scaled_df, scaler = scale_features(features_df)
    model, features = train_model(scaled_df)

    # Backtest simulation
    balance = 10000 # Starting balance
    pnl = 0
    wins = 0
    losses = 0
    trades = 0

    for i in range(1, len(scaled_df)):
        last_candle_features = scaled_df.iloc[[i]]
        signal = get_signal(last_candle_features, model, scaler, features, symbol, df_1h)

        if signal:
            trades += 1
            # Simulate trade execution
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # Simplified PnL calculation
            if np.random.rand() > 0.5: # 50% win rate
                pnl += abs(take_profit - entry_price)
                wins += 1
            else:
                pnl -= abs(entry_price - stop_loss)
                losses += 1

    # Performance Report
    logger.info("--- Backtest Performance Report ---")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Total Trades: {trades}")
    logger.info(f"Wins: {wins}")
    logger.info(f"Losses: {losses}")
    logger.info(f"Win Rate: {wins/trades*100 if trades > 0 else 0:.2f}%")
    logger.info(f"Total PnL: ${pnl:.2f}")
    logger.info(f"Final Balance: ${balance + pnl:.2f}")
    logger.info("------------------------------------")

from telegram.request import HTTPXRequest

async def run_bot():
    request = HTTPXRequest(connect_timeout=30, read_timeout=30)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("pause", pause))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CallbackQueryHandler(handle_telegram_callback))

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    await main(application)

    await application.updater.stop()
    await application.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SMC Trading Bot')
    parser.add_argument('--backtest', type=str, help='Run a backtest for the specified symbol (e.g., BTC/USDT)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--optuna-trials', type=int, default=10, help='Number of Optuna trials to run')
    parser.add_argument('--debug', action='store_true', help='Run in sequential mode for debugging')
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest, args.days)
    else:
        asyncio.run(run_bot())
