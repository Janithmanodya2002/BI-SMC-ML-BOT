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

def fetch_ohlcv(client, symbol, timeframe, since=None):
    """
    Fetches all available historical OHLCV data from Binance, caching it to a local file.
    """
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    safe_symbol = symbol.replace('/', '_')
    cache_file = f"{data_dir}/{safe_symbol}_{timeframe}.csv"

    # If cache exists, load from it
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data for {symbol} ({timeframe}) from {cache_file}")
        df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        return df

    # If cache does not exist, download the data
    logger.info(f"No cache found for {symbol} ({timeframe}). Downloading new data.")
    
    if not client:
        return pd.DataFrame()
        
    try:
        all_ohlcv = []
        if since is None:
            since = client.parse8601('2017-01-01T00:00:00Z')

        with tqdm(total=None, desc=f"Downloading {symbol} {timeframe}") as pbar:
            while True:
                ohlcv = client.fetch_ohlcv(symbol, timeframe, since=since)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                pbar.update(len(ohlcv))
        
        if not all_ohlcv:
            logger.warning(f"No new data found for {symbol} ({timeframe})")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df.to_csv(cache_file)
        logger.info(f"Cached data for {symbol} ({timeframe}) to {cache_file}")

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

def is_order_block(candle, previous_candle):
    """
    Identifies an Order Block.
    A simple definition: a bearish candle followed by a strong bullish move
    that breaks the high of the bearish candle. Vice-versa for bearish OB.
    """
    # Bullish Order Block
    if previous_candle['close'] < previous_candle['open'] and candle['close'] > candle['open']:
        if candle['high'] > previous_candle['high']:
            return True
    # Bearish Order Block
    if previous_candle['close'] > previous_candle['open'] and candle['close'] < candle['open']:
        if candle['low'] < previous_candle['low']:
            return True
    return False

def is_fair_value_gap(candle, prev_candle, next_candle):
    """
    Identifies a Fair Value Gap (FVG) or Imbalance.
    A 3-candle pattern where there is a gap between the wicks of the 1st and 3rd candles.
    """
    # Bullish FVG
    if candle['high'] < prev_candle['low'] and candle['high'] < next_candle['low']:
        return True
    # Bearish FVG
    if candle['low'] > prev_candle['high'] and candle['low'] > next_candle['high']:
        return True
    return False

def is_breaker_block(candle, previous_candle, df_slice):
    """
    Identifies a Breaker Block.
    A simplified definition: An order block that is violated.
    """
    if is_order_block(previous_candle, df_slice.iloc[-2]):
        # Bullish breaker
        if candle['close'] < previous_candle['low']:
            return True
        # Bearish breaker
        if candle['close'] > previous_candle['high']:
            return True
    return False

def is_liquidity_run(candle, recent_swing_high, recent_swing_low):
    """
    Identifies a Liquidity Run (or stop hunt).
    Flags wicks that go beyond recent swing points.
    """
    if candle['high'] > recent_swing_high or candle['low'] < recent_swing_low:
        return True
    return False

def is_mitigation_zone(candle, fvg_zones, ob_zones):
    """
    Identifies when price enters a Mitigation Zone (an unfilled FVG or OB).
    """
    for _, fvg in fvg_zones.iterrows():
        if fvg['low'] < candle['close'] < fvg['high']:
            return True
    for _, ob in ob_zones.iterrows():
        if ob['low'] < candle['close'] < ob['high']:
            return True
    return False

def label_smc_patterns_chunk(df_chunk):
    """
    Applies SMC pattern labeling to a chunk of the dataframe.
    """
    try:
        for i in range(2, len(df_chunk) - 1):
            previous_candle = df_chunk.iloc[i-1]
            current_candle = df_chunk.iloc[i]
            next_candle = df_chunk.iloc[i+1]
            df_slice = df_chunk.iloc[:i]

            labels = []

            if is_order_block(current_candle, previous_candle):
                df_chunk.at[df_chunk.index[i], 'order_block'] = True
                labels.append('OB')

            if is_fair_value_gap(current_candle, previous_candle, next_candle):
                df_chunk.at[df_chunk.index[i], 'fair_value_gap'] = True
                labels.append('FVG')

            if is_breaker_block(current_candle, previous_candle, df_slice):
                df_chunk.at[df_chunk.index[i], 'breaker_block'] = True
                labels.append('BB')

            if len(df_slice) > 10:
                recent_swing_high = df_slice['high'].rolling(window=10).max().iloc[-2]
                recent_swing_low = df_slice['low'].rolling(window=10).min().iloc[-2]
                if is_liquidity_run(current_candle, recent_swing_high, recent_swing_low):
                    df_chunk.at[df_chunk.index[i], 'liquidity_run'] = True
                    labels.append('LR')

            fvg_zones = df_chunk.loc[df_chunk['fair_value_gap']]
            ob_zones = df_chunk.loc[df_chunk['order_block']]
            if is_mitigation_zone(current_candle, fvg_zones, ob_zones):
                df_chunk.at[df_chunk.index[i], 'mitigation_zone'] = True
                labels.append('MZ')

            df_chunk.at[df_chunk.index[i], 'smc_labels'] = labels
        return df_chunk
    except Exception as e:
        logger.error(f"An error occurred in label_smc_patterns_chunk: {e}")
        return None

def process_symbol(symbol):
    """
    Downloads and labels data for a single symbol.
    """
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
        df_5m = label_smc_patterns(df_5m)
        
        logger.info(f"Labeling 1h SMC patterns for {symbol}...")
        df_1h = label_smc_patterns(df_1h)
        
        with open("last_symbol.txt", "w") as f:
            f.write(symbol)
            
        logger.info(f"Finished processing symbol: {symbol}")
        return df_5m, df_1h
    except Exception as e:
        logger.error(f"An error occurred while processing symbol {symbol}: {e}")
        return None

def label_smc_patterns(df):
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
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(label_smc_patterns_chunk, chunks), total=len(chunks), desc="Labeling SMC Patterns"))

    return pd.concat([res for res in results if res is not None])

# --- Feature Engineering ---

def create_features(df_main, df_higher_tf):
    """
    Merges multi-timeframe data and creates features.
    """
    # Resample higher timeframe data to match the main dataframe
    df_higher_tf = df_higher_tf.resample(df_main.index.freq).ffill()
    df = pd.merge(df_main, df_higher_tf, left_index=True, right_index=True, suffixes=('', '_htf'))

    # ATR for stop loss and take profit
    df.ta.atr(length=14, append=True)

    # Other indicators
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=50, append=True)

    # --- New Feature Engineering ---

    # Multi-TF Context Ratios
    df['htf_trend_slope'] = get_trend_slope(df['close_htf'])

    # Zone Distance Metrics
    for zone_type in ['order_block', 'fair_value_gap', 'breaker_block']:
        zones = df[df[zone_type]]
        if not zones.empty:
            # Find the index of the last zone for each row
            last_zone_indices = zones.index.searchsorted(df.index, side='right') - 1
            # Get the 'close' price of the last zone
            df[f'dist_to_{zone_type}'] = (df['close'] - zones['close'].iloc[last_zone_indices].values).abs()
            df[f'{zone_type}_width'] = (zones['high'].iloc[last_zone_indices].values - zones['low'].iloc[last_zone_indices].values)
        else:
            df[f'dist_to_{zone_type}'] = 9999 # Use a large number to indicate no zone
            df[f'{zone_type}_width'] = 0

    # Time-of-Day & Session Flags
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek

    df.fillna(0, inplace=True) # Fill any other NaNs with 0

    return df

def scale_features(df):
    """
    Scales the features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    # Select only numerical columns for scaling
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target variables
    targets = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run', 'mitigation_zone']
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
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    c = trial.suggest_float('C', 1e-4, 1e4, log=True)

    params = {'solver': solver, 'C': c}

    targets = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run', 'mitigation_zone']

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

def save_artifacts(models, scaler, model_path='smc_model.pkl', scaler_path='scaler.pkl'):
    """
    Saves the trained models and scaler to files.
    """
    joblib.dump(models, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Models saved to {model_path}, scaler saved to {scaler_path}")

def load_artifacts(model_path='smc_model.pkl', scaler_path='scaler.pkl'):
    """
    Loads the models and scaler from files.
    """
    try:
        models = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"Models and scaler loaded from {model_path} and {scaler_path}")
        return models, scaler
    except FileNotFoundError:
        logger.warning("Model or scaler artifacts not found. Need to train first.")
        return None, None

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
    latest_features_df = pd.DataFrame(df_latest[features])
    latest_features_df = latest_features_df[features]  # Ensure column order
    latest_features_df.fillna(0, inplace=True)

    latest_features_scaled = scaler.transform(latest_features_df)

    patterns = ['order_block', 'fair_value_gap', 'breaker_block', 'liquidity_run', 'mitigation_zone']
    
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

    models, scaler = load_artifacts()
    features = []

    if models and scaler:
        user_input = input("Found existing models. Do you want to [r]etrain or [s]tart signal generation? ").lower()
        if user_input == 's':
            logger.info("Starting signal generation with the existing models.")
        elif user_input == 'r':
            models, scaler = None, None # Force retraining
        else:
            logger.info("Invalid input. Exiting.")
            return

    if not models or not scaler:
        logger.info("Performing initial data load and model training for all symbols...")
        
        last_symbol = None
        if os.path.exists("last_symbol.txt"):
            with open("last_symbol.txt", "r") as f:
                last_symbol = f.read().strip()
        
        start_index = 0
        if last_symbol in TRADING_SYMBOLS:
            start_index = TRADING_SYMBOLS.index(last_symbol) + 1
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(process_symbol, TRADING_SYMBOLS[start_index:]), total=len(TRADING_SYMBOLS[start_index:]), desc="Processing Symbols"))

        all_dfs_5m = [result[0] for result in results if result is not None]
        all_dfs_1h = [result[1] for result in results if result is not None]

        if not all_dfs_5m:
            logger.error("No data fetched for any symbol. Exiting.")
            return

        # Combine data from all symbols
        combined_df_5m = pd.concat(all_dfs_5m)
        combined_df_1h = pd.concat(all_dfs_1h)
        
        logger.info("Creating features for combined data...")
        features_df = create_features(combined_df_5m, combined_df_1h)

        scaled_df, scaler, features = scale_features(features_df)

        logger.info("Running Optuna hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, scaled_df, features), n_trials=args.optuna_trials)
        best_params = study.best_params
        logger.info(f"Best Optuna params: {best_params}")

        models, _ = train_model(scaled_df, features, best_params)

        if models and scaler:
            save_artifacts(models, scaler)
        else:
            logger.error("Model training failed. Exiting.")
            return

    # --- Live Trading Loop ---
    logger.info("Starting live trading loop...")
    open_orders = []
    while True:
        try:
            if application.bot_data.get('paused', False):
                await asyncio.sleep(60)
                continue

            # Trail stop losses for open orders
            if authenticated_client:
                for order in open_orders:
                    atr = fetch_ohlcv(unauthenticated_client, order['symbol'], '5m', limit=20)['close'].std() # Simplified ATR
                    trail_stop_loss(authenticated_client, order, atr)

            for symbol in TRADING_SYMBOLS:
                # Fetch latest data
                latest_df_5m = fetch_ohlcv(unauthenticated_client, symbol, '5m', limit=100)
                latest_df_1h = fetch_ohlcv(unauthenticated_client, symbol, '1h', limit=100)

                if latest_df_5m.empty or latest_df_1h.empty:
                    logger.warning(f"Could not fetch latest data for {symbol}. Skipping...")
                    continue

                # Label and feature engineer the latest data
                latest_df_5m = label_smc_patterns(latest_df_5m)
                latest_df_1h = label_smc_patterns(latest_df_1h)
                latest_features_df = create_features(latest_df_5m, latest_df_1h)

                if features: # Ensure features are defined
                    # Get the very last row for signal generation
                    last_candle_features = latest_features_df.iloc[[-1]]

                    # Get signal
                    signal = get_signal(last_candle_features, models, scaler, features, symbol, latest_df_1h)

                    if signal:
                        await send_telegram_alert(signal)
                        if authenticated_client:
                            atr = latest_features_df['ATRr_14'].iloc[-1]
                            order = execute_order(authenticated_client, signal, atr)
                            if order:
                                open_orders.append(order)

            # Wait for a shorter interval for higher frequency
            logger.info("Completed a cycle through all symbols. Waiting for 1 minute...")
            await asyncio.sleep(60) # 1 minute

        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}. Reconnecting...")
            await asyncio.sleep(60)
            binance_client = get_binance_client()
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
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
    df_5m = label_smc_patterns(df_5m)
    df_1h = label_smc_patterns(df_1h)
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
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest, args.days)
    else:
        asyncio.run(run_bot())
