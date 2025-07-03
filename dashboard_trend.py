import pandas as pd
import pymysql
import json
from datetime import datetime
from Ticker import AVAILABLE_TICKERS
from Candlestick_pattern import detect_candlestick_patterns


def calculate_bollinger_bands(harga_saham, window=20, num_std=2):
    # Initialize signal column first
    harga_saham['Signal'] = None
    
    # Calculate Bollinger Bands
    harga_saham['Middle Band'] = harga_saham['Close'].rolling(window=window).mean()
    harga_saham['STD'] = harga_saham['Close'].rolling(window=window).std()
    harga_saham['Upper Band'] = harga_saham['Middle Band'] + (num_std * harga_saham['STD'])
    harga_saham['Lower Band'] = harga_saham['Middle Band'] - (num_std * harga_saham['STD'])
    
    # Generate signals only where we have complete Bollinger Bands data
    mask = ~harga_saham[['Upper Band', 'Lower Band', 'Middle Band']].isna().any(axis=1)
    
    # Update signals where we have complete data
    harga_saham.loc[mask & (harga_saham['Close'] >= harga_saham['Upper Band']), 'Signal'] = 'Sell'
    harga_saham.loc[mask & (harga_saham['Close'] <= harga_saham['Lower Band']), 'Signal'] = 'Buy'
    harga_saham.loc[mask & harga_saham['Signal'].isna(), 'Signal'] = 'Hold'
    
    return harga_saham


def calculate_ma(harga_saham, short_period=20, long_period=50):
    # Initialize signal column
    harga_saham['MaSignal'] = None
    
    # Calculate MAs with customizable periods
    harga_saham['short_MA'] = harga_saham['Close'].rolling(window=short_period).mean()
    harga_saham['long_MA'] = harga_saham['Close'].rolling(window=long_period).mean()
    
    # Generate signals only where we have both MAs
    mask = ~harga_saham[['short_MA', 'long_MA']].isna().any(axis=1)
    
    # Update signals where we have complete data
    harga_saham.loc[mask & (harga_saham['short_MA'] > harga_saham['long_MA']), 'MaSignal'] = 'Buy'
    harga_saham.loc[mask & (harga_saham['short_MA'] < harga_saham['long_MA']), 'MaSignal'] = 'Sell'
    harga_saham.loc[mask & harga_saham['MaSignal'].isna(), 'MaSignal'] = 'Hold'
    
    return harga_saham


def calculate_rsi(harga_saham, period=14, ob_level=70, os_level=30):
    # Initialize signal column
    harga_saham['RsiSignal'] = None
    
    # Calculate RSI
    delta = harga_saham['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    harga_saham['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate signals only where we have RSI
    mask = ~harga_saham['RSI'].isna()
    
    # Update signals where we have complete data using custom levels
    harga_saham.loc[mask & (harga_saham['RSI'] > ob_level), 'RsiSignal'] = 'Sell'
    harga_saham.loc[mask & (harga_saham['RSI'] < os_level), 'RsiSignal'] = 'Buy'
    harga_saham.loc[mask & harga_saham['RsiSignal'].isna(), 'RsiSignal'] = 'Hold'
    
    return harga_saham


def calculate_macd(harga_saham, short_window=12, long_window=26, signal_window=9):
    # Initialize signal column
    harga_saham['MacdSignal'] = None
    
    # Calculate MACD components with custom periods
    harga_saham['EMA_short'] = harga_saham['Close'].ewm(span=short_window, adjust=False).mean()
    harga_saham['EMA_long'] = harga_saham['Close'].ewm(span=long_window, adjust=False).mean()
    harga_saham['MACD'] = harga_saham['EMA_short'] - harga_saham['EMA_long']
    harga_saham['Signal_Line'] = harga_saham['MACD'].ewm(span=signal_window, adjust=False).mean()
    harga_saham['MACD_Hist'] = harga_saham['MACD'] - harga_saham['Signal_Line']
    
    # Generate signals only where we have complete MACD data
    mask = ~harga_saham[['MACD', 'Signal_Line']].isna().any(axis=1)
    
    # Update signals where we have complete data
    harga_saham.loc[mask & (harga_saham['MACD'] > harga_saham['Signal_Line']), 'MacdSignal'] = 'Buy'
    harga_saham.loc[mask & (harga_saham['MACD'] < harga_saham['Signal_Line']), 'MacdSignal'] = 'Sell'
    harga_saham.loc[mask & harga_saham['MacdSignal'].isna(), 'MacdSignal'] = 'Hold'
    
    return harga_saham


def calculate_adx(harga_saham, period=14, threshold=25):
    # Initialize signal column
    harga_saham['AdxSignal'] = None
    
    # Calculate TR and DM
    harga_saham['TR'] = pd.DataFrame({
        'HL': harga_saham['High'] - harga_saham['Low'],
        'HC': abs(harga_saham['High'] - harga_saham['Close'].shift()),
        'LC': abs(harga_saham['Low'] - harga_saham['Close'].shift())
    }).max(axis=1)
    
    harga_saham['+DM'] = (harga_saham['High'] - harga_saham['High'].shift()).clip(lower=0)
    harga_saham['-DM'] = (harga_saham['Low'].shift() - harga_saham['Low']).clip(lower=0)
    
    # Calculate smoothed values using custom period
    for col in ['TR', '+DM', '-DM']:
        harga_saham[f'{col}_smooth'] = harga_saham[col].rolling(period).sum()
    
    # Calculate DI and ADX
    harga_saham['+DI'] = 100 * harga_saham['+DM_smooth'] / harga_saham['TR_smooth']
    harga_saham['-DI'] = 100 * harga_saham['-DM_smooth'] / harga_saham['TR_smooth']
    harga_saham['DX'] = 100 * abs(harga_saham['+DI'] - harga_saham['-DI']) / (harga_saham['+DI'] + harga_saham['-DI'])
    harga_saham['ADX'] = harga_saham['DX'].rolling(period).mean()
    
    # Generate signals using custom threshold
    mask = ~harga_saham[['ADX', '+DI', '-DI']].isna().any(axis=1)
    
    # Update signals where we have complete data
    harga_saham.loc[mask & (harga_saham['ADX'] > threshold) & (harga_saham['+DI'] > harga_saham['-DI']), 'AdxSignal'] = 'Buy'
    harga_saham.loc[mask & (harga_saham['ADX'] > threshold) & (harga_saham['+DI'] < harga_saham['-DI']), 'AdxSignal'] = 'Sell'
    harga_saham.loc[mask & harga_saham['AdxSignal'].isna(), 'AdxSignal'] = 'Hold'
    
    return harga_saham


def calculate_volume(harga_saham, ma_window=20):
    # Initialize signal column
    harga_saham['VolumeSignal'] = 'Low Volume'
    
    # Calculate Volume Moving Average
    harga_saham['VMA'] = harga_saham['Volume'].rolling(window=ma_window).mean()
    
    # Generate signals only where we have VMA
    mask = ~harga_saham['VMA'].isna()
    
    # Update signals where we have complete data
    harga_saham.loc[mask & (harga_saham['Volume'] > harga_saham['VMA']), 'VolumeSignal'] = 'High Volume'
    
    return harga_saham


def calculate_fibonacci_retracement(harga_saham, lookback=60, retracement_levels=None):
    """
    Calculate Fibonacci retracement levels and generate signals based on price interaction with levels
    """
    if retracement_levels is None:
        retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    harga_saham = harga_saham.copy()
    harga_saham['Fibonacci_Signal'] = 'Hold'
    fib_highs = []
    fib_lows = []
    fib_levels_list = []

    # Create empty dictionary for each level
    fib_level_columns = {f'fib_{r}': [] for r in retracement_levels}

    for i in range(len(harga_saham)):
        if i < lookback - 1:
            fib_highs.append(None)
            fib_lows.append(None)
            fib_levels_list.append(None)
            for r in retracement_levels:
                fib_level_columns[f'fib_{r}'].append(None)
            continue

        recent = harga_saham.iloc[i - lookback + 1:i + 1]
        high = recent['High'].max()
        low = recent['Low'].min()
        last_close = harga_saham.iloc[i]['Close']

        levels = [high - (high - low) * r for r in retracement_levels]
        fib_highs.append(high)
        fib_lows.append(low)
        fib_levels_list.append(levels)

        # Save each level to its respective column
        for idx, r in enumerate(retracement_levels):
            fib_level_columns[f'fib_{r}'].append(levels[idx])

        # Signal logic - Buy near strong support levels (61.8%, 78.6%)
        if (high - low) > 0:  # Avoid division by zero
            if abs(last_close - levels[3]) / (high - low) < 0.02 or abs(last_close - levels[4]) / (high - low) < 0.02:
                harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Buy'
            elif abs(last_close - levels[0]) / (high - low) < 0.02 or abs(last_close - levels[1]) / (high - low) < 0.02:
                harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Sell'
            else:
                harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Hold'

    # Add fib level individual columns to DataFrame
    for r in retracement_levels:
        harga_saham[f'fib_{r}'] = fib_level_columns[f'fib_{r}']

    # Additional columns
    harga_saham['fib_high'] = fib_highs
    harga_saham['fib_low'] = fib_lows
    harga_saham['fib_levels'] = fib_levels_list

    return harga_saham


def create_db_connection():
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="",
            database="harga_saham",
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.Error as e:
        print(f"Database connection error: {e}")
        return None


def create_notification_trend_table():
    connection = create_db_connection()
    if not connection:
        print("Failed to create database connection. Table creation aborted.")
        return False
        
    cursor = connection.cursor()
    try:
        # Create main trend table with additional JSON detail column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_notification_trend_mei (
                Ticker VARCHAR(32) PRIMARY KEY,
                bollinger VARCHAR(32),
                ma VARCHAR(32),
                rsi VARCHAR(32),
                macd VARCHAR(32),
                adx VARCHAR(32),
                volume VARCHAR(32),
                candlestick VARCHAR(32),
                fibonacci VARCHAR(32),
                overall VARCHAR(32),
                trend_details JSON,
                timestamp DATETIME
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')
        
        # Create detailed trend table for complete analysis data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_notification_trend_detail_mei (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Ticker VARCHAR(32) NOT NULL,
                full_analysis_data JSON,
                indicator_calculations JSON,
                trend_analysis_details JSON,
                raw_indicator_values JSON,
                signal_history JSON,
                timestamp DATETIME,
                INDEX idx_ticker (Ticker),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')
        
        connection.commit()
        print("Tables dashboard_notification_trend_mei and dashboard_notification_trend_detail_mei verified/created successfully")
        return True
    except pymysql.Error as e:
        print(f"Error creating tables: {e}")
        return False
    finally:
        cursor.close()
        connection.close()


def create_detailed_trend_data(ticker_data, trend_result):
    """Create comprehensive detailed data for JSON storage"""
    try:
        # Helper function to convert numpy/pandas types to native Python types
        def convert_to_native_type(value):
            if pd.isna(value):
                return None
            elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype)):
                return None if pd.isna(value) else value.item()
            elif hasattr(value, 'item'):  # numpy types
                return value.item()
            elif isinstance(value, (int, float, str, bool)):
                return value
            else:
                return str(value)  # Convert unknown types to string
        
        # Get latest values for each indicator
        latest_data = ticker_data.tail(1).iloc[0] if not ticker_data.empty else {}
        
        # Full analysis data
        full_analysis = {
            "ticker": str(latest_data.get('Ticker', '')),
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_points_analyzed": int(len(ticker_data)),
            "price_data": {
                "current_price": convert_to_native_type(latest_data.get('Close', 0)),
                "open": convert_to_native_type(latest_data.get('Open', 0)),
                "high": convert_to_native_type(latest_data.get('High', 0)),
                "low": convert_to_native_type(latest_data.get('Low', 0)),
                "volume": convert_to_native_type(latest_data.get('Volume', 0))
            }
        }
        
        # Indicator calculations
        indicator_calculations = {
            "bollinger_bands": {
                "upper_band": convert_to_native_type(latest_data.get('Upper Band', 0)),
                "lower_band": convert_to_native_type(latest_data.get('Lower Band', 0)),
                "middle_band": convert_to_native_type(latest_data.get('Middle Band', 0)),
                "bandwidth": convert_to_native_type((latest_data.get('Upper Band', 0) - latest_data.get('Lower Band', 0)) / latest_data.get('Middle Band', 1)) if not pd.isna(latest_data.get('Upper Band', 0)) and latest_data.get('Middle Band', 1) != 0 else None,
                "signal": str(latest_data.get('Signal', 'N/A'))
            },
            "moving_averages": {
                "short_ma": convert_to_native_type(latest_data.get('short_MA', 0)),
                "long_ma": convert_to_native_type(latest_data.get('long_MA', 0)),
                "crossover_status": "bullish" if convert_to_native_type(latest_data.get('short_MA', 0)) > convert_to_native_type(latest_data.get('long_MA', 0)) else "bearish",
                "signal": str(latest_data.get('MaSignal', 'N/A'))
            },
            "rsi": {
                "value": convert_to_native_type(latest_data.get('RSI', 0)),
                "condition": "overbought" if convert_to_native_type(latest_data.get('RSI', 50)) > 70 else "oversold" if convert_to_native_type(latest_data.get('RSI', 50)) < 30 else "neutral",
                "signal": str(latest_data.get('RsiSignal', 'N/A'))
            },
            "macd": {
                "macd_line": convert_to_native_type(latest_data.get('MACD', 0)),
                "signal_line": convert_to_native_type(latest_data.get('Signal_Line', 0)),
                "histogram": convert_to_native_type(latest_data.get('MACD_Hist', 0)),
                "crossover_status": "bullish" if convert_to_native_type(latest_data.get('MACD', 0)) > convert_to_native_type(latest_data.get('Signal_Line', 0)) else "bearish",
                "signal": str(latest_data.get('MacdSignal', 'N/A'))
            },
            "adx": {
                "adx_value": convert_to_native_type(latest_data.get('ADX', 0)),
                "plus_di": convert_to_native_type(latest_data.get('+DI', 0)),
                "minus_di": convert_to_native_type(latest_data.get('-DI', 0)),
                "trend_strength": "strong" if convert_to_native_type(latest_data.get('ADX', 0)) > 25 else "moderate" if convert_to_native_type(latest_data.get('ADX', 0)) > 20 else "weak",
                "signal": str(latest_data.get('AdxSignal', 'N/A'))
            },
            "volume": {
                "current_volume": convert_to_native_type(latest_data.get('Volume', 0)),
                "volume_ma": convert_to_native_type(latest_data.get('VMA', 0)),
                "volume_ratio": convert_to_native_type(latest_data.get('Volume', 0) / latest_data.get('VMA', 1)) if convert_to_native_type(latest_data.get('VMA', 0)) > 0 else None,
                "signal": str(latest_data.get('VolumeSignal', 'N/A'))
            },
            "fibonacci": {
                "current_signal": str(latest_data.get('Fibonacci_Signal', 'N/A')),
                "fib_high": convert_to_native_type(latest_data.get('fib_high', 0)),
                "fib_low": convert_to_native_type(latest_data.get('fib_low', 0)),
                "levels": {
                    "23.6%": convert_to_native_type(latest_data.get('fib_0.236', 0)),
                    "38.2%": convert_to_native_type(latest_data.get('fib_0.382', 0)),
                    "50.0%": convert_to_native_type(latest_data.get('fib_0.5', 0)),
                    "61.8%": convert_to_native_type(latest_data.get('fib_0.618', 0)),
                    "78.6%": convert_to_native_type(latest_data.get('fib_0.786', 0))
                }
            },
            "candlestick": {
                "current_trend": str(latest_data.get('candlestick_trend', 'N/A')),
                "detected_patterns": str(latest_data.get('detected_patterns', 'N/A')),
                "pattern_strength": str(latest_data.get('candlestick_confidence', 'N/A')),
                "bullish_patterns_count": convert_to_native_type(latest_data.get('bullish_patterns', 0)),
                "bearish_patterns_count": convert_to_native_type(latest_data.get('bearish_patterns', 0)),
                "neutral_patterns_count": convert_to_native_type(latest_data.get('neutral_patterns', 0)),
                "signal": str(latest_data.get('Candlestick_Signal', 'N/A')),
                "pattern_details": {
                    "bullish_patterns": {
                        "three_white_soldiers": convert_to_native_type(latest_data.get('CDL3WHITESOLDIERS', 0)),
                        "morning_star": convert_to_native_type(latest_data.get('CDLMORNINGSTAR', 0)),
                        "piercing": convert_to_native_type(latest_data.get('CDLPIERCING', 0)),
                        "hammer": convert_to_native_type(latest_data.get('CDLHAMMER', 0)),
                        "inverted_hammer": convert_to_native_type(latest_data.get('CDLINVERTEDHAMMER', 0)),
                        "bullish_engulfing": convert_to_native_type(latest_data.get('CDLENGULFING_bullish', 0)),
                        "bullish_harami": convert_to_native_type(latest_data.get('CDLHARAMI_bullish', 0)),
                        "bullish_hikkake": convert_to_native_type(latest_data.get('CDLHIKKAKE_bullish', 0)),
                        "morning_doji_star": convert_to_native_type(latest_data.get('CDLMORNINGDOJISTAR', 0)),
                        "on_neck": convert_to_native_type(latest_data.get('CDLONNECK', 0)),
                        "thrusting": convert_to_native_type(latest_data.get('CDLTHRUSTING', 0))
                    },
                    "bearish_patterns": {
                        "three_black_crows": convert_to_native_type(latest_data.get('CDL3BLACKCROWS', 0)),
                        "evening_star": convert_to_native_type(latest_data.get('CDLEVENINGSTAR', 0)),
                        "shooting_star": convert_to_native_type(latest_data.get('CDLSHOOTINGSTAR', 0)),
                        "hanging_man": convert_to_native_type(latest_data.get('CDLHANGINGMAN', 0)),
                        "bearish_engulfing": convert_to_native_type(latest_data.get('CDLENGULFING_bearish', 0)),
                        "bearish_harami": convert_to_native_type(latest_data.get('CDLHARAMI_bearish', 0)),
                        "dark_cloud_cover": convert_to_native_type(latest_data.get('CDLDARKCLOUDCOVER', 0)),
                        "bearish_hikkake": convert_to_native_type(latest_data.get('CDLHIKKAKE_bearish', 0)),
                        "evening_doji_star": convert_to_native_type(latest_data.get('CDLEVENINGDOJISTAR', 0)),
                        "stalled_pattern": convert_to_native_type(latest_data.get('CDLSTALLEDPATTERN', 0)),
                        "gravestone_doji": convert_to_native_type(latest_data.get('CDLGRAVESTONEDOJI', 0))
                    },
                    "neutral_patterns": {
                        "doji": convert_to_native_type(latest_data.get('CDLDOJI', 0)),
                        "spinning_top": convert_to_native_type(latest_data.get('CDLSPINNINGTOP', 0)),
                        "marubozu": convert_to_native_type(latest_data.get('CDLMARUBOZU', 0)),
                        "takuri": convert_to_native_type(latest_data.get('CDLTAKURI', 0)),
                        "rickshaw_man": convert_to_native_type(latest_data.get('CDLRICKSHAWMAN', 0))
                    }
                }
            }
        }
        
        # Trend analysis details
        trend_analysis_details = {
            "individual_trends": {k: str(v) for k, v in trend_result.items()},  # Convert all trend values to strings
            "trend_scoring": {
                "total_indicators": int(len([t for t in trend_result.values() if t not in ['Unknown', 'Insufficient Data']])),
                "bullish_indicators": int(len([t for t in trend_result.values() if 'Uptrend' in str(t)])),
                "bearish_indicators": int(len([t for t in trend_result.values() if 'Downtrend' in str(t)])),
                "neutral_indicators": int(len([t for t in trend_result.values() if t in ['Sideways', 'Neutral', 'Hold']]))
            },
            "confidence_level": "high" if len([t for t in trend_result.values() if 'Strong' in str(t)]) > 3 else "medium" if len([t for t in trend_result.values() if t not in ['Unknown', 'Insufficient Data']]) > 5 else "low"
        }
        
        # Raw indicator values (last 5 data points for trending)
        raw_values = {}
        if len(ticker_data) >= 5:
            recent_data = ticker_data.tail(5)
            
            # Price and basic indicators
            basic_indicators = ['Close', 'Volume', 'RSI', 'MACD', 'ADX']
            for col in basic_indicators:
                if col in recent_data.columns:
                    raw_values[col] = [convert_to_native_type(x) for x in recent_data[col].fillna(0).tolist()]
            
            # Bollinger Bands
            bb_indicators = ['Upper Band', 'Middle Band', 'Lower Band']
            for col in bb_indicators:
                if col in recent_data.columns:
                    raw_values[col.replace(' ', '_').lower()] = [convert_to_native_type(x) for x in recent_data[col].fillna(0).tolist()]
            
            # Moving Averages
            ma_indicators = ['short_MA', 'long_MA']
            for col in ma_indicators:
                if col in recent_data.columns:
                    raw_values[col] = [convert_to_native_type(x) for x in recent_data[col].fillna(0).tolist()]
            
            # MACD components
            macd_indicators = ['Signal_Line', 'MACD_Hist']
            for col in macd_indicators:
                if col in recent_data.columns:
                    raw_values[col] = [convert_to_native_type(x) for x in recent_data[col].fillna(0).tolist()]
            
            # ADX components
            adx_indicators = ['+DI', '-DI']
            for col in adx_indicators:
                if col in recent_data.columns:
                    raw_values[col.replace('+', 'plus_').replace('-', 'minus_')] = [convert_to_native_type(x) for x in recent_data[col].fillna(0).tolist()]
            
            # Volume indicators
            if 'VMA' in recent_data.columns:
                raw_values['VMA'] = [convert_to_native_type(x) for x in recent_data['VMA'].fillna(0).tolist()]
            
            # Fibonacci levels (current only - too many for 5 data points)
            fib_current = {
                'fib_high': convert_to_native_type(recent_data['fib_high'].iloc[-1]) if 'fib_high' in recent_data.columns else None,
                'fib_low': convert_to_native_type(recent_data['fib_low'].iloc[-1]) if 'fib_low' in recent_data.columns else None
            }
            raw_values['fibonacci_current'] = fib_current

        # Signal history (last 10 signals for each indicator) - SUDAH LENGKAP
        signal_history = {}
        if len(ticker_data) >= 10:
            recent_signals = ticker_data.tail(10)
            signal_cols = ['Signal', 'MaSignal', 'RsiSignal', 'MacdSignal', 'AdxSignal', 'VolumeSignal', 'Fibonacci_Signal']
            
            # Add Candlestick signal if exists
            if 'Candlestick_Signal' in recent_signals.columns:
                signal_cols.append('Candlestick_Signal')
            
            for col in signal_cols:
                if col in recent_signals.columns:
                    signal_history[col] = [str(x) if x is not None else 'N/A' for x in recent_signals[col].fillna('N/A').tolist()]
        
        return {
            "full_analysis_data": full_analysis,
            "indicator_calculations": indicator_calculations,
            "trend_analysis_details": trend_analysis_details,
            "raw_indicator_values": raw_values,
            "signal_history": signal_history
        }
        
    except Exception as e:
        print(f"Error creating detailed trend data: {e}")
        return {}


def save_trend_to_db(ticker, trend_dict, ticker_data=None):
    connection = create_db_connection()
    if not connection:
        print(f"Failed to save trend for {ticker} - database connection error")
        return False
        
    cursor = connection.cursor()
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create detailed data for JSON storage
        detailed_data = create_detailed_trend_data(ticker_data, trend_dict) if ticker_data is not None else {}
        
        # Prepare trend details JSON for main table (summary)
        trend_details_json = {
            "analysis_summary": {
                "total_indicators": len(trend_dict) - 1,  # Exclude 'overall'
                "overall_trend": trend_dict.get('overall', 'Unknown'),
                "bullish_count": len([t for t in trend_dict.values() if 'Uptrend' in str(t)]),
                "bearish_count": len([t for t in trend_dict.values() if 'Downtrend' in str(t)]),
                "analysis_timestamp": now
            },
            "indicator_trends": {k: v for k, v in trend_dict.items() if k != 'overall'}
        }
        
        # Check if ticker already exists in main table
        cursor.execute("SELECT Ticker FROM dashboard_notification_trend_mei WHERE Ticker = %s", (ticker,))
        exists = cursor.fetchone()
        
        # Update or insert main trend table
        if exists:
            sql = '''
                UPDATE dashboard_notification_trend_mei
                SET bollinger=%s, ma=%s, rsi=%s, macd=%s, adx=%s, volume=%s, candlestick=%s, fibonacci=%s, overall=%s, trend_details=%s, timestamp=%s
                WHERE Ticker=%s
            '''
            params = (
                trend_dict.get('bollinger'),
                trend_dict.get('ma'),
                trend_dict.get('rsi'),
                trend_dict.get('macd'),
                trend_dict.get('adx'),
                trend_dict.get('volume'),
                trend_dict.get('candlestick'),
                trend_dict.get('fibonacci'),
                trend_dict.get('overall'),
                json.dumps(trend_details_json, ensure_ascii=False),
                now,
                ticker
            )
            print(f"[↻] Updating existing trend data for {ticker}")
        else:
            sql = '''
                INSERT INTO dashboard_notification_trend_mei
                (Ticker, bollinger, ma, rsi, macd, adx, volume, candlestick, fibonacci, overall, trend_details, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            params = (
                ticker,
                trend_dict.get('bollinger'),
                trend_dict.get('ma'),
                trend_dict.get('rsi'),
                trend_dict.get('macd'),
                trend_dict.get('adx'),
                trend_dict.get('volume'),
                trend_dict.get('candlestick'),
                trend_dict.get('fibonacci'),
                trend_dict.get('overall'),
                json.dumps(trend_details_json, ensure_ascii=False),
                now
            )
            print(f"[+] Inserting new trend data for {ticker}")
        
        cursor.execute(sql, params)
        
        # Insert detailed data into detail table (always insert new record for history)
        if detailed_data:
            detail_sql = '''
                INSERT INTO dashboard_notification_trend_detail_mei
                (Ticker, full_analysis_data, indicator_calculations, trend_analysis_details, raw_indicator_values, signal_history, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''
            detail_params = (
                ticker,
                json.dumps(detailed_data.get('full_analysis_data', {}), ensure_ascii=False),
                json.dumps(detailed_data.get('indicator_calculations', {}), ensure_ascii=False),
                json.dumps(detailed_data.get('trend_analysis_details', {}), ensure_ascii=False),
                json.dumps(detailed_data.get('raw_indicator_values', {}), ensure_ascii=False),
                json.dumps(detailed_data.get('signal_history', {}), ensure_ascii=False),
                now
            )
            cursor.execute(detail_sql, detail_params)
            print(f"[📊] Detailed analysis data saved for {ticker}")
        
        connection.commit()
        return True
    except pymysql.Error as e:
        print(f"[✗] Database error saving trend for {ticker}: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()


def get_ticker_data(ticker, required_length=60):
    connection = create_db_connection()
    if not connection:
        print(f"Failed to get data for {ticker} - database connection error")
        return pd.DataFrame()
        
    cursor = connection.cursor()

    try:
        query = "SELECT Data FROM data_saham_max_all_mei WHERE Ticker = %s"
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()

        if result and result['Data']:
            try:
                data_json = json.loads(result['Data'])
                df = pd.DataFrame(data_json)

                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.sort_values('Date', ascending=True, inplace=True)
                    
                    # Ambil lebih banyak data untuk memastikan indicator bisa dihitung dengan benar
                    # Fibonacci butuh 60 + MA panjang (50) + buffer = ~120 data minimum
                    min_data_needed = max(required_length + 60, 120)  # Buffer untuk semua indicator
                    df = df.tail(min_data_needed)  
                    
                    df.set_index('Date', inplace=True)

                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df['Ticker'] = ticker
                print(f"[✓] Data for {ticker} successfully loaded ({len(df)} rows).")
                return df
            except json.JSONDecodeError:
                print(f"[!] Invalid JSON data for {ticker}")
                return pd.DataFrame()
            except Exception as e:
                print(f"[!] Error processing data for {ticker}: {e}")
                return pd.DataFrame()
        else:
            print(f"[!] No data found for {ticker}")
            return pd.DataFrame()

    except pymysql.Error as e:
        print(f"[!] Database error retrieving data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        connection.close()


def apply_all_indicators(df):
    """Apply all technical indicators to the dataframe"""
    try:
        # Apply each indicator one by one
        df = calculate_bollinger_bands(df)
        df = calculate_ma(df)
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_adx(df)
        df = calculate_volume(df)
        df = calculate_fibonacci_retracement(df)
        
        # Apply candlestick pattern detection
        df = detect_candlestick_patterns(df)
        
        return df
    except Exception as e:
        print(f"Error applying indicators: {e}")
        return df


def analyze_latest_trend(ticker_data, lookback_periods={
    'bollinger': 20,
    'ma': 50,  # For long MA comparison
    'rsi': 14,
    'macd': 26, # Use longest MACD period
    'adx': 14,
    'volume': 20,
    'candlestick': 10,
    'fibonacci': 20
}): #lookback periods digunakan untuk penentuan trend terakhir
    """
    Analyze trend based on latest data points for each indicator
    """
    if ticker_data.empty:
        return None
        
    latest_data = ticker_data.copy()
    trends = {}
    
    # Bollinger Bands Trend Analysis
    if all(col in latest_data.columns for col in ['Upper Band', 'Lower Band', 'Middle Band']):
        bb_data = latest_data.tail(lookback_periods['bollinger'])
        price_vs_middle = bb_data['Close'].iloc[-1] > bb_data['Middle Band'].iloc[-1]
        bandwidth = (bb_data['Upper Band'].iloc[-1] - bb_data['Lower Band'].iloc[-1]) / bb_data['Middle Band'].iloc[-1]
        
        if bb_data['Close'].iloc[-1] > bb_data['Upper Band'].iloc[-1]:
            trends['bollinger'] = 'Strong Uptrend'
        elif bb_data['Close'].iloc[-1] < bb_data['Lower Band'].iloc[-1]:
            trends['bollinger'] = 'Strong Downtrend'
        elif price_vs_middle and bandwidth > 0.2:
            trends['bollinger'] = 'Uptrend'
        elif not price_vs_middle and bandwidth > 0.2:
            trends['bollinger'] = 'Downtrend'
        else:
            trends['bollinger'] = 'Sideways'

    # Moving Average Trend Analysis
    if all(col in latest_data.columns for col in ['short_MA', 'long_MA']):
        ma_data = latest_data.tail(lookback_periods['ma'])
        short_ma_trend = ma_data['short_MA'].diff().mean() > 0
        long_ma_trend = ma_data['long_MA'].diff().mean() > 0
        current_cross = ma_data['short_MA'].iloc[-1] > ma_data['long_MA'].iloc[-1]
        
        if current_cross and short_ma_trend and long_ma_trend:
            trends['ma'] = 'Strong Uptrend'
        elif not current_cross and not short_ma_trend and not long_ma_trend:
            trends['ma'] = 'Strong Downtrend'
        elif current_cross:
            trends['ma'] = 'Uptrend'
        elif not current_cross:
            trends['ma'] = 'Downtrend'
        else:
            trends['ma'] = 'Sideways'

    # RSI Trend Analysis
    if 'RSI' in latest_data.columns:
        rsi_data = latest_data.tail(lookback_periods['rsi'])
        latest_rsi = rsi_data['RSI'].iloc[-1]
        rsi_trend = rsi_data['RSI'].diff().mean() > 0
        
        if latest_rsi > 70:
            trends['rsi'] = 'Overbought'
        elif latest_rsi < 30:
            trends['rsi'] = 'Oversold'
        elif latest_rsi > 50 and rsi_trend:
            trends['rsi'] = 'Uptrend'
        elif latest_rsi < 50 and not rsi_trend:
            trends['rsi'] = 'Downtrend'
        else:
            trends['rsi'] = 'Neutral'

    # MACD Trend Analysis
    if all(col in latest_data.columns for col in ['MACD', 'Signal_Line', 'MACD_Hist']):
        macd_data = latest_data.tail(lookback_periods['macd'])
        hist_trend = macd_data['MACD_Hist'].diff().mean() > 0
        macd_above_signal = macd_data['MACD'].iloc[-1] > macd_data['Signal_Line'].iloc[-1]
        hist_positive = macd_data['MACD_Hist'].iloc[-1] > 0
        
        if macd_above_signal and hist_positive and hist_trend:
            trends['macd'] = 'Strong Uptrend'
        elif not macd_above_signal and not hist_positive and not hist_trend:
            trends['macd'] = 'Strong Downtrend'
        elif macd_above_signal:
            trends['macd'] = 'Uptrend'
        elif not macd_above_signal:
            trends['macd'] = 'Downtrend'
        else:
            trends['macd'] = 'Neutral'

    # ADX Trend Analysis
    if all(col in latest_data.columns for col in ['ADX', '+DI', '-DI']):
        adx_data = latest_data.tail(lookback_periods['adx'])
        latest_adx = adx_data['ADX'].iloc[-1]
        di_plus_greater = adx_data['+DI'].iloc[-1] > adx_data['-DI'].iloc[-1]
        
        if latest_adx > 25:
            if di_plus_greater:
                trends['adx'] = 'Strong Uptrend'
            else:
                trends['adx'] = 'Strong Downtrend'
        elif latest_adx > 20:
            if di_plus_greater:
                trends['adx'] = 'Uptrend'
            else:
                trends['adx'] = 'Downtrend'
        else:
            trends['adx'] = 'No Trend'

    # Volume Trend Analysis
    if all(col in latest_data.columns for col in ['Volume', 'VMA']):
        volume_data = latest_data.tail(lookback_periods['volume'])
        vol_trend = volume_data['Volume'].diff().mean() > 0
        above_avg = volume_data['Volume'].iloc[-1] > volume_data['VMA'].iloc[-1]
        
        if above_avg and vol_trend:
            trends['volume'] = 'Strong Volume'
        elif above_avg:
            trends['volume'] = 'Above Average'
        elif not above_avg and not vol_trend:
            trends['volume'] = 'Weak Volume'
        else:
            trends['volume'] = 'Average Volume'

    # Candlestick Trend Analysis
    if 'candlestick_trend' in latest_data.columns:
        candlestick_data = latest_data.tail(lookback_periods['candlestick'])
        latest_trend = candlestick_data['candlestick_trend'].iloc[-1]
        
        # Count recent trend occurrences
        recent_trends = candlestick_data['candlestick_trend'].value_counts()
        
        if latest_trend == 'Uptrend' and recent_trends.get('Uptrend', 0) > len(candlestick_data) * 0.6:
            trends['candlestick'] = 'Strong Uptrend'
        elif latest_trend == 'Downtrend' and recent_trends.get('Downtrend', 0) > len(candlestick_data) * 0.6:
            trends['candlestick'] = 'Strong Downtrend'
        elif latest_trend == 'Uptrend':
            trends['candlestick'] = 'Uptrend'
        elif latest_trend == 'Downtrend':
            trends['candlestick'] = 'Downtrend'
        else:
            trends['candlestick'] = 'Sideways'

    # Fibonacci Trend Analysis
    if 'Fibonacci_Signal' in latest_data.columns:
        fib_data = latest_data.tail(lookback_periods['fibonacci'])
        recent_signals = fib_data['Fibonacci_Signal'].value_counts()
        latest_signal = fib_data['Fibonacci_Signal'].iloc[-1]
        
        # Analyze Fibonacci level interactions
        buy_signals = recent_signals.get('Buy', 0)
        sell_signals = recent_signals.get('Sell', 0)
        hold_signals = recent_signals.get('Hold', 0)
        
        if buy_signals > sell_signals * 1.5:
            trends['fibonacci'] = 'Strong Uptrend'
        elif sell_signals > buy_signals * 1.5:
            trends['fibonacci'] = 'Strong Downtrend'
        elif buy_signals > sell_signals:
            trends['fibonacci'] = 'Uptrend'
        elif sell_signals > buy_signals:
            trends['fibonacci'] = 'Downtrend'
        else:
            trends['fibonacci'] = 'Neutral'

    # Calculate overall trend
    trend_scores = {
        'Strong Uptrend': 2,
        'Uptrend': 1,
        'Sideways': 0,
        'Neutral': 0,
        'Downtrend': -1,
        'Strong Downtrend': -2,
        'Overbought': 1,
        'Oversold': -1,
        'Strong Volume': 1,
        'Weak Volume': -1,
        'Above Average': 0.5,
        'Average Volume': 0,
        'No Trend': 0
    }
    
    total_score = sum(trend_scores.get(trend, 0) for trend in trends.values())
    num_indicators = len(trends)
    
    if num_indicators > 0:
        avg_score = total_score / num_indicators
        if avg_score > 1:
            overall_trend = 'Strong Uptrend'
        elif avg_score > 0.3:
            overall_trend = 'Uptrend'
        elif avg_score < -1:
            overall_trend = 'Strong Downtrend'
        elif avg_score < -0.3:
            overall_trend = 'Downtrend'
        else:
            overall_trend = 'Sideways'
    else:
        overall_trend = 'Unknown'

    trends['overall'] = overall_trend
    
    return trends


def process_ticker(ticker):
    try:
        print(f"\n[⏳] Processing ticker: {ticker}")
        # Get stock data
        df = get_ticker_data(ticker)
        
        if df.empty:
            print(f"[!] No data available for {ticker}, skipping...")
            return
            
        # Apply all technical indicators
        df = apply_all_indicators(df)
        
        # Analyze trends based on the indicators
        trend_result = analyze_latest_trend(df)
        
        if trend_result:
            # Save results to database (will update if already exists) - pass df for detailed data
            success = save_trend_to_db(ticker, trend_result, df)
            if success:
                print(f"[✓] Trend for {ticker} saved successfully. Overall: {trend_result['overall']}")
                print(f"    Candlestick: {trend_result.get('candlestick', 'N/A')}, Fibonacci: {trend_result.get('fibonacci', 'N/A')}")
            else:
                print(f"[✗] Failed to save trend for {ticker}")
        else:
            print(f"[!] Trend analysis failed for {ticker}")
    except Exception as e:
        print(f"[✗] Error processing {ticker}: {e}")


def analyze_and_save_all_tickers_trend():
    print("Starting trend analysis for all tickers...")
    
    # First create/verify table exists
    if not create_notification_trend_table():
        print("Failed to verify/create database table. Aborting analysis.")
        return
        
    print(f"Processing {len(AVAILABLE_TICKERS)} tickers...")
    
    # Process each ticker
    for ticker in AVAILABLE_TICKERS:
        process_ticker(ticker)
    
    print("Trend analysis completed for all tickers.")


# Run the program
if __name__ == "__main__":
    analyze_and_save_all_tickers_trend()