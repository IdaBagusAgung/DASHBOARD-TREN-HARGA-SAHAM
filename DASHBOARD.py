import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash import dash_table
import plotly.graph_objs as go
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymysql
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import time
import traceback
import logging
import plotly.express as px

# awadawdada

# from get_data import get_stock_data
# from get_data_max import get_stock_data_max
from get_backtesting_data import get_backtesting_data
from Ticker import AVAILABLE_TICKERS
from Explanation import indicator_explanations
from Candlestick_pattern import detect_candlestick_patterns, get_candlestick_explanations, get_candlestick_explanations, get_pattern_description
from bulk_analysis import render_bulk_analysis_page, register_backtesting_callbacks


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database Functions
def create_db_connection():
    """
    Create a connection to the database using pymysql.
    """
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="harga_saham",
        cursorclass=pymysql.cursors.DictCursor
    )


def get_all_trends_from_db():
    connection = create_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT * FROM dashboard_notification_trend_mei')
        rows = cursor.fetchall()
        all_trends = {}
        for row in rows:
            ticker = row['Ticker']
            all_trends[ticker] = {
                'bollinger': row.get('bollinger'),
                'ma': row.get('ma'),
                'rsi': row.get('rsi'),
                'macd': row.get('macd'),
                'adx': row.get('adx'),
                'volume': row.get('volume'),
                'fibonacci': row.get('fibonacci'),
                'candlestick': row.get('candlestick'),
                'overall': row.get('overall')
            }
        return all_trends
    finally:
        cursor.close()
        connection.close()


# Initialize Dash app with new className
app = Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ]
)
app.title = "Dashboard Analisis Saham"


# Global variable to store the current dataset
current_dataset = None




# Add callback to set default ticker
@app.callback(
    Output('ticker-dropdown', 'value'),
    [Input('ticker-dropdown', 'options')]
)
def set_default_ticker(available_options):
    return 'BBCA.JK'




def load_and_process_data(period='1y', ticker=None, interval='1min'):
    """
    Load and process data based on selected period, ticker and interval
    """
    try:
        if not ticker:
            return pd.DataFrame()
            
        connection = create_db_connection()
        cursor = connection.cursor()
        
        # Base query to get minute data
        base_sql = f"SELECT Data FROM data_saham_minute_all_april WHERE Ticker = '{ticker}'"
        
        if period == 'realtime':
            # Get minute data
            cursor.execute(base_sql)
            result = cursor.fetchone()
            
            if not result:
                print(f"No data found for ticker {ticker}")
                return pd.DataFrame()
                
            # Convert JSON data to DataFrame
            df = pd.DataFrame(json.loads(result['Data']))
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Resample data based on selected interval
            if interval != '1min':
                # Define resample rules
                resample_rules = {
                    '5min': '5T',
                    '10min': '10T',
                    '15min': '15T',
                    '30min': '30T',
                    '45min': '45T',
                    '1h': 'H',
                    '1d': 'D',
                    '1w': 'W',
                    '1m': 'M'
                }
                
                rule = resample_rules.get(interval, '1T')
                
                # Resample OHLCV data
                resampled = df.resample(rule, on='Datetime').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                # Reset index to make Datetime a column again
                resampled = resampled.reset_index()
                df = resampled
                
        elif period == '1y':
            sql = f"SELECT Data FROM data_saham_hour_all_mei WHERE Ticker = '{ticker}'"
            cursor.execute(sql)
            result = cursor.fetchone()
            df = pd.DataFrame(json.loads(result['Data']))
        else:
            # sql = f"SELECT Data FROM data_saham_max_all_mei WHERE Ticker = '{ticker}'"
            # cursor.execute(sql)
            # result = cursor.fetchone()
            # df = pd.DataFrame(json.loads(result['Data']))

            sql = f"SELECT Data FROM data_saham_max_all_mei WHERE Ticker = '{ticker}'"
            cursor.execute(sql)
            result = cursor.fetchone()

            # Load data JSON ke DataFrame
            df = pd.DataFrame(json.loads(result['Data']))

            df.rename(columns={'Date': 'Datetime'}, inplace=True)

            # Pastikan kolom 'Datetime' dalam format datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'])

            # Cari tanggal terakhir dalam dataset
            last_date = df['Datetime'].max()

            # Hitung batas waktu dua tahun ke belakang dari tanggal terakhir
            start_date = last_date - timedelta(days=1*365)

            # Filter data hanya dua tahun terakhir
            df = df[df['Datetime'] >= start_date]



            
        # Handle date column names
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        
        # Ensure datetime conversion
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        if df['Datetime'].dt.tz is not None:
            df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        
        # Standardize column names
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Add ticker column
        df['Ticker'] = ticker
        
        # Calculate technical indicators
        if not df.empty:
            df = calculate_bollinger_bands(df)
            df = calculate_ma(df)
            df = calculate_rsi(df)
            df = calculate_macd(df)
            df = calculate_adx(df)
            df = calculate_volume(df)

            df = calculate_fibonacci_retracement(df)
            
            # Add candlestick pattern detection
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                df = detect_candlestick_patterns(df)
        
        return df
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        return pd.DataFrame()
        
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()





def calculate_bollinger_bands(harga_saham, window=20, num_std=2, signal_mode='touch'):
    # Initialize signal column first
    harga_saham['Signal'] = None
    
    # Calculate Bollinger Bands
    harga_saham['Middle Band'] = harga_saham['Close'].rolling(window=window).mean()
    harga_saham['STD'] = harga_saham['Close'].rolling(window=window).std()
    harga_saham['Upper Band'] = harga_saham['Middle Band'] + (num_std * harga_saham['STD'])
    harga_saham['Lower Band'] = harga_saham['Middle Band'] - (num_std * harga_saham['STD'])
    
    # Generate signals only where we have complete Bollinger Bands data
    mask = ~harga_saham[['Upper Band', 'Lower Band', 'Middle Band']].isna().any(axis=1)
    
    if signal_mode == 'touch':
        # Original logic - touch the bands
        harga_saham.loc[mask & (harga_saham['Close'] >= harga_saham['Upper Band']), 'Signal'] = 'Sell'
        harga_saham.loc[mask & (harga_saham['Close'] <= harga_saham['Lower Band']), 'Signal'] = 'Buy'
    elif signal_mode == 'cross':
        # New logic - cross/penetrate the bands
        # Check if price crosses above upper band (from below)
        prev_below_upper = harga_saham['Close'].shift(1) < harga_saham['Upper Band'].shift(1)
        current_above_upper = harga_saham['Close'] > harga_saham['Upper Band']
        cross_above_upper = prev_below_upper & current_above_upper
        
        # Check if price crosses below lower band (from above)
        prev_above_lower = harga_saham['Close'].shift(1) > harga_saham['Lower Band'].shift(1)
        current_below_lower = harga_saham['Close'] < harga_saham['Lower Band']
        cross_below_lower = prev_above_lower & current_below_lower
        
        harga_saham.loc[mask & cross_above_upper, 'Signal'] = 'Sell'
        harga_saham.loc[mask & cross_below_lower, 'Signal'] = 'Buy'

    harga_saham.loc[mask & harga_saham['Signal'].isna(), 'Signal'] = 'Hold'
    
    return harga_saham

# Update the calculate_ma function to accept short_period and long_period parameters
def calculate_ma(harga_saham, short_period=20, long_period=50, medium_period=35, ma_lines='2'):
    # Initialize signal column
    harga_saham['MaSignal'] = None
    
    # Calculate MAs
    harga_saham['short_MA'] = harga_saham['Close'].rolling(window=short_period).mean()
    harga_saham['long_MA'] = harga_saham['Close'].rolling(window=long_period).mean()
    
    if ma_lines == '3':
        # Calculate medium MA for 3-line mode
        harga_saham['medium_MA'] = harga_saham['Close'].rolling(window=medium_period).mean()
        
        # Generate signals for 3-MA system
        mask = ~harga_saham[['short_MA', 'medium_MA', 'long_MA']].isna().any(axis=1)
        
        # Bullish: short > medium > long (ascending order)
        bullish_alignment = (
            (harga_saham['short_MA'] > harga_saham['medium_MA']) & 
            (harga_saham['medium_MA'] > harga_saham['long_MA'])
        )
        
        # Bearish: short < medium < long (descending order)
        bearish_alignment = (
            (harga_saham['short_MA'] < harga_saham['medium_MA']) & 
            (harga_saham['medium_MA'] < harga_saham['long_MA'])
        )
        
        harga_saham.loc[mask & bullish_alignment, 'MaSignal'] = 'Buy'
        harga_saham.loc[mask & bearish_alignment, 'MaSignal'] = 'Sell'
        harga_saham.loc[mask & harga_saham['MaSignal'].isna(), 'MaSignal'] = 'Hold'
        
    else:
        # Original 2-MA system
        mask = ~harga_saham[['short_MA', 'long_MA']].isna().any(axis=1)
        
        harga_saham.loc[mask & (harga_saham['short_MA'] > harga_saham['long_MA']), 'MaSignal'] = 'Buy'
        harga_saham.loc[mask & (harga_saham['short_MA'] < harga_saham['long_MA']), 'MaSignal'] = 'Sell'
        harga_saham.loc[mask & harga_saham['MaSignal'].isna(), 'MaSignal'] = 'Hold'
    
    print(f"DEBUG MA: Columns in dataframe: {harga_saham.columns.tolist()}")
    
    return harga_saham

@app.callback(
    Output('ma-medium-container', 'style'),
    [Input('ma-lines-count', 'value')]
)
def toggle_medium_ma_parameter(ma_lines):
    if ma_lines == '3':
        return {'display': 'block'}
    return {'display': 'none'}


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



# def calculate_macd(harga_saham, short_window=12, long_window=26, signal_window=9):
#     # Initialize signal column
#     harga_saham['MacdSignal'] = None
    
#     # Calculate MACD components with custom periods
#     harga_saham['EMA_short'] = harga_saham['Close'].ewm(span=short_window, adjust=False).mean()
#     harga_saham['EMA_long'] = harga_saham['Close'].ewm(span=long_window, adjust=False).mean()
#     harga_saham['MACD'] = harga_saham['EMA_short'] - harga_saham['EMA_long']
#     harga_saham['Signal_Line'] = harga_saham['MACD'].ewm(span=signal_window, adjust=False).mean()
#     harga_saham['MACD_Hist'] = harga_saham['MACD'] - harga_saham['Signal_Line']
    
#     # Generate signals only where we have complete MACD data
#     mask = ~harga_saham[['MACD', 'Signal_Line']].isna().any(axis=1)
    
#     # Update signals where we have complete data
#     harga_saham.loc[mask & (harga_saham['MACD'] > harga_saham['Signal_Line']), 'MacdSignal'] = 'Buy'
#     harga_saham.loc[mask & (harga_saham['MACD'] < harga_saham['Signal_Line']), 'MacdSignal'] = 'Sell'
#     harga_saham.loc[mask & harga_saham['MacdSignal'].isna(), 'MacdSignal'] = 'Hold'
    
#     return harga_saham


def calculate_macd(harga_saham, short_window=12, long_window=26, signal_window=9):
    harga_saham = harga_saham.copy()
    
    # Hitung EMA pendek dan panjang
    harga_saham['EMA_short'] = harga_saham['Close'].ewm(span=short_window, adjust=False).mean()
    harga_saham['EMA_long'] = harga_saham['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Hitung MACD dan signal line
    harga_saham['MACD'] = harga_saham['EMA_short'] - harga_saham['EMA_long']
    harga_saham['Signal_Line'] = harga_saham['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    # Hitung MACD Histogram
    harga_saham['MACD_Hist'] = harga_saham['MACD'] - harga_saham['Signal_Line']
    
    # Tentukan periode minimum yang valid untuk analisis (perlu long_window + signal_window data)
    min_valid_index = long_window + signal_window - 2  # -2 karena index mulai dari 0
    
    # Set nilai awal menjadi NaN agar tidak ada sinyal di periode lookback
    harga_saham.loc[:min_valid_index, ['MACD', 'Signal_Line', 'MACD_Hist']] = None

    # Inisialisasi sinyal
    harga_saham['MacdSignal'] = None

    # Mask hanya untuk data yang valid setelah lookback
    mask = ~harga_saham[['MACD', 'Signal_Line']].isna().any(axis=1)
    
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


# def calculate_fibonacci_retracement(harga_saham, lookback=60, retracement_levels=None):
#     if retracement_levels is None:
#         retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
#     harga_saham = harga_saham.copy()
#     harga_saham['Fibonacci_Signal'] = 'Hold'
#     fib_highs = []
#     fib_lows = []
#     fib_levels_list = []

#     for i in range(len(harga_saham)):
#         if i < lookback - 1:
#             fib_highs.append(None)
#             fib_lows.append(None)
#             fib_levels_list.append(None)
#             continue
#         recent = harga_saham.iloc[i - lookback + 1:i + 1]
#         high = recent['High'].max()
#         low = recent['Low'].min()
#         last_close = harga_saham.iloc[i]['Close']
#         levels = [high - (high - low) * r for r in retracement_levels]
#         fib_highs.append(high)
#         fib_lows.append(low)
#         fib_levels_list.append(levels)
#         # Signal logic
#         if abs(last_close - levels[3]) / (high - low) < 0.02 or abs(last_close - levels[4]) / (high - low) < 0.02:
#             harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Buy'
#         elif abs(last_close - levels[0]) / (high - low) < 0.02 or abs(last_close - levels[1]) / (high - low) < 0.02:
#             harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Sell'
#         else:
#             harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Hold'

#     # Optionally store the last levels for plotting
#     if fib_highs[-1] is not None and fib_lows[-1] is not None and fib_levels_list[-1] is not None:
#         harga_saham.attrs['fibonacci_levels'] = {
#             'high': fib_highs[-1],
#             'low': fib_lows[-1],
#             'levels': fib_levels_list[-1]
#         }
#     harga_saham['fib_high'] = fib_highs
#     harga_saham['fib_low'] = fib_lows
#     harga_saham['fib_levels'] = fib_levels_list
#     return harga_saham



def calculate_fibonacci_retracement(harga_saham, lookback=60, retracement_levels=None):
    if retracement_levels is None:
        retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    harga_saham = harga_saham.copy()
    harga_saham['Fibonacci_Signal'] = 'Hold'
    fib_highs = []
    fib_lows = []
    fib_levels_list = []

    # Buat dictionary kosong untuk setiap level
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

        # Simpan tiap level ke kolom masing-masing
        for idx, r in enumerate(retracement_levels):
            fib_level_columns[f'fib_{r}'].append(levels[idx])

        # Signal logic
        if abs(last_close - levels[3]) / (high - low) < 0.02 or abs(last_close - levels[4]) / (high - low) < 0.02:
            harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Buy'
        elif abs(last_close - levels[0]) / (high - low) < 0.02 or abs(last_close - levels[1]) / (high - low) < 0.02:
            harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Sell'
        else:
            harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Hold'

    # Tambahkan kolom fib level individual ke DataFrame
    for r in retracement_levels:
        harga_saham[f'fib_{r}'] = fib_level_columns[f'fib_{r}']

    # Kolom tambahan
    harga_saham['fib_high'] = fib_highs
    harga_saham['fib_low'] = fib_lows
    harga_saham['fib_levels'] = fib_levels_list

    # Menyimpan level terakhir (opsional)
    if fib_highs[-1] is not None and fib_lows[-1] is not None and fib_levels_list[-1] is not None:
        harga_saham.attrs['fibonacci_levels'] = {
            'high': fib_highs[-1],
            'low': fib_lows[-1],
            'levels': fib_levels_list[-1]
        }

    return harga_saham






# def calculate_fibonacci_retracement(harga_saham, lookback=60, retracement_levels=None):
#     if retracement_levels is None:
#         retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    
#     harga_saham = harga_saham.copy()
    
#     # PASTIKAN kolom Fibonacci_Signal diinisialisasi
#     if 'Fibonacci_Signal' not in harga_saham.columns:
#         harga_saham['Fibonacci_Signal'] = 'Hold'
    
#     fib_highs = []
#     fib_lows = []
#     fib_levels_list = []

#     for i in range(len(harga_saham)):
#         if i < lookback - 1:
#             fib_highs.append(None)
#             fib_lows.append(None)
#             fib_levels_list.append(None)
#             continue
            
#         recent = harga_saham.iloc[i - lookback + 1:i + 1]
#         high = recent['High'].max()
#         low = recent['Low'].min()
#         last_close = harga_saham.iloc[i]['Close']
#         levels = [high - (high - low) * r for r in retracement_levels]
        
#         fib_highs.append(high)
#         fib_lows.append(low)
#         fib_levels_list.append(levels)
        
#         # Signal logic - pastikan tidak ada division by zero
#         if (high - low) > 0:
#             if abs(last_close - levels[3]) / (high - low) < 0.02 or abs(last_close - levels[4]) / (high - low) < 0.02:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Buy'
#             elif abs(last_close - levels[0]) / (high - low) < 0.02 or abs(last_close - levels[1]) / (high - low) < 0.02:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Sell'
#             else:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Hold'

#     # PASTIKAN kolom ditambahkan dengan benar
#     harga_saham['fib_high'] = fib_highs
#     harga_saham['fib_low'] = fib_lows
#     harga_saham['fib_levels'] = fib_levels_list
    
#     return harga_saham


# def calculate_fibonacci_retracement(harga_saham, lookback=60, retracement_levels=None):
#     if retracement_levels is None:
#         retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

#     harga_saham = harga_saham.copy()

#     # Inisialisasi kolom sinyal jika belum ada
#     if 'Fibonacci_Signal' not in harga_saham.columns:
#         harga_saham['Fibonacci_Signal'] = 'Hold'

#     fib_highs = []
#     fib_lows = []
#     fib_levels_list = []

#     # Kolom untuk masing-masing level Fibonacci
#     fib_236 = []
#     fib_382 = []
#     fib_50 = []
#     fib_618 = []
#     fib_786 = []

#     for i in range(len(harga_saham)):
#         if i < lookback - 1:
#             fib_highs.append(None)
#             fib_lows.append(None)
#             fib_levels_list.append(None)

#             fib_236.append(None)
#             fib_382.append(None)
#             fib_50.append(None)
#             fib_618.append(None)
#             fib_786.append(None)
#             continue

#         recent = harga_saham.iloc[i - lookback + 1:i + 1]
#         high = recent['High'].max()
#         low = recent['Low'].min()
#         last_close = harga_saham.iloc[i]['Close']
#         levels = [high - (high - low) * r for r in retracement_levels]

#         fib_highs.append(high)
#         fib_lows.append(low)
#         fib_levels_list.append(levels)

#         # Menyimpan level Fibonacci individual
#         fib_236.append(levels[0])
#         fib_382.append(levels[1])
#         fib_50.append(levels[2])
#         fib_618.append(levels[3])
#         fib_786.append(levels[4])

#         # Signal logic
#         if (high - low) > 0:
#             if abs(last_close - levels[3]) / (high - low) < 0.02 or abs(last_close - levels[4]) / (high - low) < 0.02:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Buy'
#             elif abs(last_close - levels[0]) / (high - low) < 0.02 or abs(last_close - levels[1]) / (high - low) < 0.02:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Sell'
#             else:
#                 harga_saham.at[harga_saham.index[i], 'Fibonacci_Signal'] = 'Hold'

#     # Ubah fib_levels_list menjadi string agar bisa ditampilkan
#     fib_levels_str = [str(levels) if levels is not None else None for levels in fib_levels_list]

#     # Tambahkan kolom ke DataFrame
#     harga_saham['fib_high'] = fib_highs
#     harga_saham['fib_low'] = fib_lows
#     harga_saham['fib_levels'] = fib_levels_str  # pastikan ini string
#     harga_saham['fib_236'] = fib_236
#     harga_saham['fib_382'] = fib_382
#     harga_saham['fib_50'] = fib_50
#     harga_saham['fib_618'] = fib_618
#     harga_saham['fib_786'] = fib_786

#     return harga_saham





# Fungsi untuk mengambil data berdasarkan ticker yang dipilih
def get_stock_data_by_ticker(ticker):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='harga_saham'
    )
    cursor = connection.cursor()

    try:
        sql = f"SELECT Data FROM data_saham_max_new WHERE Ticker = '{ticker}'"
        cursor.execute(sql)
        result = cursor.fetchone()

        if result:
            data_saham = json.loads(result[0])
            df = pd.DataFrame(data_saham)
            df['Date'] = pd.to_datetime(df['Date'])
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
            df['Ticker'] = ticker
            return df
        else:
            return pd.DataFrame()
    finally:
        cursor.close()
        connection.close()

# Callback untuk mengisi dropdown ticker
@app.callback(
    Output('ticker-dropdown', 'options'),
    Input('data-period-selector', 'value')
)
def update_ticker_dropdown_options(selected_period):
    return [{'label': ticker, 'value': ticker} for ticker in AVAILABLE_TICKERS]

# Callback untuk memuat data berdasarkan ticker yang dipilih
def update_data_based_on_ticker(selected_ticker, selected_period):
    if not selected_ticker:
        return pd.DataFrame(), {}

    # Ambil data untuk ticker yang dipilih
    data = get_stock_data_by_ticker(selected_ticker)

    if data.empty:
        return pd.DataFrame(), {}

    # Proses data sesuai kebutuhan (contoh: filter berdasarkan periode)
    if selected_period == '1y':
        one_year_ago = datetime.now() - timedelta(days=365)
        data = data[data['Datetime'] >= one_year_ago]

    return data, {}



# # Updated analyze_signals function with more detailed information

# def analyze_signals(filtered_df, selected_technicals):
#     """
#     Analyze signals from the dataframe and create a summary.
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame with signals
#     selected_technicals (list): Selected technical indicators
    
#     Returns:
#     dict: Dictionary containing signal summary
#     """
#     signal_summary = {
#         'total_signals': {'Buy': 0, 'Sell': 0, 'Hold': 0},
#         'signal_details': {}
#     }
    
#     # Add Final_Signal to selected_technicals for analysis
#     all_technicals = selected_technicals.copy() if selected_technicals else []
#     if 'Final_Signal' not in all_technicals:
#         all_technicals.append('Final_Signal')
    
#     # For each technical indicator, analyze the signals
#     for tech in all_technicals:
#         signal_col = tech if tech in filtered_df.columns else 'signal_akhir' if tech == 'Final_Signal' else None
        
#         if signal_col is None or signal_col not in filtered_df.columns:
#             signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
#             continue
        
#         # Initialize signal details for this technical
#         signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
        
#         # Track candlestick trends only if selected
#         if (tech == 'Candlestick_Signal' or 
#             (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)):
#             candlestick_trends = {'Uptrend': 0, 'Downtrend': 0, 'Sideways': 0, 'Unknown': 0}
#             signal_summary['candlestick_trends'] = candlestick_trends
        
#         # Track volume signals only if selected
#         if (tech == 'Volume_Signal' or 
#             (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)):
#             volume_signals = {'High Volume': 0, 'Low Volume': 0}
#             signal_summary['volume_signals'] = volume_signals
        
#         # Collect signal details
#         for idx, row in filtered_df.iterrows():
#             if pd.isna(row[signal_col]):
#                 continue
                
#             signal = row[signal_col]
#             if signal not in ['Buy', 'Sell', 'Hold']:
#                 continue
                
#             # For regular signals
#             signal_detail = {
#                 'datetime': row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime']),
#                 'price': float(row['Close']),
#                 'signal': signal,
#                 'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
#             }
            
#             # Add volume reason only if Volume_Signal is selected
#             if ((tech == 'Volume_Signal' or 
#                 (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)) 
#                 and 'VolumeSignal' in row):
#                 if not pd.isna(row['VolumeSignal']):
#                     signal_detail['volume_reason'] = f"{row['VolumeSignal']} - " + (row.get('volume_reason', 'Volume analysis') if not pd.isna(row.get('volume_reason', '')) else 'Volume analysis')
#                     # Count volume signals
#                     if 'volume_signals' in signal_summary:
#                         signal_summary['volume_signals'][row['VolumeSignal']] = signal_summary['volume_signals'].get(row['VolumeSignal'], 0) + 1
#                 else:
#                     signal_detail['volume_reason'] = 'Volume analysis not available'
            
#             # Add candlestick pattern details only if Candlestick_Signal is selected
#             if ((tech == 'Candlestick_Signal' or 
#                 (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)) 
#                 and 'candlestick_trend' in row):
#                 if 'detected_patterns' in row and not pd.isna(row['detected_patterns']):
#                     signal_detail['patterns'] = row['detected_patterns']
#                 else:
#                     signal_detail['patterns'] = 'No specific patterns detected'
                    
#                 if not pd.isna(row['candlestick_trend']):
#                     signal_detail['trend'] = row['candlestick_trend']
#                     # Count candlestick trends
#                     if 'candlestick_trends' in signal_summary:
#                         signal_summary['candlestick_trends'][row['candlestick_trend']] = signal_summary['candlestick_trends'].get(row['candlestick_trend'], 0) + 1
#                 else:
#                     signal_detail['trend'] = 'Unknown'
#                     if 'candlestick_trends' in signal_summary:
#                         signal_summary['candlestick_trends']['Unknown'] += 1
                    
#                 if 'candlestick_reason' in row and not pd.isna(row['candlestick_reason']):
#                     signal_detail['candlestick_reason'] = row['candlestick_reason']
#                 else:
#                     signal_detail['candlestick_reason'] = 'Based on pattern analysis'
                    
#             # Add signal reasoning for final signal
#             if tech == 'Final_Signal' and 'signal_reason' in row and not pd.isna(row['signal_reason']):
#                 signal_detail['reason'] = row['signal_reason']
#             elif tech == 'Final_Signal':
#                 signal_detail['reason'] = 'Combined analysis of all indicators'
            
#             # Add to appropriate signal list
#             signal_summary['signal_details'][tech][signal].append(signal_detail)
            
#             # Only count towards total signals for Final_Signal
#             if tech == 'Final_Signal':
#                 signal_summary['total_signals'][signal] += 1
    
#     # Add detailed volume data if Volume_Signal was selected
#     if 'Volume_Signal' in selected_technicals:
#         # Extract volume-specific data for a separate table
#         volume_data = []
#         for signal_type in ['Buy', 'Sell', 'Hold']:
#             if signal_type in signal_summary['signal_details'].get('Volume_Signal', {}):
#                 for entry in signal_summary['signal_details']['Volume_Signal'][signal_type]:
#                     volume_type = 'Unknown'
#                     if 'volume_reason' in entry and entry['volume_reason']:
#                         parts = entry['volume_reason'].split(' - ', 1)
#                         if len(parts) > 0:
#                             volume_type = parts[0]
                    
#                     volume_data.append({
#                         'datetime': entry.get('datetime', ''),
#                         'price': entry.get('price', 0),
#                         'volume': entry.get('volume', 0),
#                         'volume_type': volume_type,
#                         'signal': entry.get('signal', ''),
#                         'analysis': entry.get('volume_reason', 'Volume analysis')
#                     })
        
#         signal_summary['volume_data'] = volume_data
    
#     return signal_summary


def analyze_signals(filtered_df, selected_technicals):
    """
    Analyze signals from the dataframe and create a summary.
    Enhanced version with better volume analysis.
    """
    signal_summary = {
        'total_signals': {'Buy': 0, 'Sell': 0, 'Hold': 0},
        'signal_details': {}
    }
    
    # Add Final_Signal to selected_technicals for analysis
    all_technicals = selected_technicals.copy() if selected_technicals else []
    if 'Final_Signal' not in all_technicals:
        all_technicals.append('Final_Signal')
    
    # For each technical indicator, analyze the signals
    for tech in all_technicals:
        signal_col = tech if tech in filtered_df.columns else 'signal_akhir' if tech == 'Final_Signal' else None
        
        if signal_col is None or signal_col not in filtered_df.columns:
            signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
            continue
        
        # Initialize signal details for this technical
        signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
        
        # Track candlestick trends only if selected
        if (tech == 'Candlestick_Signal' or 
            (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)):
            candlestick_trends = {'Uptrend': 0, 'Downtrend': 0, 'Sideways': 0, 'Unknown': 0}
            signal_summary['candlestick_trends'] = candlestick_trends
        
        # Track volume signals only if selected
        if (tech == 'Volume_Signal' or 
            (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)):
            volume_signals = {'High Volume': 0, 'Low Volume': 0}
            signal_summary['volume_signals'] = volume_signals
        
        # Collect signal details
        for idx, row in filtered_df.iterrows():
            if pd.isna(row[signal_col]):
                continue
                
            signal = row[signal_col]
            if signal not in ['Buy', 'Sell', 'Hold']:
                continue
                
            # For regular signals
            signal_detail = {
                'datetime': row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime']),
                'price': float(row['Close']),
                'signal': signal,
                'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
            }
            
            # Add volume reason only if Volume_Signal is selected
            if ((tech == 'Volume_Signal' or 
                (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)) 
                and 'VolumeSignal' in row):
                if not pd.isna(row['VolumeSignal']):
                    signal_detail['volume_reason'] = f"{row['VolumeSignal']} - Volume analysis"
                    # Count volume signals
                    if 'volume_signals' in signal_summary:
                        signal_summary['volume_signals'][row['VolumeSignal']] = signal_summary['volume_signals'].get(row['VolumeSignal'], 0) + 1
                else:
                    signal_detail['volume_reason'] = 'Volume analysis not available'
            
            # Enhanced pattern detection for detected_patterns - FIXED VERSION
            if ((tech == 'Candlestick_Signal' or 
                (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)) 
                and 'candlestick_trend' in row):
                
                # Get detected patterns - FIXED ERROR HANDLING
                detected_patterns_str = row.get('detected_patterns', '')
                
                # Ensure detected_patterns_str is always a string
                if pd.isna(detected_patterns_str) or detected_patterns_str == '' or not isinstance(detected_patterns_str, str):
                    # If no patterns in detected_patterns column, check individual pattern columns
                    pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL') and 
                                     col not in ['Candlestick_Signal', 'candlestick_confidence']]
                    
                    detected_individual_patterns = []
                    for pattern_col in pattern_columns:
                        if (pattern_col in row and 
                            not pd.isna(row[pattern_col]) and 
                            int(row[pattern_col]) == 1):
                            # Clean pattern name and ensure it's a string
                            pattern_name = str(pattern_col).replace('CDL', '').replace('_', ' ').title()
                            detected_individual_patterns.append(pattern_name)
                    
                    if detected_individual_patterns:
                        detected_patterns_str = ', '.join(detected_individual_patterns)
                    else:
                        detected_patterns_str = 'No specific patterns detected'
                
                # Ensure detected_patterns_str is a string before using it
                detected_patterns_str = str(detected_patterns_str) if detected_patterns_str is not None else 'No specific patterns detected'
                
                # Set detected patterns
                signal_detail['detected_patterns'] = detected_patterns_str
                signal_detail['patterns'] = detected_patterns_str  # For compatibility
                    
                # Add trend information
                if not pd.isna(row['candlestick_trend']):
                    signal_detail['trend'] = row['candlestick_trend']
                    # Count candlestick trends
                    if 'candlestick_trends' in signal_summary:
                        signal_summary['candlestick_trends'][row['candlestick_trend']] = signal_summary['candlestick_trends'].get(row['candlestick_trend'], 0) + 1
                else:
                    signal_detail['trend'] = 'Unknown'
                    if 'candlestick_trends' in signal_summary:
                        signal_summary['candlestick_trends']['Unknown'] += 1
                    
                # Add candlestick analysis reason
                if 'candlestick_confidence' in row and not pd.isna(row['candlestick_confidence']):
                    confidence = int(row['candlestick_confidence'])
                    signal_detail['candlestick_reason'] = f"Tingkat keyakinan pola: {confidence}% - {detected_patterns_str}"
                else:
                    signal_detail['candlestick_reason'] = f"Analisis pola: {detected_patterns_str}"
                    
            # Add signal reasoning for final signal
            if tech == 'Final_Signal' and 'signal_reason' in row and not pd.isna(row['signal_reason']):
                signal_detail['reason'] = row['signal_reason']
            elif tech == 'Final_Signal':
                signal_detail['reason'] = 'Analisis gabungan dari semua indikator'
            
            # Add to appropriate signal list
            signal_summary['signal_details'][tech][signal].append(signal_detail)
            
            # Only count towards total signals for Final_Signal
            if tech == 'Final_Signal':
                signal_summary['total_signals'][signal] += 1
    


    
    # PERBAIKAN: Improved volume data processing menggunakan logic yang sudah ada
    if 'Volume_Signal' in selected_technicals:
        volume_data = []
        volume_values = []
        
        print(f"DEBUG: Processing volume data for {len(filtered_df)} rows")
        print(f"DEBUG: Available columns: {list(filtered_df.columns)}")
        
        # Create comprehensive volume analysis data menggunakan kolom yang sudah ada
        for idx, row in filtered_df.iterrows():
            # Cek kolom yang diperlukan sesuai dengan logic volume yang sudah ada
            required_cols = ['Volume', 'VMA', 'VolumeSignal', 'Volume_Signal']
            missing_cols = [col for col in required_cols if col not in filtered_df.columns]
            
            if missing_cols:
                print(f"DEBUG: Missing volume columns: {missing_cols}")
                continue
                
            # Skip rows dengan data kosong
            if pd.isna(row.get('Volume')) or pd.isna(row.get('VMA')) or pd.isna(row.get('VolumeSignal')):
                continue
                
            try:
                current_volume = int(row['Volume'])
                vma_value = int(row['VMA']) if not pd.isna(row['VMA']) else 0
                volume_signal_type = row['VolumeSignal']  # 'High Volume' atau 'Low Volume'
                volume_signal = row['Volume_Signal']  # 'Buy', 'Sell', 'Hold'
                
                volume_values.append(current_volume)
                
                # Calculate volume ratio
                volume_ratio = current_volume / vma_value if vma_value > 0 else 0
                
                # Calculate price change
                price_change = 0
                if idx > 0:
                    try:
                        prev_idx = filtered_df.index[max(0, filtered_df.index.get_loc(idx) - 1)]
                        prev_close = filtered_df.loc[prev_idx, 'Close']
                        if not pd.isna(prev_close) and not pd.isna(row['Close']):
                            price_change = ((row['Close'] - prev_close) / prev_close) * 100
                    except:
                        price_change = 0
                
                # Simple analysis berdasarkan logic yang sudah ada
                if volume_signal_type == 'High Volume':
                    if volume_ratio > 2:
                        analysis = f"Very high volume ({volume_ratio:.1f}x average)"
                        volume_trend = 'Spike'
                    elif volume_ratio > 1.5:
                        analysis = f"High volume ({volume_ratio:.1f}x average)"
                        volume_trend = 'High'
                    else:
                        analysis = f"Above average volume ({volume_ratio:.1f}x average)"
                        volume_trend = 'Above Average'
                else:  # Low Volume
                    analysis = f"Low volume ({volume_ratio:.1f}x average)"
                    volume_trend = 'Low'
                
                # Format datetime
                try:
                    formatted_datetime = row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime'])
                except:
                    formatted_datetime = str(row.get('Datetime', 'Unknown'))
                
                volume_data.append({
                    'datetime': formatted_datetime,
                    'price': float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                    'volume': current_volume,
                    'vma': vma_value,
                    'volume_ratio': volume_ratio,
                    'volume_type': volume_signal_type,
                    'signal': volume_signal,
                    'analysis': analysis,
                    'volume_trend': volume_trend,
                    'price_change': price_change
                })
                
            except Exception as e:
                print(f"DEBUG: Error processing volume data at row {idx}: {str(e)}")
                continue
        
        print(f"DEBUG: Processed {len(volume_data)} volume data points")
        
        # Calculate simple volume statistics
        if volume_values:
            avg_volume = sum(volume_values) / len(volume_values)
            peak_volume = max(volume_values)
            min_volume = min(volume_values)
            
            # Simple trend calculation
            if len(volume_values) >= 10:
                recent_avg = sum(volume_values[-5:]) / 5
                older_avg = sum(volume_values[-10:-5]) / 5
                if recent_avg > older_avg * 1.1:
                    trend = 'Increasing'
                elif recent_avg < older_avg * 0.9:
                    trend = 'Decreasing'
                else:
                    trend = 'Stable'
            else:
                trend = 'Insufficient Data'
            
            signal_summary['volume_stats'] = {
                'average_volume': avg_volume,
                'peak_volume': peak_volume,
                'min_volume': min_volume,
                'trend': trend,
                'total_signals': len(volume_data),
                'high_volume_count': len([v for v in volume_data if v['volume_type'] == 'High Volume']),
                'low_volume_count': len([v for v in volume_data if v['volume_type'] == 'Low Volume'])
            }
        
        signal_summary['volume_data'] = volume_data
        print(f"DEBUG: Final volume_data length: {len(signal_summary.get('volume_data', []))}")
    
    return signal_summary



# def analyze_signals(filtered_df, selected_technicals):
#     """
#     Analyze signals from the dataframe and create a summary.
#     """
#     signal_summary = {
#         'total_signals': {'Buy': 0, 'Sell': 0, 'Hold': 0},
#         'signal_details': {}
#     }
    
#     # Add Final_Signal to selected_technicals for analysis
#     all_technicals = selected_technicals.copy() if selected_technicals else []
#     if 'Final_Signal' not in all_technicals:
#         all_technicals.append('Final_Signal')
    
#     # For each technical indicator, analyze the signals
#     for tech in all_technicals:
#         signal_col = tech if tech in filtered_df.columns else 'signal_akhir' if tech == 'Final_Signal' else None
        
#         if signal_col is None or signal_col not in filtered_df.columns:
#             signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
#             continue
        
#         # Initialize signal details for this technical
#         signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
        
#         # Track volume signals - PERBAIKAN DI SINI
#         if tech == 'Volume_Signal' and 'VolumeSignal' in filtered_df.columns:
#             volume_signals = {'High Volume': 0, 'Low Volume': 0}
#             signal_summary['volume_signals'] = volume_signals
        
#         # Collect signal details
#         for idx, row in filtered_df.iterrows():
#             if pd.isna(row[signal_col]):
#                 continue
                
#             signal = row[signal_col]
#             if signal not in ['Buy', 'Sell', 'Hold']:
#                 continue
                
#             # For regular signals
#             signal_detail = {
#                 'datetime': row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime']),
#                 'price': float(row['Close']),
#                 'signal': signal,
#                 'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
#             }
            
#             # Add volume reason - PERBAIKAN DI SINI
#             if tech == 'Volume_Signal' and 'VolumeSignal' in filtered_df.columns:
#                 volume_signal_val = row.get('VolumeSignal', None)
#                 if not pd.isna(volume_signal_val):
#                     signal_detail['volume_reason'] = f"{volume_signal_val} - Volume analysis"
#                     # Count volume signals
#                     if 'volume_signals' in signal_summary:
#                         signal_summary['volume_signals'][volume_signal_val] = signal_summary['volume_signals'].get(volume_signal_val, 0) + 1
#                 else:
#                     signal_detail['volume_reason'] = 'Volume analysis not available'
            
#             # Add to appropriate signal list
#             signal_summary['signal_details'][tech][signal].append(signal_detail)
            
#             # Only count towards total signals for Final_Signal
#             if tech == 'Final_Signal':
#                 signal_summary['total_signals'][signal] += 1
    
#     return signal_summary


# def analyze_signals(filtered_df, selected_technicals):
#     """
#     Analyze signals from the dataframe and create a summary.
#     Enhanced version with better volume analysis and candlestick pattern detection.
#     """
#     signal_summary = {
#         'total_signals': {'Buy': 0, 'Sell': 0, 'Hold': 0},
#         'signal_details': {}
#     }
    
#     # Add Final_Signal to selected_technicals for analysis
#     all_technicals = selected_technicals.copy() if selected_technicals else []
#     if 'Final_Signal' not in all_technicals:
#         all_technicals.append('Final_Signal')
    
#     # For each technical indicator, analyze the signals
#     for tech in all_technicals:
#         signal_col = tech if tech in filtered_df.columns else 'signal_akhir' if tech == 'Final_Signal' else None
        
#         if signal_col is None or signal_col not in filtered_df.columns:
#             signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
#             continue
        
#         # Initialize signal details for this technical
#         signal_summary['signal_details'][tech] = {'Buy': [], 'Sell': [], 'Hold': []}
        
#         # Track candlestick trends only if selected
#         if (tech == 'Candlestick_Signal' or 
#             (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)):
#             candlestick_trends = {'Uptrend': 0, 'Downtrend': 0, 'Sideways': 0, 'Unknown': 0}
#             signal_summary['candlestick_trends'] = candlestick_trends
        
#         # Track volume signals only if selected
#         if (tech == 'Volume_Signal' or 
#             (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)):
#             volume_signals = {'High Volume': 0, 'Low Volume': 0}
#             signal_summary['volume_signals'] = volume_signals
        
#         # Collect signal details
#         for idx, row in filtered_df.iterrows():
#             if pd.isna(row[signal_col]):
#                 continue
                
#             signal = row[signal_col]
#             if signal not in ['Buy', 'Sell', 'Hold']:
#                 continue
                
#             # For regular signals
#             signal_detail = {
#                 'datetime': row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime']),
#                 'price': float(row['Close']),
#                 'signal': signal,
#                 'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
#             }
            
#             # Add volume reason only if Volume_Signal is selected
#             if ((tech == 'Volume_Signal' or 
#                 (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)) 
#                 and 'VolumeSignal' in row):
#                 if not pd.isna(row['VolumeSignal']):
#                     signal_detail['volume_reason'] = f"{row['VolumeSignal']} - Volume analysis"
#                     # Count volume signals
#                     if 'volume_signals' in signal_summary:
#                         signal_summary['volume_signals'][row['VolumeSignal']] = signal_summary['volume_signals'].get(row['VolumeSignal'], 0) + 1
#                 else:
#                     signal_detail['volume_reason'] = 'Volume analysis not available'
            
#             # PERBAIKAN: Enhanced pattern detection untuk detected_patterns
#             if ((tech == 'Candlestick_Signal' or 
#                 (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)) 
#                 and 'candlestick_trend' in row):
                
#                 # Get actual pattern names dari kolom CDL
#                 detected_patterns = []
#                 pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL') and 
#                                  col not in ['Candlestick_Signal', 'candlestick_confidence']]
                
#                 # Cek pattern yang aktif pada row ini
#                 for pattern_col in pattern_columns:
#                     if (pattern_col in row and 
#                         not pd.isna(row[pattern_col]) and 
#                         int(row[pattern_col]) == 1):  # Pattern terdeteksi
#                         # Convert CDL pattern names to readable format
#                         pattern_name = pattern_col.replace('CDL', '').replace('_', ' ').title()
#                         detected_patterns.append(pattern_name)
                
#                 # Set detected patterns
#                 if detected_patterns:
#                     signal_detail['detected_patterns'] = ', '.join(detected_patterns)
#                     signal_detail['patterns'] = ', '.join(detected_patterns)  # Untuk kompatibilitas
#                 else:
#                     signal_detail['detected_patterns'] = 'No specific patterns detected'
#                     signal_detail['patterns'] = 'No specific patterns detected'
                    
#                 # Add trend information
#                 if not pd.isna(row['candlestick_trend']):
#                     signal_detail['trend'] = row['candlestick_trend']
#                     # Count candlestick trends
#                     if 'candlestick_trends' in signal_summary:
#                         signal_summary['candlestick_trends'][row['candlestick_trend']] = signal_summary['candlestick_trends'].get(row['candlestick_trend'], 0) + 1
#                 else:
#                     signal_detail['trend'] = 'Unknown'
#                     if 'candlestick_trends' in signal_summary:
#                         signal_summary['candlestick_trends']['Unknown'] += 1
                    
#                 if 'candlestick_reason' in row and not pd.isna(row['candlestick_reason']):
#                     signal_detail['candlestick_reason'] = row['candlestick_reason']
#                 else:
#                     signal_detail['candlestick_reason'] = 'Based on pattern analysis'
                    
#             # Add signal reasoning for final signal
#             if tech == 'Final_Signal' and 'signal_reason' in row and not pd.isna(row['signal_reason']):
#                 signal_detail['reason'] = row['signal_reason']
#             elif tech == 'Final_Signal':
#                 signal_detail['reason'] = 'Combined analysis of all indicators'
            
#             # Add to appropriate signal list
#             signal_summary['signal_details'][tech][signal].append(signal_detail)
            
#             # Only count towards total signals for Final_Signal
#             if tech == 'Final_Signal':
#                 signal_summary['total_signals'][signal] += 1
    
#     # Enhanced volume data processing if Volume_Signal was selected
#     if 'Volume_Signal' in selected_technicals:
#         volume_data = []
#         volume_values = []
        
#         # Create comprehensive volume analysis data
#         for idx, row in filtered_df.iterrows():
#             if 'VolumeSignal' not in row or pd.isna(row.get('VolumeSignal')):
#                 continue
                
#             current_volume = int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
#             volume_values.append(current_volume)
            
#             # Determine volume type and analysis
#             volume_type = row.get('VolumeSignal', 'Unknown')
#             vma_value = row.get('VMA', 0)
            
#             # Calculate volume ratio vs average
#             volume_ratio = current_volume / vma_value if vma_value > 0 else 0
            
#             # Enhanced volume analysis
#             if volume_type == 'High Volume':
#                 if volume_ratio > 2:
#                     analysis = f"Very high volume ({volume_ratio:.1f}x average) - Strong market interest"
#                     volume_trend = 'Spike'
#                 elif volume_ratio > 1.5:
#                     analysis = f"High volume ({volume_ratio:.1f}x average) - Increased activity"
#                     volume_trend = 'High'
#                 else:
#                     analysis = f"Above average volume ({volume_ratio:.1f}x average) - Moderate interest"
#                     volume_trend = 'Above Average'
#             else:  # Low Volume
#                 if volume_ratio < 0.5:
#                     analysis = f"Very low volume ({volume_ratio:.1f}x average) - Weak participation"
#                     volume_trend = 'Very Low'
#                 elif volume_ratio < 0.8:
#                     analysis = f"Low volume ({volume_ratio:.1f}x average) - Below normal activity"
#                     volume_trend = 'Low'
#                 else:
#                     analysis = f"Near average volume ({volume_ratio:.1f}x average) - Normal activity"
#                     volume_trend = 'Normal'
            
#             # Determine signal based on volume and price action
#             price_change = 0
#             if idx > 0:
#                 prev_close = filtered_df.iloc[idx-1]['Close']
#                 price_change = ((row['Close'] - prev_close) / prev_close) * 100
            
#             # Enhanced signal logic based on volume and price
#             if volume_type == 'High Volume':
#                 if price_change > 1:
#                     signal = 'Buy'  # High volume with price increase
#                 elif price_change < -1:
#                     signal = 'Sell'  # High volume with price decrease
#                 else:
#                     signal = 'Hold'  # High volume but no significant price change
#             else:  # Low Volume
#                 if abs(price_change) < 0.5:
#                     signal = 'Hold'  # Low volume, low volatility
#                 elif price_change > 0:
#                     signal = 'Hold'  # Low volume rally may not sustain
#                 else:
#                     signal = 'Hold'  # Low volume decline may not continue
            
#             volume_data.append({
#                 'datetime': row['Datetime'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Datetime'], pd.Timestamp) else str(row['Datetime']),
#                 'price': float(row['Close']),
#                 'volume': current_volume,
#                 'volume_ratio': volume_ratio,
#                 'volume_type': volume_type,
#                 'signal': signal,
#                 'analysis': analysis,
#                 'volume_trend': volume_trend,
#                 'price_change': price_change,
#                 'vma': int(vma_value) if vma_value > 0 else 0
#             })
        
#         # Calculate volume statistics
#         if volume_values:
#             avg_volume = sum(volume_values) / len(volume_values)
#             peak_volume = max(volume_values)
#             min_volume = min(volume_values)
            
#             # Determine overall volume trend
#             recent_volumes = volume_values[-10:] if len(volume_values) >= 10 else volume_values
#             older_volumes = volume_values[-20:-10] if len(volume_values) >= 20 else volume_values[:-10] if len(volume_values) > 10 else []
            
#             if older_volumes:
#                 recent_avg = sum(recent_volumes) / len(recent_volumes)
#                 older_avg = sum(older_volumes) / len(older_volumes)
#                 if recent_avg > older_avg * 1.1:
#                     trend = 'Increasing'
#                 elif recent_avg < older_avg * 0.9:
#                     trend = 'Decreasing'
#                 else:
#                     trend = 'Stable'
#             else:
#                 trend = 'Insufficient Data'
            
#             # Add volume statistics to signal summary
#             signal_summary['volume_stats'] = {
#                 'average_volume': avg_volume,
#                 'peak_volume': peak_volume,
#                 'min_volume': min_volume,
#                 'trend': trend,
#                 'total_signals': len(volume_data),
#                 'high_volume_count': len([v for v in volume_data if v['volume_type'] == 'High Volume']),
#                 'low_volume_count': len([v for v in volume_data if v['volume_type'] == 'Low Volume'])
#             }
        
#         signal_summary['volume_data'] = volume_data
    
#     return signal_summary








# def analyze_latest_trend(ticker_data, lookback_periods={
#     'bollinger': 20,
#     'ma': 50,  # For long MA comparison
#     'rsi': 14,
#     'macd': 26, # Use longest MACD period
#     'adx': 14,
#     'volume': 20
# }):
#     """
#     Analyze trend based on latest data points for each indicator
#     """
#     if ticker_data.empty:
#         return None
        
#     latest_data = ticker_data.copy()
#     trends = {}
    
#     # Bollinger Bands Trend Analysis
#     if 'Upper Band' in latest_data.columns and 'Lower Band' in latest_data.columns:
#         bb_data = latest_data.tail(lookback_periods['bollinger'])
#         price_vs_middle = bb_data['Close'].iloc[-1] > bb_data['Middle Band'].iloc[-1]
#         bandwidth = (bb_data['Upper Band'].iloc[-1] - bb_data['Lower Band'].iloc[-1]) / bb_data['Middle Band'].iloc[-1]
        
#         if bb_data['Close'].iloc[-1] > bb_data['Upper Band'].iloc[-1]:
#             trends['bollinger'] = 'Strong Uptrend'
#         elif bb_data['Close'].iloc[-1] < bb_data['Lower Band'].iloc[-1]:
#             trends['bollinger'] = 'Strong Downtrend'
#         elif price_vs_middle and bandwidth > 0.2:
#             trends['bollinger'] = 'Uptrend'
#         elif not price_vs_middle and bandwidth > 0.2:
#             trends['bollinger'] = 'Downtrend'
#         else:
#             trends['bollinger'] = 'Sideways'

#     # Moving Average Trend Analysis
#     if 'short_MA' in latest_data.columns and 'long_MA' in latest_data.columns:
#         ma_data = latest_data.tail(lookback_periods['ma'])
#         short_ma_trend = ma_data['short_MA'].diff().mean() > 0
#         long_ma_trend = ma_data['long_MA'].diff().mean() > 0
#         current_cross = ma_data['short_MA'].iloc[-1] > ma_data['long_MA'].iloc[-1]
        
#         if current_cross and short_ma_trend and long_ma_trend:
#             trends['ma'] = 'Strong Uptrend'
#         elif not current_cross and not short_ma_trend and not long_ma_trend:
#             trends['ma'] = 'Strong Downtrend'
#         elif current_cross:
#             trends['ma'] = 'Uptrend'
#         elif not current_cross:
#             trends['ma'] = 'Downtrend'
#         else:
#             trends['ma'] = 'Sideways'

#     # RSI Trend Analysis
#     if 'RSI' in latest_data.columns:
#         rsi_data = latest_data.tail(lookback_periods['rsi'])
#         latest_rsi = rsi_data['RSI'].iloc[-1]
#         rsi_trend = rsi_data['RSI'].diff().mean() > 0
        
#         if latest_rsi > 70:
#             trends['rsi'] = 'Overbought'
#         elif latest_rsi < 30:
#             trends['rsi'] = 'Oversold'
#         elif latest_rsi > 50 and rsi_trend:
#             trends['rsi'] = 'Uptrend'
#         elif latest_rsi < 50 and not rsi_trend:
#             trends['rsi'] = 'Downtrend'
#         else:
#             trends['rsi'] = 'Neutral'

#     # MACD Trend Analysis
#     if all(col in latest_data.columns for col in ['MACD', 'Signal_Line', 'MACD_Hist']):
#         macd_data = latest_data.tail(lookback_periods['macd'])
#         hist_trend = macd_data['MACD_Hist'].diff().mean() > 0
#         macd_above_signal = macd_data['MACD'].iloc[-1] > macd_data['Signal_Line'].iloc[-1]
#         hist_positive = macd_data['MACD_Hist'].iloc[-1] > 0
        
#         if macd_above_signal and hist_positive and hist_trend:
#             trends['macd'] = 'Strong Uptrend'
#         elif not macd_above_signal and not hist_positive and not hist_trend:
#             trends['macd'] = 'Strong Downtrend'
#         elif macd_above_signal:
#             trends['macd'] = 'Uptrend'
#         elif not macd_above_signal:
#             trends['macd'] = 'Downtrend'
#         else:
#             trends['macd'] = 'Neutral'

#     # ADX Trend Analysis
#     if all(col in latest_data.columns for col in ['ADX', '+DI', '-DI']):
#         adx_data = latest_data.tail(lookback_periods['adx'])
#         latest_adx = adx_data['ADX'].iloc[-1]
#         di_plus_greater = adx_data['+DI'].iloc[-1] > adx_data['-DI'].iloc[-1]
        
#         if latest_adx > 25:
#             if di_plus_greater:
#                 trends['adx'] = 'Strong Uptrend'
#             else:
#                 trends['adx'] = 'Strong Downtrend'
#         elif latest_adx > 20:
#             if di_plus_greater:
#                 trends['adx'] = 'Uptrend'
#             else:
#                 trends['adx'] = 'Downtrend'
#         else:
#             trends['adx'] = 'No Trend'

#     # Volume Trend Analysis
#     if 'Volume' in latest_data.columns and 'VMA' in latest_data.columns:
#         volume_data = latest_data.tail(lookback_periods['volume'])
#         vol_trend = volume_data['Volume'].diff().mean() > 0
#         above_avg = volume_data['Volume'].iloc[-1] > volume_data['VMA'].iloc[-1]
        
#         if above_avg and vol_trend:
#             trends['volume'] = 'Strong Volume'
#         elif above_avg:
#             trends['volume'] = 'Above Average'
#         elif not above_avg and not vol_trend:
#             trends['volume'] = 'Weak Volume'
#         else:
#             trends['volume'] = 'Average Volume'

#     # Calculate overall trend
#     trend_scores = {
#         'Strong Uptrend': 2,
#         'Uptrend': 1,
#         'Sideways': 0,
#         'Neutral': 0,
#         'Downtrend': -1,
#         'Strong Downtrend': -2,
#         'Overbought': 1,
#         'Oversold': -1,
#         'Strong Volume': 1,
#         'Weak Volume': -1,
#         'Above Average': 0.5,
#         'Average Volume': 0,
#         'No Trend': 0
#     }
    
#     total_score = sum(trend_scores.get(trend, 0) for trend in trends.values())
#     num_indicators = len(trends)
    
#     if num_indicators > 0:
#         avg_score = total_score / num_indicators
#         if avg_score > 1:
#             overall_trend = 'Strong Uptrend'
#         elif avg_score > 0.3:
#             overall_trend = 'Uptrend'
#         elif avg_score < -1:
#             overall_trend = 'Strong Downtrend'
#         elif avg_score < -0.3:
#             overall_trend = 'Downtrend'
#         else:
#             overall_trend = 'Sideways'
#     else:
#         overall_trend = 'Unknown'

#     trends['overall'] = overall_trend
    
#     return trends




def update_trend_notification(bell_clicks, close_clicks, n_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, "", None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Set fixed width and proper positioning
    base_class = 'fixed top-16 right-4 w-[90vw] md:w-[600px] lg:w-[900px] xl:w-[1200px] bg-white shadow-lg rounded-lg z-50 transform transition-all duration-300'
    
    try:
        content = html.Div([
            # Header with improved styling
            html.Div([
                html.H3("Market Trend Analysis", className="text-2xl font-bold"),
                html.Div([
                    html.Button(
                        "↺", 
                        id='refresh-trends',
                        className="text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full p-2 mr-2",
                        title="Refresh Trends"
                    ),
                    html.Button(
                        "×", 
                        id='close-notification',
                        className="text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full p-2",
                        title="Close Panel"
                    )
                ], className="flex items-center")
            ], className="flex justify-between items-center p-4 border-b sticky top-0 bg-white z-20 shadow-sm"),

            # Search and Filter Section with improved layout
            html.Div([
                # Search box with more intuitive styling
                html.Div([
                    html.I(className="fas fa-search absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"),
                    dcc.Input(
                        id='ticker-search',
                        type='text',
                        placeholder='Search ticker...',
                        className="w-full pl-10 p-3 border rounded-lg"
                    )
                ], className="relative mb-4"),
                
                # Filter buttons with clear visual hierarchy
                html.Div([
                    html.Button("All", id='filter-all', 
                              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg text-sm font-medium mr-2"),
                    html.Button("Uptrend", id='filter-uptrend', 
                              className="px-4 py-2 bg-green-100 text-green-800 hover:bg-green-200 rounded-lg text-sm font-medium mr-2"),
                    html.Button("Sideways", id='filter-sideways', 
                              className="px-4 py-2 bg-gray-100 text-gray-800 hover:bg-gray-200 rounded-lg text-sm font-medium mr-2"),
                    html.Button("Downtrend", id='filter-downtrend', 
                              className="px-4 py-2 bg-red-100 text-red-800 hover:bg-red-200 rounded-lg text-sm font-medium mr-2")
                ], className="flex flex-wrap gap-2 mb-2")
            ], className="p-4 border-b sticky top-16 bg-white z-10 shadow-sm"),

            # Main content with improved layout
            html.Div(
                id="trend-content",
                className="flex-grow px-4 py-2 space-y-4",
                style={"maxHeight": "70vh", "overflowY": "auto"}
            ),

            # Market Overview Section
            html.Div([
                html.Div([
                    html.H4("Market Overview", className="text-lg font-semibold"),
                    html.Div([
                        html.Div([
                            html.Span("Bullish: ", className="font-semibold"),
                            html.Span(id="bullish-count", className="text-green-600 font-bold")
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Bearish: ", className="font-semibold"),
                            html.Span(id="bearish-count", className="text-red-600 font-bold")
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Neutral: ", className="font-semibold"),
                            html.Span(id="neutral-count", className="text-gray-600 font-bold")
                        ])
                    ], className="flex")
                ], className="flex justify-between items-center")
            ], className="border-t bg-white p-4 sticky bottom-0 z-10 shadow-inner")
        ], className="h-[90vh] flex flex-col")

        if trigger_id == 'notification-bell':
            return f"{base_class} opacity-100 scale-100 pointer-events-auto", "", content
        elif trigger_id == 'close-notification':
            return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update
        else:
            return dash.no_update, "", content

    except Exception as e:
        print(f"Error in trend notification: {str(e)}")
        return base_class, "", html.Div("Error loading trends")



def create_ticker_card(ticker, trend_info):
    # Horizontal layout for indicator details
    indicator_icons = {
        'bollinger': 'fas fa-chart-line',
        'ma': 'fas fa-chart-area',
        'rsi': 'fas fa-tachometer-alt',
        'macd': 'fas fa-exchange-alt',
        'adx': 'fas fa-compass',
        'volume': 'fas fa-chart-bar'
    }
    # Remove 'overall' from indicator list
    indicator_details = [
        html.Div([
            html.I(className=f"{indicator_icons.get(indicator.lower(), 'fas fa-chart-line')} mr-1 text-gray-400"),
            html.Span(f"{indicator.title()}: ", className="text-xs text-gray-500"),
            html.Span(str(value), className=f"text-xs font-medium {get_trend_color(str(value))}")
        ], className="flex items-center mr-4 mb-1")
        for indicator, value in trend_info.items() if indicator != 'overall'
    ]
    return html.Div([
        html.Div([
            html.Span(ticker, className="font-bold text-base mr-2"),
            html.Span(
                trend_info['overall'],
                className=f"px-3 py-1 rounded-full text-xs font-medium {get_trend_color(trend_info['overall'])} bg-opacity-20"
            )
        ], className="flex items-center mb-2"),
        html.Div(indicator_details, className="flex flex-wrap gap-y-1 gap-x-3")
    ], className="p-4 bg-white border rounded-lg hover:border-blue-300 hover:shadow-md transition-all h-full")


@app.callback(
    [Output("bullish-count", "children", allow_duplicate = True),
     Output("bearish-count", "children", allow_duplicate = True),
     Output("neutral-count", "children", allow_duplicate = True)],
    [Input("trend-update-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_market_overview(n_intervals):
    try:
        all_trends = get_all_trends_from_db()
        bullish = sum(1 for t in all_trends.values() if t['overall'] in ['Strong Uptrend', 'Uptrend'])
        bearish = sum(1 for t in all_trends.values() if t['overall'] in ['Strong Downtrend', 'Downtrend'])
        neutral = sum(1 for t in all_trends.values() if t['overall'] == 'Sideways')
        
        return str(bullish), str(bearish), str(neutral)
    except Exception as e:
        return "0", "0", "0"




def get_trend_color(trend):
    if 'Strong Uptrend' in trend or 'Uptrend' in trend or 'Overbought' in trend:
        return 'text-green-600'
    elif 'Strong Downtrend' in trend or 'Downtrend' in trend or 'Oversold' in trend:
        return 'text-red-600'
    return 'text-gray-600'





@app.callback(
    Output("trend-content", "children"),
    [Input("ticker-search", "value"),
     Input("filter-all", "n_clicks"),
     Input("filter-uptrend", "n_clicks"),
     Input("filter-sideways", "n_clicks"),
     Input("filter-downtrend", "n_clicks")],
    [State("trend-content", "children")]
)
def filter_trend_content(search_value, all_clicks, up_clicks, side_clicks, down_clicks, current_content):
    import dash
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_content
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Show loading while fetching data
    loading_spinner = dcc.Loading(
        id="trend-content-loading",
        type="circle",
        children=html.Div("Loading...", className="text-center p-4 text-gray-500")
    )
    
    # Always show loading spinner first
    if trigger_id == "ticker-search" and search_value:
        # Show spinner while searching
        return loading_spinner
    
    # Get all trends data from DB
    all_trends = get_all_trends_from_db()
    trend_groups = {
        'Strong Uptrend': [],
        'Uptrend': [],
        'Sideways': [],
        'Downtrend': [],
        'Strong Downtrend': []
    }
    for ticker, trends in all_trends.items():
        if trends['overall'] in trend_groups:
            trend_groups[trends['overall']].append((ticker, trends))
    
    content = []
    # Handle search filter
    if trigger_id == "ticker-search" and search_value:
        search_value = search_value.upper()
        matching = [(ticker, trends) for ticker, trends in all_trends.items() if search_value in ticker]
        if matching:
            # Find the trend group for each matching ticker
            group_map = {}
            for ticker, trends in matching:
                group = trends['overall']
                if group not in group_map:
                    group_map[group] = []
                group_map[group].append((ticker, trends))
            for group, tickers in group_map.items():
                content.append(create_trend_section(group, tickers))
        else:
            content = [html.Div(f"No tickers found matching '{search_value}'", className="p-4 text-center text-gray-500")]
        return html.Div(content)
    
    # Handle trend filters
    elif trigger_id in ["filter-all", "filter-uptrend", "filter-sideways", "filter-downtrend"]:
        if trigger_id == "filter-all":
            # Show all trends
            for trend_type, tickers in trend_groups.items():
                if tickers:  # Only show sections with tickers
                    content.append(create_trend_section(trend_type, tickers))
                    
        elif trigger_id == "filter-uptrend":
            # Show uptrend and strong uptrend
            for trend_type in ['Strong Uptrend', 'Uptrend']:
                if trend_groups[trend_type]:
                    content.append(create_trend_section(trend_type, trend_groups[trend_type]))
                    
        elif trigger_id == "filter-sideways":
            # Show sideways trend
            if trend_groups['Sideways']:
                content.append(create_trend_section('Sideways', trend_groups['Sideways']))
                
        elif trigger_id == "filter-downtrend":
            # Show downtrend and strong downtrend
            for trend_type in ['Strong Downtrend', 'Downtrend']:
                if trend_groups[trend_type]:
                    content.append(create_trend_section(trend_type, trend_groups[trend_type]))
    
    # If no specific filter, show all trends
    else:
        for trend_type, tickers in trend_groups.items():
            if tickers:  # Only show sections with tickers
                content.append(create_trend_section(trend_type, tickers))

    # Add market overview
    market_overview = html.Div([
        html.H3("Market Overview", className="text-lg font-bold mb-2"),
        html.Div([
            html.Div([
                html.Span("Bullish: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Strong Uptrend']) + len(trend_groups['Uptrend'])),
                    className="text-green-600"
                )
            ], className="mr-4"),
            html.Div([
                html.Span("Bearish: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Strong Downtrend']) + len(trend_groups['Downtrend'])),
                    className="text-red-600"
                )
            ], className="mr-4"),
            html.Div([
                html.Span("Neutral: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Sideways'])),
                    className="text-gray-600"
                )
            ])
        ], className="flex")
    ], className="mt-4 p-4 bg-white rounded-lg shadow-sm")
    
    content.append(market_overview)
    
    return html.Div(content)



def create_trend_section(trend_type, tickers):
    # Responsive grid: 2 columns on small, 3 on md, 4 on lg+
    return html.Div([
        html.Div(
            html.H3(trend_type, className="font-bold text-lg mb-2"),
            className="sticky top-0 bg-white z-10"
        ),
        html.Div([
            create_ticker_card(ticker, trends)
            for ticker, trends in tickers
        ], className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 max-h-[350px] overflow-y-auto pr-2")
    ], className="bg-white p-4 rounded-lg mb-4")


@app.callback(
    [Output("trend-notification", "className" , allow_duplicate = True),
     Output("trend-content", "children" , allow_duplicate = True),
     Output("notification-count", "children" , allow_duplicate = True)],
    [Input("notification-bell", "n_clicks"),
     Input("close-notification", "n_clicks"),
     Input("refresh-trends", "n_clicks"),
     Input("filter-all", "n_clicks"),
     Input("filter-uptrend", "n_clicks"),
     Input("filter-sideways", "n_clicks"),
     Input("filter-downtrend", "n_clicks"),
     Input("trend-update-interval", "n_intervals")],
    [State("trend-notification", "className"),
     State("ticker-search", "value")],
     prevent_initial_call=True
)
def update_notification_panel(bell_clicks, close_clicks, refresh_clicks, 
                             all_clicks, uptrend_clicks, sideways_clicks, downtrend_clicks, 
                             interval, current_class, search_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_class, "", ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    base_class = 'fixed top-16 right-4 w-[90vw] md:w-[600px] lg:w-[900px] xl:w-[1200px] bg-white shadow-lg rounded-lg z-50 transform transition-all duration-300'
    
    # Handle opening and closing of notification panel
    if trigger_id == 'notification-bell':
        if 'opacity-0' in current_class:
            new_class = f"{base_class} opacity-100 scale-100 pointer-events-auto"
        else:
            new_class = f"{base_class} opacity-0 scale-95 pointer-events-none"
        
        if 'opacity-0' in new_class:
            return new_class, dash.no_update, dash.no_update
    
    elif trigger_id == 'close-notification':
        return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update
    
    # Get trend data
    all_trends = get_all_trends_from_db()
    trend_groups = {
        'Strong Uptrend': [],
        'Uptrend': [],
        'Sideways': [],
        'Downtrend': [],
        'Strong Downtrend': []
    }
    
    # Group tickers by trend
    for ticker, trends in all_trends.items():
        if trends['overall'] in trend_groups:
            trend_groups[trends['overall']].append((ticker, trends))
    
    # Handle search filtering
    if search_value:
        search_value = search_value.upper()
        for trend_type in trend_groups:
            trend_groups[trend_type] = [(ticker, trends) for ticker, trends in trend_groups[trend_type] 
                                       if search_value in ticker]
    
    # Handle filter buttons
    content = []
    if trigger_id == 'filter-uptrend' or (trigger_id != 'filter-all' and 
                                         trigger_id != 'filter-sideways' and 
                                         trigger_id != 'filter-downtrend'):
        # Show all trends by default or when uptrend selected
        for trend_type in ['Strong Uptrend', 'Uptrend', 'Sideways', 'Downtrend', 'Strong Downtrend']:
            if trend_groups[trend_type]:
                content.append(create_trend_section(trend_type, trend_groups[trend_type]))
    elif trigger_id == 'filter-uptrend':
        # Show only uptrends
        for trend_type in ['Strong Uptrend', 'Uptrend']:
            if trend_groups[trend_type]:
                content.append(create_trend_section(trend_type, trend_groups[trend_type]))
    elif trigger_id == 'filter-sideways':
        # Show only sideways
        if trend_groups['Sideways']:
            content.append(create_trend_section('Sideways', trend_groups['Sideways']))
    elif trigger_id == 'filter-downtrend':
        # Show only downtrends
        for trend_type in ['Downtrend', 'Strong Downtrend']:
            if trend_groups[trend_type]:
                content.append(create_trend_section(trend_type, trend_groups[trend_type]))
    
    # If no content after filtering, show message
    if not content:
        if search_value:
            content = [html.Div(f"No tickers found matching '{search_value}'", 
                               className="p-4 text-center text-gray-500")]
        else:
            content = [html.Div("No trends available for the selected filter", 
                               className="p-4 text-center text-gray-500")]
    
    # Update counts for market overview
    bullish_count = len(trend_groups['Strong Uptrend']) + len(trend_groups['Uptrend'])
    bearish_count = len(trend_groups['Downtrend']) + len(trend_groups['Strong Downtrend'])
    neutral_count = len(trend_groups['Sideways'])
    
    # Update notification badge count (show only significant changes)
    notification_count = len(trend_groups['Strong Uptrend']) + len(trend_groups['Strong Downtrend'])
    badge_text = str(notification_count) if notification_count > 0 else ""
    
    # Keep panel open/closed state the same unless explicitly toggled
    if trigger_id not in ['notification-bell', 'close-notification']:
        return current_class, content, badge_text
        
    return current_class, content, badge_text


@app.callback(
    [Output("bullish-count", "children"),
     Output("bearish-count", "children"),
     Output("neutral-count", "children")],
    [Input("trend-content", "children")]
)
def update_market_overview_counters(content):
    try:
        all_trends = get_all_trends_from_db()
        bullish = sum(1 for t in all_trends.values() if t['overall'] in ['Strong Uptrend', 'Uptrend'])
        bearish = sum(1 for t in all_trends.values() if t['overall'] in ['Strong Downtrend', 'Downtrend'])
        neutral = sum(1 for t in all_trends.values() if t['overall'] == 'Sideways')
        
        return f"{bullish}", f"{bearish}", f"{neutral}"
    except Exception as e:
        return "0", "0", "0"










@app.callback(
    Output("trend-sections", "children"),
    [Input(f'filter-{trend.lower().replace(" ", "-")}', 'n_clicks')
     for trend in ['Strong Uptrend', 'Uptrend', 'Sideways', 'Downtrend', 'Strong Downtrend']]
)
def update_trend_display(*clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_trend = button_id.replace('filter-', '').replace('-', ' ').title()
    
    filtered_content = [section for section in trend_sections 
                       if section['props']['id'].startswith(selected_trend)]
    
    return filtered_content



app.layout = html.Div([

    # Main Container
    html.Div([
        # Toggle button for collapsed sidebar
        html.Button(
            "≫",
            id='show-sidebar-button',
            className="fixed top-4 left-4 bg-white rounded-full p-2 shadow-md z-50",
            style={'display': 'none'}
        ),


        # Update bagian notifikasi di app.layout
        html.Div([
            # Notification bell dan panel
            html.Button([
                html.I(className="fas fa-bell"),
                html.Span(id='notification-count', 
                        className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center")
            ], id='notification-bell', 
            className="fixed top-4 right-4 p-2 text-xl bg-white rounded-full shadow-lg hover:bg-gray-100 z-50"),
            
            # Notification Panel
            html.Div([
                html.Div([
                    # Header with title and close button
                    html.Div([
                        html.H3("Market Trend Analysis", className="text-lg font-bold"),
                        html.Div([
                            html.Button("↺", id='refresh-trends', 
                                    className="text-gray-500 hover:text-gray-700 p-2 mr-2"),
                            html.Button("×", id='close-notification',
                                    className="text-gray-500 hover:text-gray-700 p-2")
                        ], className="flex items-center")
                    ], className="flex justify-between items-center p-4 border-b"),
                    
                    # Search and Filter Section
                    html.Div([
                        dcc.Input(
                            id='ticker-search',
                            type='text',
                            placeholder='Search ticker...',
                            className="w-full p-2 border rounded mb-2"
                        ),
                        # Filter buttons
                        html.Div([
                            html.Button("All", 
                                    id='filter-all',
                                    className="px-3 py-1 bg-gray-200 rounded-full text-sm font-medium mr-2"),
                            html.Button("Uptrend", 
                                    id='filter-uptrend',
                                    className="px-3 py-1 bg-green-500 text-white rounded-full text-sm font-medium mr-2"),
                            html.Button("Sideways", 
                                    id='filter-sideways',
                                    className="px-3 py-1 bg-gray-500 text-white rounded-full text-sm font-medium mr-2"),
                            html.Button("Downtrend", 
                                    id='filter-downtrend',
                                    className="px-3 py-1 bg-red-500 text-white rounded-full text-sm font-medium mr-2")
                        ], className="flex flex-wrap mb-4 p-2")
                    ], className="p-4 border-b sticky top-16 bg-white z-10"),
                    
                    # Content container dengan scrolling yang benar
                    html.Div(
                        id="trend-content",
                        className="flex-grow px-4 py-2 space-y-4",
                        style={"maxHeight": "70vh", "overflowY": "auto"}
                    ),
                    
                    # Market Overview (sticky at bottom) akan ditambahkan dinamis
                ], className="bg-white rounded-lg shadow-lg flex flex-col h-[90vh]")
            ], id='trend-notification',
            className="fixed top-16 right-4 w-[600px] bg-white shadow-lg rounded-lg z-50 transform transition-all duration-300 opacity-0 scale-95 pointer-events-none"),


            # Interval untuk update otomatis
            dcc.Interval(id='trend-update-interval',
                        interval=5*60*1000,  # Update setiap 5 menit
                        n_intervals=0)
        ], className="z-50"),


    


        # Sidebar
        html.Div([
            # Sidebar Header
            html.Div([
                html.H1("Control Panel", className="text-3xl font-bold mb-2"),
                html.Button(
                    "≪",
                    id='sidebar-toggle',
                    className="absolute -right-3 top-4 bg-white rounded-full p-2 shadow-md hover:bg-gray-100"
                )
            ], className="relative mb-2 p-4"),

            # Sidebar Content
            html.Div([
                # Dataset Period Selector
                html.Div([
                    html.H2("Data Period", className="text-lg font-semibold mb-2"),
                    dcc.RadioItems(
                        id='data-period-selector',
                        options=[
                            {'label': ' Data Daily 1 Tahun Terakhir', 'value': 'max'},
                            {'label': ' Data Hourly 1 Tahun Terakhir', 'value': '1y'},
                            {'label': ' Data Realtime', 'value': 'realtime'},
                        ],
                        value='max',  # Changed from '1y' to 'max'
                        className="space-y-2"
                    ),
                    # Add conditional dropdown for realtime intervals
                    html.Div(
                        id='realtime-interval-container',
                        style={'display': 'none'},
                        children=[
                            html.H3("Select Time Interval", className="text-md font-semibold mt-3 mb-2"),
                            dcc.Dropdown(
                                id='realtime-interval-selector',
                                options=[
                                    {'label': '1 Minute', 'value': '1min'},
                                    {'label': '5 Minutes', 'value': '5min'},
                                    {'label': '10 Minutes', 'value': '10min'},
                                    {'label': '15 Minutes', 'value': '15min'},
                                    {'label': '30 Minutes', 'value': '30min'},
                                    {'label': '45 Minutes', 'value': '45min'},
                                    {'label': '1 Hour', 'value': '1h'},
                                    {'label': '1 Day', 'value': '1d'},
                                    {'label': '1 Week', 'value': '1w'},
                                    {'label': '1 Month', 'value': '1m'}
                                ],
                                placeholder="Select interval",
                                className="w-full"
                            )
                        ]
                    )
                ], className="mb-6"),

                # Ticker Selection
                html.Div([
                    html.H2("Select Ticker", className="text-lg font-semibold mb-2"),
                    dcc.Loading(
                        id="loading-ticker",
                        type="default",
                        children=[
                            dcc.Dropdown(
                                id='ticker-dropdown',
                                className="w-full"
                            )
                        ]
                    )
                ], className="mb-6"),

                # Date Range
                html.Div([
                    html.H2("Date Range", className="text-lg font-semibold mb-2"),
                    html.Div([
                        # Start Date
                        html.Div([
                            html.Label("Start Date", className="block text-sm mb-1"),
                            html.Div([
                                dcc.Dropdown(
                                    id='start-year-dropdown',
                                    placeholder="Year",
                                    className="w-full"
                                ),
                                dcc.Dropdown(
                                    id='start-month-dropdown',
                                    placeholder="Month",
                                    className="w-full"
                                ),
                                dcc.Dropdown(
                                    id='start-day-dropdown',
                                    placeholder="Day",
                                    className="w-full"
                                ),
                            ], className="grid grid-cols-3 gap-2")
                        ], className="mb-4"),
                        
                        # End Date
                        html.Div([
                            html.Label("End Date", className="block text-sm mb-1"),
                            html.Div([
                                dcc.Dropdown(
                                    id='end-year-dropdown',
                                    placeholder="Year",
                                    className="w-full"
                                ),
                                dcc.Dropdown(
                                    id='end-month-dropdown',
                                    placeholder="Month",
                                    className="w-full"
                                ),
                                dcc.Dropdown(
                                    id='end-day-dropdown',
                                    placeholder="Day",
                                    className="w-full"
                                ),
                            ], className="grid grid-cols-3 gap-2")
                        ])
                    ], className="space-y-4")
                ], className="mb-6"),

                # Technical Analysis
                # Technical Analysis Section with Dynamic Parameters
                html.Div([
                    html.H2("Technical Analysis", className="text-lg font-semibold mb-2"),
                    dcc.Checklist(
                        id='technical-checklist',
                        options=[
                            {'label': ' Bollinger Bands', 'value': 'Bollinger_Signal'},
                            {'label': ' Moving Average', 'value': 'MA_Signal'},
                            {'label': ' RSI', 'value': 'RSI_Signal'},
                            {'label': ' MACD', 'value': 'MACD_Signal'},
                            {'label': ' ADX', 'value': 'ADX_Signal'},
                            {'label': ' Volume', 'value': 'Volume_Signal'},
                            {'label': ' Fibonacci Retracement', 'value': 'Fibonacci_Signal'},
                            {'label': ' Candlestick Patterns', 'value': 'Candlestick_Signal'},
                        ],
                        value=['Bollinger_Signal'],
                        className="space-y-2"
                    ),

                    html.Div([
                        html.Div([
                            html.H2("Explanation Chart", className="text-1xl font-semibold mb-2"),
                            html.Label("Show Explanations:", className="text-sm"),
                            dcc.RadioItems(
                                id='show-explanations',
                                options=[
                                    {'label': ' Yes', 'value': 'yes'},
                                    {'label': ' No', 'value': 'no'}
                                ],
                                value='yes',
                                className="space-y-2"
                            ),
                            html.P("Toggle explanations for charts below each graph.", className="text-xs text-gray-500 mt-1")
                        ], className="border border-gray-200 rounded p-4 my-4"),
                    ], className="mb-6"),
                    
                    # Parameters Container - will show/hide based on selection
                    html.Div([
                        # Bollinger Bands Parameters
                        html.Div(
                            id='bollinger-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("Bollinger Bands Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Period", className="text-xs"),
                                    dcc.Input(
                                        id='bb-period',
                                        type='number',
                                        value=20,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Jumlah periode (bar/hari) untuk menghitung Simple Moving Average (SMA) yang menjadi dasar pita Bollinger.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Standard Deviation", className="text-xs"),
                                    dcc.Input(
                                        id='bb-std',
                                        type='number',
                                        value=2,
                                        min=0.1,
                                        step=0.1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Menentukan lebar pita atas dan bawah berdasarkan berapa deviasi standar dari SMA.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Signal Mode", className="text-xs"),
                                    dcc.RadioItems(
                                        id='bb-signal-mode',
                                        options=[
                                            {'label': 'Touch Bands (Default)', 'value': 'touch'},
                                            {'label': 'Cross Bands (Strict)', 'value': 'cross'}
                                        ],
                                        value='touch',
                                        className="mt-1"
                                    ),
                                    html.P("Touch: Sinyal ketika harga menyentuh band. Cross: Sinyal ketika harga menembus band.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # Moving Average Parameters
                        html.Div(
                            id='ma-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("Moving Average Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Number of MA Lines", className="text-xs"),
                                    dcc.RadioItems(
                                        id='ma-lines-count',
                                        options=[
                                            {'label': '2 Lines (Fast & Slow)', 'value': '2'},
                                            {'label': '3 Lines (Fast, Medium & Slow)', 'value': '3'}
                                        ],
                                        value='2',
                                        className="mt-1 mb-2"
                                    ),
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Fast MA Period", className="text-xs"),
                                    dcc.Input(
                                        id='ma-short',
                                        type='number',
                                        value=20,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode rata-rata bergerak jangka pendek untuk mengukur tren harga yang lebih responsif.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Medium MA Period", className="text-xs"),
                                    dcc.Input(
                                        id='ma-medium',
                                        type='number',
                                        value=35,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode rata-rata bergerak menengah (hanya untuk mode 3 lines).", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2", id='ma-medium-container'),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Slow MA Period", className="text-xs"),
                                    dcc.Input(
                                        id='ma-long',
                                        type='number',
                                        value=50,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode rata-rata bergerak jangka panjang untuk mengidentifikasi tren jangka lebih luas dan mengurangi noise.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # RSI Parameters
                        html.Div(
                            id='rsi-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("RSI Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Period", className="text-xs"),
                                    dcc.Input(
                                        id='rsi-period',
                                        type='number',
                                        value=14,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Jumlah periode untuk menghitung Relative Strength Index (RSI), biasanya 14 hari.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Overbought Level", className="text-xs"),
                                    dcc.Input(
                                        id='rsi-overbought',
                                        type='number',
                                        value=70,
                                        min=50,
                                        max=100,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Tingkat di atasnya dianggap overbought; potensi pembalikan turun.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Oversold Level", className="text-xs"),
                                    dcc.Input(
                                        id='rsi-oversold',
                                        type='number',
                                        value=30,
                                        min=0,
                                        max=50,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Tingkat di bawahnya dianggap oversold; potensi pembalikan naik.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # MACD Parameters
                        html.Div(
                            id='macd-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("MACD Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Fast Period", className="text-xs"),
                                    dcc.Input(
                                        id='macd-fast',
                                        type='number',
                                        value=12,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode EMA cepat; respon lebih cepat terhadap perubahan harga.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Slow Period", className="text-xs"),
                                    dcc.Input(
                                        id='macd-slow',
                                        type='number',
                                        value=26,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode EMA lambat; membantu meredam noise dan mengonfirmasi tren.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Signal Period", className="text-xs"),
                                    dcc.Input(
                                        id='macd-signal',
                                        type='number',
                                        value=9,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode EMA untuk garis sinyal yang memicu sinyal beli/jual.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # ADX Parameters
                        html.Div(
                            id='adx-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("ADX Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Period", className="text-xs"),
                                    dcc.Input(
                                        id='adx-period',
                                        type='number',
                                        value=14,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Jumlah periode untuk menghitung Average Directional Index (ADX) mengukur kekuatan tren.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                html.Hr(className="my-2 border-gray-300"),
                                html.Div([
                                    html.Label("Threshold", className="text-xs"),
                                    dcc.Input(
                                        id='adx-threshold',
                                        type='number',
                                        value=25,
                                        min=0,
                                        max=100,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Nilai ADX di atas threshold menunjukkan tren yang kuat.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # Volume Parameters
                        html.Div(
                            id='volume-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("Volume Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("MA Period", className="text-xs"),
                                    dcc.Input(
                                        id='volume-period',
                                        type='number',
                                        value=20,
                                        min=1,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Periode rata-rata untuk menghitung rata-rata volume sebagai indikator konfirmasi tren.", className="text-xs text-gray-500 mt-1")
                                ])
                            ]
                        ),

                        # Fibonacci Parameters
                        html.Div(
                            id='fibonacci-params',
                            className="border border-gray-200 rounded p-4 my-4",
                            style={'display': 'none'},
                            children=[
                                html.H3("Fibonacci Parameters", className="text-sm font-semibold mb-2"),
                                html.Div([
                                    html.Label("Lookback Period", className="text-xs"),
                                    dcc.Input(
                                        id='fibonacci-lookback',
                                        type='number',
                                        value=60,
                                        min=10,
                                        className="w-full p-1 border rounded text-sm"
                                    ),
                                    html.P("Jumlah bar/hari terakhir untuk mencari high/low Fibonacci.", className="text-xs text-gray-500 mt-1")
                                ], className="mb-2"),
                                # Optionally: add custom retracement levels input
                            ]
                        ),

                        html.Div(
                          id='candlestick-params',
                          className="border border-gray-200 rounded p-4 my-4",
                          style={'display': 'none'},
                          children=[
                              html.H3("Candlestick Pattern Parameters", className="text-sm font-semibold mb-2"),
                              html.Div([
                                  html.Label("Confidence Threshold", className="text-xs"),
                                  dcc.Slider(
                                      id='candlestick-confidence',
                                      min=0,
                                      max=100,
                                      step=5,
                                      value=50,
                                      marks={i: f'{i}%' for i in range(0, 101, 25)},
                                      className="w-full"
                                  ),
                                  html.P("Minimum confidence level required for candlestick signals (higher values mean fewer but stronger signals).", 
                                        className="text-xs text-gray-500 mt-1")
                              ], className="mb-2"),
                              html.Div([
                                  html.Label("Pattern Categories", className="text-xs"),
                                  dcc.Checklist(
                                      id='candlestick-categories',
                                      options=[
                                          {'label': ' Reversal Patterns', 'value': 'reversal'},
                                          {'label': ' Continuation Patterns', 'value': 'continuation'},
                                          {'label': ' Indecision Patterns', 'value': 'indecision'}
                                      ],
                                      value=['reversal', 'continuation'],
                                      className="space-y-1"
                                  ),
                                  html.P("Select which types of candlestick patterns to include in analysis.",
                                        className="text-xs text-gray-500 mt-1")
                              ]),
                          ]
                      ),

                    ], className="mt-4")
                ], className="mb-6"),

                html.Div([
                    html.Button("Backtesting", id="backtesting-button", className="bg-blue-500 text-white px-4 py-2 rounded")
                ], className="flex justify-end mt-4"),

                html.Div([
                    html.Button([
                        html.Div([
                            html.I(className="fas fa-chart-line mr-2"), 
                            "Bulk Analysis"
                        ], className="flex items-center")
                    ], id="bulk-analysis-button", className="bg-blue-500 text-white px-4 py-2 rounded w-full mb-4")
                ], className="mt-4"),



            ], className="p-4 space-y-4")
        ], id='sidebar', className="w-96 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300"),

        # Main Content
        html.Div([
            # Title for Technical Analysis
            html.H1("DASHBOARD TREN HARGA SAHAM", className="text-3xl font-bold mb-8 text-center text-gray-800"),
            
            # Charts with Loading
            dcc.Loading(
                id="loading-charts",
                type="default",
                children=[html.Div(id='charts-container', className="mb-8")]
            ),
            
            # Table with Loading
            dcc.Loading(
                id="loading-table",
                type="default",
                children=[
                    html.Div([
                        html.H2("Data Table", className="text-xl font-bold mb-4"),
                        dash_table.DataTable(
                            id='stock-table',
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            style_header={
                                'backgroundColor': 'rgb(240, 240, 240)',
                                'fontWeight': 'bold'
                            },
                            page_size=10
                        )
                    ], className="bg-white rounded-lg shadow p-4 mb-8")
                ]
            ),
            
            # Signal Summary with Loading
            dcc.Loading(
                id="loading-signals",
                type="default",
                children=[
                    html.Div([
                        html.H2("Signal Summary", className="text-xl font-bold mb-4"),
                        html.Div(id='interactive-signal-table', className="mb-6"),
                        
                        # Save Controls
                        html.Div([
                            html.Div([
                                dcc.Input(
                                    id='save-title',
                                    type='text',
                                    placeholder='Enter title for signals',
                                    className="flex-1 p-2 border rounded"
                                ),
                                html.Button(
                                    'Save Signals',
                                    id='save-button',
                                    className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                                )
                            ], className="flex items-center gap-4"),
                            html.Div(id='save-status', className="mt-2 text-sm text-gray-600")
                        ], className="border-t pt-4")
                    ], className="bg-white rounded-lg shadow p-4")
                ]
            )
        ], id='main-content', className="flex-1 p-6 overflow-x-hidden overflow-y-auto bg-gray-50")
    ], className="flex h-screen bg-gray-100")
], className="h-screen")


# Add this callback after your existing callbacks
@app.callback(
    Output('realtime-interval-container', 'style'),
    [Input('data-period-selector', 'value')]
)
def toggle_realtime_interval(selected_period):
    if selected_period == 'realtime':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('sidebar', 'className', allow_duplicate=True),
     Output('show-sidebar-button', 'style', allow_duplicate=True)],
    [Input('bulk-analysis-button', 'n_clicks')],
    prevent_initial_call=True
)
def navigate_to_bulk_analysis(n_clicks):
    """Navigate to the bulk analysis page"""
    if n_clicks:
        return (
            render_bulk_analysis_page(),
            "w-0 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300",
            {'display': 'block'}
        )
    return dash.no_update, dash.no_update, dash.no_update

# register_bulk_analysis_callbacks(app)

register_backtesting_callbacks(app)


# Trend notification now uses precomputed results from the database.
# Remove or comment out the definition of get_trend_notification_data
# def get_trend_notification_data(period='1y'):
#     ... (function body) ...

@app.callback(
    [Output("trend-notification", "className", allow_duplicate=True),
     Output("trend-content", "children", allow_duplicate=True),
     Output("notification-count", "children", allow_duplicate=True)],
    [Input("notification-bell", "n_clicks"),
     Input("close-notification", "n_clicks"),
     Input("filter-all", "n_clicks"),
     Input("filter-uptrend", "n_clicks"),
     Input("filter-sideways", "n_clicks"),
     Input("filter-downtrend", "n_clicks"),
     Input("trend-update-interval", "n_intervals")],
    [State("trend-notification", "className"),
     State("ticker-search", "value")],
    prevent_initial_call=True
)
def toggle_notification_and_update_content(bell_clicks, close_clicks, all_clicks, uptrend_clicks, sideways_clicks, downtrend_clicks, interval, current_class, search_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_class, "", ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    base_class = 'fixed top-16 right-4 w-[600px] bg-white shadow-lg rounded-lg z-50 transform transition-all duration-300'

    # Handle opening and closing of the notification panel
    if trigger_id == 'notification-bell':
        if 'opacity-0' in current_class:
            return f"{base_class} opacity-100 scale-100 pointer-events-auto", dash.no_update, dash.no_update
        return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update

    if trigger_id == 'close-notification':
        return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update

    # Fetch all trends data
    all_trends = get_all_trends_from_db()
    trend_groups = {
        'Strong Uptrend': [],
        'Uptrend': [],
        'Sideways': [],
        'Downtrend': [],
        'Strong Downtrend': []
    }

    # Group tickers by their overall trend
    for ticker, trends in all_trends.items():
        trend_groups[trends['overall']].append((ticker, trends))

    # Filter trends based on user selection
    content = []
    if trigger_id == 'filter-all':
        for trend_type, tickers in trend_groups.items():
            if tickers:
                content.append(create_trend_section(trend_type, tickers))
    elif trigger_id == 'filter-uptrend':
        for trend_type in ['Strong Uptrend', 'Uptrend']:
            if trend_groups[trend_type]:
                content.append(create_trend_section(trend_type, trend_groups[trend_type]))
    elif trigger_id == 'filter-sideways':
        if trend_groups['Sideways']:
            content.append(create_trend_section('Sideways', trend_groups['Sideways']))
    elif trigger_id == 'filter-downtrend':
        for trend_type in ['Strong Downtrend', 'Downtrend']:
            if trend_groups[trend_type]:
                content.append(create_trend_section(trend_type, trend_groups[trend_type]))

    # Handle search functionality
    if search_value:
        search_value = search_value.upper()
        filtered_content = []
        for trend_type, tickers in trend_groups.items():
            matching_tickers = [(ticker, trends) for ticker, trends in tickers if search_value in ticker]
            if matching_tickers:
                filtered_content.append(create_trend_section(trend_type, matching_tickers))
        content = filtered_content

    # Add market overview at the bottom
    market_overview = html.Div([
        html.H4("Market Overview", className="text-lg font-bold mb-2"),
        html.Div([
            html.Div([
                html.Span("Bullish: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Strong Uptrend']) + len(trend_groups['Uptrend'])),
                    className="text-green-600"
                )
            ], className="mr-4"),
            html.Div([
                html.Span("Bearish: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Strong Downtrend']) + len(trend_groups['Downtrend'])),
                    className="text-red-600"
                )
            ], className="mr-4"),
            html.Div([
                html.Span("Neutral: ", className="font-semibold"),
                html.Span(
                    str(len(trend_groups['Sideways'])),
                    className="text-gray-600"
                )
            ])
        ], className="flex")
    ], className="mt-4 p-4 bg-white rounded-lg shadow-sm")
    content.append(market_overview)

    # Wrap content with dcc.Loading
    loading_content = dcc.Loading(
        id="loading-trend-content",
        type="circle",
        children=html.Div(
            content,
            className="max-h-[70vh] overflow-y-auto px-4 scrollbar scrollbar-thumb-gray-500 scrollbar-track-gray-200"
        )
    )

    # Update notification count
    significant_changes = sum(
        1 for trends in all_trends.values()
        if trends['overall'] in ['Strong Uptrend', 'Strong Downtrend']
    )
    return current_class, loading_content, str(significant_changes)




@app.callback(
    [Output("trend-notification", "className", allow_duplicate=True),
     Output("trend-content", "children", allow_duplicate=True),
     Output("notification-count", "children", allow_duplicate=True)],
    [Input("notification-bell", "n_clicks"),
     Input("close-notification", "n_clicks"),
     Input("trend-update-interval", "n_intervals")],
    [State("trend-notification", "className")],
     prevent_initial_call = True
)
def toggle_notification_and_run_analysis(bell_clicks, close_clicks, interval, current_class):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_class, "", ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    base_class = 'fixed top-16 right-4 w-[600px] bg-white shadow-lg rounded-lg z-50 transform transition-all duration-300'

    # Handle opening and closing of the notification panel
    if trigger_id == 'notification-bell':
        if 'opacity-0' in current_class:
            # Run trend analysis for all tickers
            all_trends = get_all_trends_from_db()
            trend_groups = {
                'Strong Uptrend': [],
                'Uptrend': [],
                'Sideways': [],
                'Downtrend': [],
                'Strong Downtrend': []
            }

            # Group tickers by their overall trend
            for ticker, trends in all_trends.items():
                trend_groups[trends['overall']].append((ticker, trends))

            # Generate content for the notification panel
            content = []
            for trend_type, tickers in trend_groups.items():
                if tickers:
                    content.append(create_trend_section(trend_type, tickers))

            # Add market overview at the bottom
            market_overview = html.Div([
                html.H4("Market Overview", className="text-lg font-bold mb-2"),
                html.Div([
                    html.Div([
                        html.Span("Bullish: ", className="font-semibold"),
                        html.Span(
                            str(len(trend_groups['Strong Uptrend']) + len(trend_groups['Uptrend'])),
                            className="text-green-600"
                        )
                    ], className="mr-4"),
                    html.Div([
                        html.Span("Bearish: ", className="font-semibold"),
                        html.Span(
                            str(len(trend_groups['Strong Downtrend']) + len(trend_groups['Downtrend'])),
                            className="text-red-600"
                        )
                    ], className="mr-4"),
                    html.Div([
                        html.Span("Neutral: ", className="font-semibold"),
                        html.Span(
                            str(len(trend_groups['Sideways'])),
                            className="text-gray-600"
                        )
                    ])
                ], className="flex")
            ], className="p-4 bg-white rounded-lg shadow-sm sticky bottom-0 z-10")

            # Update notification count
            significant_changes = sum(
                1 for trends in all_trends.values()
                if trends['overall'] in ['Strong Uptrend', 'Strong Downtrend']
            )

            # Wrap content with proper scrolling container
            scrollable_content = html.Div([
                html.Div(content, className="pb-16"),  # Add padding at bottom for market overview
                market_overview
            ], className="h-[70vh] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100")

            return f"{base_class} opacity-100 scale-100 pointer-events-auto", scrollable_content, str(significant_changes)

        return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update

    if trigger_id == 'close-notification':
        return f"{base_class} opacity-0 scale-95 pointer-events-none", dash.no_update, dash.no_update

    return current_class, "", ""













@app.callback(
    [Output('main-content', 'children'),
     Output('sidebar', 'className', allow_duplicate=True),  # Add allow_duplicate
     Output('show-sidebar-button', 'style', allow_duplicate=True)],  # Add allow_duplicate
    [Input('backtesting-button', 'n_clicks')],
    prevent_initial_call=True
)
def navigate_to_backtesting(n_clicks):
    """Callback for the main navigation button that takes users to the backtesting section"""
    if n_clicks:
        return (
            render_backtesting_page(),
            "w-0 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300",
            {'display': 'block'}
        )
    return dash.no_update, dash.no_update, dash.no_update


def render_backtesting_page():
    """Renders the backtesting page with improved navigation bar"""
    return html.Div([
        # Navigation bar
        html.Nav(
            html.Div([
                # Left section with back link and title
                html.Div([
                    html.A(
                        "← Back",
                        href="/",
                        className="inline-block px-4 py-2 text-white hover:bg-blue-700 rounded-lg mr-4 transition-colors"
                    ),
                    html.H1("Backtesting", className="text-xl font-bold text-white")
                ], className="flex items-center"),
                
                # Navigation buttons container (right section)
                html.Div([
                    html.Button(
                        "Analysis",
                        id="btn-backtesting-analysis",
                        className="px-4 py-2 text-white hover:text-white border-b-2 border-white",
                        n_clicks=0
                    ),
                    html.Button(
                        "Profit Analysis",
                        id="btn-backtesting-profit",
                        className="px-4 py-2 ml-4 text-white hover:text-white border-b-2 border-transparent hover:border-white transition-all duration-200",
                        n_clicks=0
                    )
                ], className="flex items-center")
            ], className="container mx-auto px-6 py-3 flex justify-between items-center"),
            className="bg-blue-600 shadow-lg mb-6"
        ),

        # Content container
        html.Div(
            id="backtesting-content",
            children=render_backtesting_analysis(),
            className="container mx-auto px-6"
        )
    ])



# Add this callback to handle navigation button highlighting
@app.callback(
    [Output("btn-backtesting-analysis", "className"),
     Output("btn-backtesting-profit", "className")],
    [Input("btn-backtesting-analysis", "n_clicks"),
     Input("btn-backtesting-profit", "n_clicks")]
)
def update_active_button(analysis_clicks, profit_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default to analysis being active
        return "px-4 py-2 text-white hover:text-white border-b-2 border-white", "px-4 py-2 ml-4 text-white hover:text-white border-b-2 border-transparent hover:border-white transition-all duration-200"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-backtesting-analysis":
        return "px-4 py-2 text-white hover:text-white border-b-2 border-white", "px-4 py-2 ml-4 text-white hover:text-white border-b-2 border-transparent hover:border-white transition-all duration-200"
    else:
        return "px-4 py-2 text-white hover:text-white border-b-2 border-transparent hover:border-white transition-all duration-200", "px-4 py-2 ml-4 text-white hover:text-white border-b-2 border-white"

@app.callback(
    Output("backtesting-content", "children"),
    [Input("btn-backtesting-analysis", "n_clicks"),
     Input("btn-backtesting-profit", "n_clicks")],
    prevent_initial_call=True
)
def switch_backtesting_page(analysis_clicks, profit_clicks):
    """Callback to switch between backtesting analysis and profit pages"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Default to analysis page
        return render_backtesting_analysis()
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-backtesting-analysis":
        return render_backtesting_analysis()
    elif button_id == "btn-backtesting-profit":
        return render_backtesting_profit()
    
    # Default case
    return render_backtesting_analysis()




def render_backtesting_analysis():
    # Ambil data dari database
    df = get_backtesting_data()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Get min and max dates
    min_date = df['Datetime'].min()
    max_date = df['Datetime'].max()
    
    # Create year options
    years = list(range(min_date.year, max_date.year + 1))
    year_options = [{'label': str(year), 'value': year} for year in years]

    return html.Div([
        html.H2("Backtesting Analysis", className="text-2xl font-bold mb-4"),
        
        # Dataset Selection
        html.Div([
            html.Label("Select Dataset Title:", className="text-sm"),
            dcc.Dropdown(
                id='backtesting-title-dropdown-analysis',
                options=[{'label': title, 'value': title} for title in df['Title'].unique()],
                placeholder="Select a dataset title",
                className="w-full p-2 border rounded"
            ),
        ], className="mt-4"),

        # Analysis Selection
        html.Div([
            html.Label("Select Analysis:", className="text-sm"),
            dcc.Dropdown(
                id='analysis-dropdown-analysis',
                placeholder="Select an analysis",
                className="w-full p-2 border rounded"
            ),
        ], className="mt-4"),
        
        # Indicator Selection
        html.Div([
            html.Label("Select Indicators for Analysis:", className="text-sm"),
            dcc.Dropdown(
                id='indicator-selection',
                options=[],  # Will be populated based on dataset
                multi=True,
                placeholder="Select indicators (optional)",
                className="w-full p-2 border rounded"
            ),
        ], className="mt-4"),

        # Date Range
        html.Div([
            html.H2("Date Range", className="text-lg font-semibold mb-2"),
            html.Div([
                # Start Date
                html.Div([
                    html.Label("Start Date", className="block text-sm mb-1"),
                    html.Div([
                        dcc.Dropdown(
                            id='start-year-dropdown-analysis',
                            options=year_options,
                            placeholder="Year",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='start-month-dropdown-analysis',
                            placeholder="Month",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='start-day-dropdown-analysis',
                            placeholder="Day",
                            className="w-full"
                        ),
                    ], className="grid grid-cols-3 gap-2")
                ], className="mb-4"),

                # End Date
                html.Div([
                    html.Label("End Date", className="block text-sm mb-1"),
                    html.Div([
                        dcc.Dropdown(
                            id='end-year-dropdown-analysis',
                            options=year_options,
                            placeholder="Year",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='end-month-dropdown-analysis',
                            placeholder="Month",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='end-day-dropdown-analysis',
                            placeholder="Day",
                            className="w-full"
                        ),
                    ], className="grid grid-cols-3 gap-2")
                ])
            ], className="space-y-4")
        ], className="mt-4"),

        # Signal Filter
        html.Div([
            html.Label("Filter Signals:", className="text-sm"),
            dcc.Checklist(
                id='signal-filter-analysis',
                options=[
                    {'label': 'Buy', 'value': 'Buy'},
                    {'label': 'Sell', 'value': 'Sell'},
                    {'label': 'Hold', 'value': 'Hold'}
                ],
                value=['Buy', 'Sell', 'Hold'],
                className="flex space-x-4"
            )
        ], className="mt-4"),

        html.Div(id='backtesting-table-container', className="mt-4"),

        html.Div(id="signal-line-chart", className="mt-6"),

        # Correlation Analysis Button
        html.Div([
            html.Button(
                "Analyze Indicator Correlation",
                id="run-correlation-analysis",
                className="bg-purple-500 text-white px-4 py-2 rounded mt-2 flex items-center justify-center"
            ),
            html.Div(id="signal-correlation-analysis", className="mt-4"),
        ], className="mt-4 mb-4"),

        # Capital Input & Run Button
        html.Div([
            html.Label("Capital Amount:", className="text-sm"),
            dcc.Input(
                id='capital-input-analysis',
                type='number',
                placeholder='Enter initial capital',
                className="w-full p-2 border rounded"
            ),
            html.Button(
                "Run Analysis",
                id="run-analysis",
                className="bg-blue-500 text-white px-4 py-2 rounded mt-2 flex items-center justify-center"
            )
        ], className="mt-4"),

        # Results Container
        html.Div([
            dcc.Loading(
                id="loading-analysis-results",
                type="circle",
                color="#4CAF50",
                children=[
                    html.Div(
                        id="analysis-results",
                        className="mt-4",
                        children=html.Div(
                            "Enter all inputs and click 'Run Analysis' to see results.",
                            className="text-gray-500 p-4 bg-gray-50 rounded"
                        )
                    )
                ]
            )
        ], className="mt-6 mb-6"),

    ], className="p-4 bg-white rounded shadow")















# Update the run_simple_analysis function to handle indicator-based backtesting
@app.callback(
    Output("analysis-results", "children"),
    [Input("run-analysis", "n_clicks")],
    [State("backtesting-title-dropdown-analysis", "value"),
     State("capital-input-analysis", "value"),
     State("analysis-dropdown-analysis", "value"),
     State("start-year-dropdown-analysis", "value"),
     State("start-month-dropdown-analysis", "value"),
     State("start-day-dropdown-analysis", "value"),
     State("end-year-dropdown-analysis", "value"),
     State("end-month-dropdown-analysis", "value"),
     State("end-day-dropdown-analysis", "value"),
     State("signal-filter-analysis", "value"),
     State("indicator-selection", "value")]  # New state for indicator selection
)
def run_backtest_analysis(n_clicks, title, capital, selected_analysis, 
                       start_year, start_month, start_day,
                       end_year, end_month, end_day,
                       signal_filter, selected_indicators):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
        
    if not all([title, capital]):
        return html.Div("Please provide all required inputs.", className="text-red-500")

    try:
        # Get data and filter by date range
        df = get_backtesting_data()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
        
        # Filter by title and date range
        filtered_df = df[
            (df['Title'] == title) & 
            (df['Datetime'] >= start_date) &
            (df['Datetime'] <= end_date)
        ]

        if filtered_df.empty:
            return html.Div("No data available for selected criteria.", className="text-red-500")

        # Extract unique indicators from the dataset
        available_indicators = filtered_df['Analysis_info'].unique()
        
        if not selected_indicators or len(selected_indicators) == 0:
            # If no indicators selected, use all available indicators
            selected_indicators = list(available_indicators)
        
        # Group results by indicator analysis
        all_results = []  # To store results for all tested strategies
        combination_results = []  # To store results for combination strategies
        
        # First, test individual indicators
        for indicator in selected_indicators:
            indicator_df = filtered_df[filtered_df['Analysis_info'] == indicator]
            
            if indicator_df.empty:
                continue
            
            # Sort by datetime to ensure chronological order
            indicator_df = indicator_df.sort_values('Datetime')
            
            # Get buy and sell signals
            buy_signals = indicator_df[indicator_df['Signal'] == 'Buy']
            sell_signals = indicator_df[indicator_df['Signal'] == 'Sell']
            
            if len(buy_signals) == 0 or len(sell_signals) == 0:
                continue
                
            # Test different combinations of buy/sell signals
            for buy_idx in range(min(3, len(buy_signals))):  # Test up to first 3 buy signals
                for sell_idx in range(min(3, len(sell_signals))):  # Test up to first 3 sell signals
                    strategy_name = f"{indicator} (Buy {buy_idx+1}, Sell {sell_idx+1})"
                    
                    # Reset for this test
                    initial_capital = float(capital)
                    current_capital = initial_capital
                    position = None
                    trades = []
                    
                    # Get signals for this test
                    test_buys = buy_signals.iloc[buy_idx::3]  # Take every 3rd buy signal starting from the selected index
                    test_sells = sell_signals.iloc[sell_idx::3]  # Take every 3rd sell signal starting from the selected index
                    
                    # Combine and sort signals by datetime
                    test_signals = pd.concat([
                        test_buys[['Datetime', 'Price', 'Signal', 'Volume']],
                        test_sells[['Datetime', 'Price', 'Signal', 'Volume']]
                    ]).sort_values('Datetime')
                    
                    # Run backtest simulation
                    for idx, row in test_signals.iterrows():
                        if position is None and row['Signal'] == 'Buy':
                            # Open position
                            shares = current_capital // row['Price']
                            if shares > 0:
                                position = {
                                    'entry_date': row['Datetime'],
                                    'entry_price': row['Price'],
                                    'entry_volume': row['Volume'],
                                    'shares': shares,
                                    'cost': shares * row['Price'],
                                    'indicator': indicator
                                }
                                current_capital -= position['cost']
                                
                        elif position is not None and row['Signal'] == 'Sell':
                            # Close position
                            proceeds = position['shares'] * row['Price']
                            profit = proceeds - position['cost']
                            
                            # Calculate price and volume changes
                            price_change_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
                            volume_change_pct = ((row['Volume'] - position['entry_volume']) / position['entry_volume']) * 100 if position['entry_volume'] > 0 else 0
                            
                            trades.append({
                                'Entry Date': position['entry_date'],
                                'Entry Price': position['entry_price'],
                                'Entry Volume': position['entry_volume'],
                                'Exit Date': row['Datetime'],
                                'Exit Price': row['Price'],
                                'Exit Volume': row['Volume'],
                                'Shares': position['shares'],
                                'Profit/Loss': profit,
                                'Return %': (profit / position['cost']) * 100,
                                'Price Change %': price_change_pct,
                                'Volume Change %': volume_change_pct,
                                'Indicator': position['indicator']
                            })
                            current_capital += proceeds
                            position = None
                    
                    # Calculate metrics for this strategy
                    total_trades = len(trades)
                    if total_trades > 0:
                        winning_trades = len([t for t in trades if t['Profit/Loss'] > 0])
                        total_profit = sum(t['Profit/Loss'] for t in trades)
                        profit_percentage = ((current_capital - initial_capital) / initial_capital) * 100
                        avg_price_change = sum(t['Price Change %'] for t in trades) / total_trades
                        avg_volume_change = sum(t['Volume Change %'] for t in trades) / total_trades
                        
                        # Add to results
                        all_results.append({
                            'Strategy': strategy_name,
                            'Initial Capital': initial_capital,
                            'Final Capital': current_capital,
                            'Total Profit/Loss': total_profit,
                            'Return %': profit_percentage,
                            'Total Trades': total_trades,
                            'Win Rate': (winning_trades/total_trades*100) if total_trades > 0 else 0,
                            'Avg Price Change %': avg_price_change,
                            'Avg Volume Change %': avg_volume_change,
                            'Trades': trades
                        })
        
        # Test combinations of indicators (if multiple indicators were selected)
        if len(selected_indicators) > 1:
            for buy_indicator in selected_indicators:
                for sell_indicator in selected_indicators:
                    if buy_indicator == sell_indicator:
                        continue  # Skip same indicator combinations as they were already tested
                        
                    # Get buy signals from first indicator
                    buy_df = filtered_df[(filtered_df['Analysis_info'] == buy_indicator) & 
                                         (filtered_df['Signal'] == 'Buy')]
                    
                    # Get sell signals from second indicator
                    sell_df = filtered_df[(filtered_df['Analysis_info'] == sell_indicator) & 
                                          (filtered_df['Signal'] == 'Sell')]
                    
                    if buy_df.empty or sell_df.empty:
                        continue
                        
                    # Sort by datetime
                    buy_df = buy_df.sort_values('Datetime')
                    sell_df = sell_df.sort_values('Datetime')
                    
                    # Test combinations of first few signals
                    for buy_idx in range(min(2, len(buy_df))):  # First 2 buy signals
                        for sell_idx in range(min(2, len(sell_df))):  # First 2 sell signals
                            combo_name = f"{buy_indicator} (Buy {buy_idx+1}) + {sell_indicator} (Sell {sell_idx+1})"
                            
                            # Reset for this test
                            initial_capital = float(capital)
                            current_capital = initial_capital
                            position = None
                            trades = []
                            
                            # Get signals for this test
                            test_buys = buy_df.iloc[buy_idx::2]  # Every 2nd buy signal from selected index
                            test_sells = sell_df.iloc[sell_idx::2]  # Every 2nd sell signal from selected index
                            
                            # Combine and sort signals by datetime
                            test_signals = pd.concat([
                                test_buys[['Datetime', 'Price', 'Signal', 'Volume', 'Analysis_info']],
                                test_sells[['Datetime', 'Price', 'Signal', 'Volume', 'Analysis_info']]
                            ]).sort_values('Datetime')
                            
                            # Run backtest simulation
                            for idx, row in test_signals.iterrows():
                                if position is None and row['Signal'] == 'Buy' and row['Analysis_info'] == buy_indicator:
                                    # Open position
                                    shares = current_capital // row['Price']
                                    if shares > 0:
                                        position = {
                                            'entry_date': row['Datetime'],
                                            'entry_price': row['Price'],
                                            'entry_volume': row['Volume'],
                                            'shares': shares,
                                            'cost': shares * row['Price'],
                                            'buy_indicator': row['Analysis_info']
                                        }
                                        current_capital -= position['cost']
                                        
                                elif position is not None and row['Signal'] == 'Sell' and row['Analysis_info'] == sell_indicator:
                                    # Close position
                                    proceeds = position['shares'] * row['Price']
                                    profit = proceeds - position['cost']
                                    
                                    # Calculate price and volume changes
                                    price_change_pct = ((row['Price'] - position['entry_price']) / position['entry_price']) * 100
                                    volume_change_pct = ((row['Volume'] - position['entry_volume']) / position['entry_volume']) * 100 if position['entry_volume'] > 0 else 0
                                    
                                    trades.append({
                                        'Entry Date': position['entry_date'],
                                        'Entry Price': position['entry_price'],
                                        'Entry Volume': position['entry_volume'],
                                        'Exit Date': row['Datetime'],
                                        'Exit Price': row['Price'],
                                        'Exit Volume': row['Volume'],
                                        'Shares': position['shares'],
                                        'Profit/Loss': profit,
                                        'Return %': (profit / position['cost']) * 100,
                                        'Price Change %': price_change_pct,
                                        'Volume Change %': volume_change_pct,
                                        'Buy Indicator': position['buy_indicator'],
                                        'Sell Indicator': row['Analysis_info']
                                    })
                                    current_capital += proceeds
                                    position = None
                            
                            # Calculate metrics for this combination
                            total_trades = len(trades)
                            if total_trades > 0:
                                winning_trades = len([t for t in trades if t['Profit/Loss'] > 0])
                                total_profit = sum(t['Profit/Loss'] for t in trades)
                                profit_percentage = ((current_capital - initial_capital) / initial_capital) * 100
                                avg_price_change = sum(t['Price Change %'] for t in trades) / total_trades
                                avg_volume_change = sum(t['Volume Change %'] for t in trades) / total_trades
                                
                                # Add to combination results
                                combination_results.append({
                                    'Strategy': combo_name,
                                    'Initial Capital': initial_capital,
                                    'Final Capital': current_capital,
                                    'Total Profit/Loss': total_profit,
                                    'Return %': profit_percentage,
                                    'Total Trades': total_trades,
                                    'Win Rate': (winning_trades/total_trades*100) if total_trades > 0 else 0,
                                    'Avg Price Change %': avg_price_change,
                                    'Avg Volume Change %': avg_volume_change,
                                    'Trades': trades
                                })
        
        # Sort results by profitability (Return %)
        all_results = sorted(all_results, key=lambda x: x['Return %'], reverse=True)
        combination_results = sorted(combination_results, key=lambda x: x['Return %'], reverse=True)
        
        # Create analysis summary components
        components = []

        
        
        # Add overall best strategy section
        all_strategies = all_results + combination_results
        if all_strategies:
            best_strategy = max(all_strategies, key=lambda x: x['Return %'])
            components.append(html.Div([
                html.H3("Best Performing Strategy", className="text-xl font-bold mb-2 text-green-700"),
                html.Div([
                    html.Div([
                        html.Span("Strategy: ", className="font-semibold"),
                        html.Span(best_strategy['Strategy'], className="ml-1")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Return: ", className="font-semibold"),
                        html.Span(f"{best_strategy['Return %']:.2f}%", 
                                className=f"ml-1 {'text-green-600' if best_strategy['Return %'] >= 0 else 'text-red-600'}")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Total Profit: ", className="font-semibold"),
                        html.Span(f"{best_strategy['Total Profit/Loss']:,.2f}", 
                                className=f"ml-1 {'text-green-600' if best_strategy['Total Profit/Loss'] >= 0 else 'text-red-600'}")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Win Rate: ", className="font-semibold"),
                        html.Span(f"{best_strategy['Win Rate']:.2f}%", className="ml-1")
                    ], className="mb-1"),
                ], className="p-3 bg-green-50 rounded border border-green-200 mb-4")
            ]))
        
        # Add individual indicator analysis section
        if all_results:
            components.append(html.H3("Individual Indicator Analysis", className="text-xl font-bold mt-4 mb-3"))
            
            # Create a table for individual indicator results
            indicator_table = dash_table.DataTable(
                data=[{
                    'Strategy': r['Strategy'],
                    'Return %': f"{r['Return %']:.2f}%",
                    'Total Profit/Loss': f"{r['Total Profit/Loss']:,.2f}",
                    'Win Rate': f"{r['Win Rate']:.2f}%",
                    'Avg Price Change %': f"{r['Avg Price Change %']:.2f}%",
                    'Avg Volume Change %': f"{r['Avg Volume Change %']:.2f}%",
                    'Total Trades': r['Total Trades']
                } for r in all_results],
                columns=[
                    {'name': 'Strategy', 'id': 'Strategy'},
                    {'name': 'Return %', 'id': 'Return %'},
                    {'name': 'Total Profit/Loss', 'id': 'Total Profit/Loss'},
                    {'name': 'Win Rate', 'id': 'Win Rate'},
                    {'name': 'Avg Price Change %', 'id': 'Avg Price Change %'},
                    {'name': 'Avg Volume Change %', 'id': 'Avg Volume Change %'},
                    {'name': 'Total Trades', 'id': 'Total Trades'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={
                    'backgroundColor': 'rgb(240, 240, 240)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Return %} < 0'},
                        'color': 'red'
                    },
                    {
                        'if': {'filter_query': '{Return %} >= 0'},
                        'color': 'green'
                    }
                ],
                sort_action='native',
                page_size=5
            )
            components.append(indicator_table)
            
            # Add indicator insights
            if len(all_results) > 0:
                best_indicator = all_results[0]
                components.append(html.Div([
                    html.H4("Indicator Insights", className="text-lg font-semibold mt-4 mb-2"),
                    html.P([
                        f"The best individual indicator strategy was ",
                        html.Strong(best_indicator['Strategy']),
                        f", which produced a return of ",
                        html.Strong(f"{best_indicator['Return %']:.2f}%"),
                        f". This strategy had an average price change of ",
                        html.Strong(f"{best_indicator['Avg Price Change %']:.2f}%"),
                        f" and average volume change of ",
                        html.Strong(f"{best_indicator['Avg Volume Change %']:.2f}%"),
                        f" between entry and exit points."
                    ], className="mb-2"),
                    html.P([
                        "The strategy's performance was ",
                        html.Strong("primarily driven by price movements") if abs(best_indicator['Avg Price Change %']) > abs(best_indicator['Avg Volume Change %']) else html.Strong("significantly influenced by volume changes"),
                        "."
                    ], className="mb-2")
                ], className="mt-3 mb-4"))
        
        # Add combination strategy analysis section
        if combination_results:
            components.append(html.H3("Combination Strategy Analysis", className="text-xl font-bold mt-5 mb-3"))
            
            # Create a table for combination results
            combo_table = dash_table.DataTable(
                data=[{
                    'Strategy': r['Strategy'],
                    'Return %': f"{r['Return %']:.2f}%",
                    'Total Profit/Loss': f"{r['Total Profit/Loss']:,.2f}",
                    'Win Rate': f"{r['Win Rate']:.2f}%",
                    'Avg Price Change %': f"{r['Avg Price Change %']:.2f}%",
                    'Avg Volume Change %': f"{r['Avg Volume Change %']:.2f}%",
                    'Total Trades': r['Total Trades']
                } for r in combination_results],
                columns=[
                    {'name': 'Strategy', 'id': 'Strategy'},
                    {'name': 'Return %', 'id': 'Return %'},
                    {'name': 'Total Profit/Loss', 'id': 'Total Profit/Loss'},
                    {'name': 'Win Rate', 'id': 'Win Rate'},
                    {'name': 'Avg Price Change %', 'id': 'Avg Price Change %'},
                    {'name': 'Avg Volume Change %', 'id': 'Avg Volume Change %'},
                    {'name': 'Total Trades', 'id': 'Total Trades'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={
                    'backgroundColor': 'rgb(240, 240, 240)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Return %} < 0'},  # Changed from 'contains'
                        'color': 'red'
                    },
                    {
                        'if': {'filter_query': '{Return %} >= 0'},  # Changed from 'not contains'
                        'color': 'green'
                    }
                ],
                sort_action='native',
                page_size=5
            )
            components.append(combo_table)
            
            # Add combination insights
            if len(combination_results) > 0:
                best_combo = combination_results[0]
                components.append(html.Div([
                    html.H4("Combination Strategy Insights", className="text-lg font-semibold mt-4 mb-2"),
                    html.P([
                        f"The best combination strategy was ",
                        html.Strong(best_combo['Strategy']),
                        f", which produced a return of ",
                        html.Strong(f"{best_combo['Return %']:.2f}%"),
                        f". "
                    ], className="mb-2"),
                    
                    # Compare with best individual indicator
                    html.P([
                        "Compared to the best individual indicator strategy (",
                        html.Strong(all_results[0]['Strategy'] if all_results else "N/A"),
                        f"), this combination ",
                        html.Strong("performed better") if best_combo['Return %'] > all_results[0]['Return %'] else html.Strong("performed worse"),
                        f" by ",
                        html.Strong(f"{abs(best_combo['Return %'] - all_results[0]['Return %']):.2f}%"),
                        "."
                    ], className="mb-2")
                ], className="mt-3 mb-4"))
        
        # Add overall conclusion section
        components.append(html.Div([
            html.H3("Analysis Conclusion", className="text-xl font-bold mt-5 mb-3"),
            html.Div([
                html.P([
                    "Based on backtesting results, ",
                    html.Strong(best_strategy['Strategy'] if all_strategies else "N/A"),
                    " is the most profitable strategy for this dataset in the selected timeframe."
                ], className="mb-2"),
                
                html.P([
                    "The analysis suggests that ",
                    html.Strong("indicator combinations outperformed individual indicators") 
                        if combination_results and combination_results[0]['Return %'] > all_results[0]['Return %'] 
                        else html.Strong("individual indicators performed better than combinations"),
                    " for this particular dataset."
                ], className="mb-2"),
                
                html.P([
                    "Trade performance was ",
                    html.Strong("more influenced by price movements") 
                        if best_strategy['Avg Price Change %'] > best_strategy['Avg Volume Change %'] 
                        else html.Strong("more influenced by volume changes"),
                    ", suggesting that ",
                    "price-based signals were more reliable indicators" 
                        if best_strategy['Avg Price Change %'] > best_strategy['Avg Volume Change %'] 
                        else "volume changes played a significant role in trade outcomes",
                    "."
                ], className="mb-2"),
            ], className="p-4 bg-blue-50 rounded border border-blue-200")
        ]))
        
        # Show detailed trades for the best strategy
        if all_strategies:
            best_strategy = max(all_strategies, key=lambda x: x['Return %'])
            if best_strategy['Trades']:
                components.append(html.H3("Trade Details for Best Strategy", className="text-xl font-bold mt-5 mb-3"))
                
                trades_table_data = []
                for i, trade in enumerate(best_strategy['Trades']):
                    # Format trade data for display
                    trade_data = {
                        'Trade #': i + 1,
                        'Entry Date': trade['Entry Date'].strftime('%Y-%m-%d %H:%M'),
                        'Entry Price': f"{trade['Entry Price']:.2f}",
                        'Exit Date': trade['Exit Date'].strftime('%Y-%m-%d %H:%M'),
                        'Exit Price': f"{trade['Exit Price']:.2f}",
                        'Profit/Loss': f"{trade['Profit/Loss']:.2f}",
                        'Return %': f"{trade['Return %']:.2f}%",
                        'Price Change %': f"{trade['Price Change %']:.2f}%",
                        'Volume Change %': f"{trade['Volume Change %']:.2f}%"
                    }
                    trades_table_data.append(trade_data)
                
                trades_table = dash_table.DataTable(
                    data=trades_table_data,
                    columns=[{'name': col, 'id': col} for col in trades_table_data[0].keys()],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={
                        'backgroundColor': 'rgb(240, 240, 240)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Profit/Loss} < 0'},  # Changed from 'contains'
                            'color': 'red'
                        },
                        {
                            'if': {'filter_query': '{Profit/Loss} >= 0'},  # Changed from 'not contains'
                            'color': 'green'
                        }
                    ],
                    page_size=5
                )
                components.append(trades_table),
        
        # Add detailed trades section for all scenarios
        components.append(html.Div([
            html.H3("All Trading Scenarios Details", className="text-xl font-bold mt-5 mb-3"),
            
            # Create tabs for different views
            dcc.Tabs([
                dcc.Tab(label="Individual Indicator Trades", children=[
                    html.Div([
                        html.H4(f"Trades for {result['Strategy']}", className="text-lg font-semibold mt-4 mb-2"),
                        dash_table.DataTable(
                            data=[{
                                'Entry Date': trade['Entry Date'].strftime('%Y-%m-%d %H:%M'),
                                'Entry Price': f"{trade['Entry Price']:.2f}",
                                'Exit Date': trade['Exit Date'].strftime('%Y-%m-%d %H:%M'),
                                'Exit Price': f"{trade['Exit Price']:.2f}",
                                'Shares': trade['Shares'],
                                'Profit/Loss': f"{trade['Profit/Loss']:.2f}",
                                'Return %': f"{trade['Return %']:.2f}%",
                                'Price Change %': f"{trade['Price Change %']:.2f}%",
                                'Volume Change %': f"{trade['Volume Change %']:.2f}%"
                            } for trade in result['Trades']],
                            columns=[
                                {'name': col, 'id': col} for col in [
                                    'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price',
                                    'Shares', 'Profit/Loss', 'Return %', 'Price Change %',
                                    'Volume Change %'
                                ]
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={
                                'backgroundColor': 'rgb(240, 240, 240)',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Profit/Loss} contains "-"'},
                                    'color': 'red'
                                },
                                {
                                    'if': {'filter_query': '{Profit/Loss} > "0.00"'},
                                    'color': 'green'
                                }
                            ],
                            sort_action='native',
                            page_size=10
                        )
                    ]) for result in all_results
                ]),

                dcc.Tab(label="Combination Strategy Trades", children=[
                    html.Div([
                        html.H4(f"Trades for {combo['Strategy']}", className="text-lg font-semibold mt-4 mb-2"),
                        dash_table.DataTable(
                            data=[{
                                'Entry Date': trade['Entry Date'].strftime('%Y-%m-%d %H:%M'),
                                'Entry Price': f"{trade['Entry Price']:.2f}",
                                'Exit Date': trade['Exit Date'].strftime('%Y-%m-%d %H:%M'),
                                'Exit Price': f"{trade['Exit Price']:.2f}",
                                'Shares': trade['Shares'],
                                'Profit/Loss': f"{trade['Profit/Loss']:.2f}",
                                'Return %': f"{trade['Return %']:.2f}%",
                                'Buy Indicator': trade.get('Buy Indicator', ''),
                                'Sell Indicator': trade.get('Sell Indicator', '')
                            } for trade in combo['Trades']],
                            columns=[
                                {'name': col, 'id': col} for col in [
                                    'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price',
                                    'Shares', 'Profit/Loss', 'Return %', 'Buy Indicator',
                                    'Sell Indicator'
                                ]
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={
                                'backgroundColor': 'rgb(240, 240, 240)',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Profit/Loss} contains "-"'},
                                    'color': 'red'
                                },
                                {
                                    'if': {'filter_query': '{Profit/Loss} > "0.00"'},
                                    'color': 'green'
                                }
                            ],
                            sort_action='native',
                            page_size=10
                        )
                    ]) for combo in combination_results
                ])
            ])
        ], className="mt-6 mb-6 p-4 bg-white rounded shadow-sm"))
        
        return html.Div(components)

    except Exception as e:
        return html.Div([
            html.H3("Error", className="text-xl font-bold text-red-600 mb-2"),
            html.P(f"An error occurred during analysis: {str(e)}", className="text-red-500"),
            html.Pre(traceback.format_exc(), className="bg-gray-100 p-4 rounded text-sm mt-2")
        ])




# Update callback for backtesting display to include the indicators
@app.callback(
    [Output("backtesting-table-container", "children", allow_duplicate=True),
     Output("analysis-results", "children", allow_duplicate=True),
     Output("signal-line-chart", "children", allow_duplicate=True)],  
    [Input("backtesting-title-dropdown-analysis", "value"),
     Input("analysis-dropdown-analysis", "value"),
     Input("indicator-selection", "value"),  # Added indicator selection
     Input("start-year-dropdown-analysis", "value"),
     Input("start-month-dropdown-analysis", "value"),
     Input("start-day-dropdown-analysis", "value"),
     Input("end-year-dropdown-analysis", "value"),
     Input("end-month-dropdown-analysis", "value"),
     Input("end-day-dropdown-analysis", "value"),
     Input("signal-filter-analysis", "value")],
    prevent_initial_call=True
)
def update_backtesting_analysis_display(title, selected_analysis, selected_indicators,
                                      start_year, start_month, start_day,
                                      end_year, end_month, end_day,
                                      signal_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    if not title:
        return html.Div("Please select a dataset.", className="text-gray-500"), None, None

    try:
        # Ambil data backtesting
        df = get_backtesting_data()
        if df.empty:
            return html.Div("No data available in the dataset.", className="text-gray-500"), None, None

        # Filter data berdasarkan judul
        backtesting_data = df[df['Title'] == title]
        if backtesting_data.empty:
            return html.Div(f"No data found for the selected title: {title}.", className="text-gray-500"), None, None

        # Filter data berdasarkan analisis atau indikator yang dipilih
        if selected_analysis:
            backtesting_data = backtesting_data[backtesting_data['Analysis_info'] == selected_analysis]
            if backtesting_data.empty:
                return html.Div(f"No data available for analysis: {selected_analysis}.", className="text-gray-500"), None, None
        elif selected_indicators:
            # Filter by selected indicators
            backtesting_data = backtesting_data[backtesting_data['Analysis_info'].isin(selected_indicators)]
            if backtesting_data.empty:
                return html.Div(f"No data available for selected indicators.", className="text-gray-500"), None, None

        # Validasi dan filter berdasarkan rentang tanggal
        try:
            start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
            end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
            
            backtesting_data['Datetime'] = pd.to_datetime(backtesting_data['Datetime'])
            backtesting_data = backtesting_data[
                (backtesting_data['Datetime'] >= start_date) & 
                (backtesting_data['Datetime'] <= end_date)
            ]
            
            if backtesting_data.empty:
                return html.Div("No data available for the selected date range.", className="text-gray-500"), None, None
                
        except Exception as e:
            return html.Div(f"Invalid date range: {str(e)}", className="text-red-500"), None, None

        # Filter signal
        if signal_filter:
            backtesting_data = backtesting_data[backtesting_data['Signal'].isin(signal_filter)]
            if backtesting_data.empty:
                return html.Div("No data available for selected signals.", className="text-gray-500"), None, None

        # Create indicator summary grouped by analysis info
        indicator_groups = backtesting_data.groupby('Analysis_info')
        indicator_summaries = []
        
        for indicator, group in indicator_groups:
            signal_counts = group['Signal'].value_counts()
            
            indicator_summaries.append(html.Div([
                html.H4(f"{indicator} Signals", className="text-md font-semibold mb-2"),
                html.Div([
                    html.Div([
                        html.Span("Buy: ", className="font-semibold"),
                        html.Span(str(signal_counts.get('Buy', 0)), className="text-green-600")
                    ], className="mr-4"),
                    html.Div([
                        html.Span("Sell: ", className="font-semibold"),
                        html.Span(str(signal_counts.get('Sell', 0)), className="text-red-600")
                    ], className="mr-4"),
                    html.Div([
                        html.Span("Hold: ", className="font-semibold"),
                        html.Span(str(signal_counts.get('Hold', 0)), className="text-blue-600")
                    ])
                ], className="flex")
            ], className="mb-3"))
        
        indicator_summary = html.Div([
            html.H3("Signal Summary by Indicator", className="text-lg font-semibold mb-3"),
            html.Div(indicator_summaries)
        ], className="mb-4 p-4 bg-white rounded shadow-sm")

        # Create data table with indicator grouping
        data_table = dash_table.DataTable(
            data=backtesting_data.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in backtesting_data.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Signal} = "Buy"'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                    'color': 'green'
                },
                {
                    'if': {'filter_query': '{Signal} = "Sell"'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'color': 'red'
                },
                {
                    'if': {'filter_query': '{Signal} = "Hold"'},
                    'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                    'color': 'blue'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            },
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_size=10
        )

        # Create line chart with indicators
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=backtesting_data['Datetime'],
            y=backtesting_data['Price'],
            mode='lines',
            name='Price',
            line=dict(color='gray', width=1)
        ))

        # Add signals by indicator
        for indicator in backtesting_data['Analysis_info'].unique():
            indicator_data = backtesting_data[backtesting_data['Analysis_info'] == indicator]
            
            # Add buy signals for this indicator
            buy_mask = indicator_data['Signal'] == 'Buy'
            if buy_mask.any():
                fig.add_trace(go.Scatter(
                    x=indicator_data.loc[buy_mask, 'Datetime'],
                    y=indicator_data.loc[buy_mask, 'Price'],
                    mode='markers',
                    name=f'{indicator} Buy',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ))
            
            # Add sell signals for this indicator
            sell_mask = indicator_data['Signal'] == 'Sell'
            if sell_mask.any():
                fig.add_trace(go.Scatter(
                    x=indicator_data.loc[sell_mask, 'Datetime'],
                    y=indicator_data.loc[sell_mask, 'Price'],
                    mode='markers',
                    name=f'{indicator} Sell',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ))
            
            # Add hold signals for this indicator
            hold_mask = indicator_data['Signal'] == 'Hold'
            if hold_mask.any():
                fig.add_trace(go.Scatter(
                    x=indicator_data.loc[hold_mask, 'Datetime'],
                    y=indicator_data.loc[hold_mask, 'Price'],
                    mode='markers',
                    name=f'{indicator} Hold',
                    marker=dict(color='blue', size=6, symbol='circle')
                ))

        fig.update_layout(
            title=f"Price and Signals for {selected_analysis or 'Selected Indicators'}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return (
            html.Div([indicator_summary, data_table]), 
            None,  # analysis-results
            dcc.Graph(figure=fig)  # signal-line-chart
        )

    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="text-red-500"), None, None





# Update the callback for indicator dropdown population
@app.callback(
    Output("indicator-selection", "options"),
    [Input("backtesting-title-dropdown-analysis", "value")]
)
def update_indicator_dropdown(title):
    if not title:
        return []

    try:
        df = get_backtesting_data()
        title_data = df[df['Title'] == title]
        
        # Get unique indicators from the dataset
        unique_indicators = title_data['Analysis_info'].unique()
        return [{'label': indicator, 'value': indicator} for indicator in unique_indicators]
    except Exception as e:
        return []

# Add a callback to analyze correlation between indicators
@app.callback(
    Output("signal-correlation-analysis", "children"),
    [Input("run-correlation-analysis", "n_clicks")],
    [State("backtesting-title-dropdown-analysis", "value"),
     State("indicator-selection", "value"),
     State("start-year-dropdown-analysis", "value"),
     State("start-month-dropdown-analysis", "value"),
     State("start-day-dropdown-analysis", "value"),
     State("end-year-dropdown-analysis", "value"),
     State("end-month-dropdown-analysis", "value"),
     State("end-day-dropdown-analysis", "value")]
)
def analyze_indicator_correlation(n_clicks, title, selected_indicators, 
                              start_year, start_month, start_day,
                              end_year, end_month, end_day):
    if not n_clicks or not title or not selected_indicators or len(selected_indicators) < 2:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Get data and filter by date range
        df = get_backtesting_data()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
        
        # Filter by title and date range
        filtered_df = df[
            (df['Title'] == title) & 
            (df['Datetime'] >= start_date) &
            (df['Datetime'] <= end_date) &
            (df['Analysis_info'].isin(selected_indicators))
        ]
        
        if filtered_df.empty:
            return html.Div("No data available for selected criteria.", className="text-red-500")
        
        # Reshape data to get signals by indicator
        signal_df = pd.DataFrame()
        
        # Add a numeric signal value column (1 for Buy, -1 for Sell, 0 for Hold)
        filtered_df['SignalValue'] = 0
        filtered_df.loc[filtered_df['Signal'] == 'Buy', 'SignalValue'] = 1
        filtered_df.loc[filtered_df['Signal'] == 'Sell', 'SignalValue'] = -1
        
        # Create pivot table with indicators as columns and datetime as index
        pivot_df = filtered_df.pivot_table(
            index='Datetime', 
            columns='Analysis_info', 
            values='SignalValue',
            aggfunc='mean'  # Use mean if there are multiple signals for the same datetime
        ).fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        # Create heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}"
        ))
        
        fig.update_layout(
            title="Indicator Signal Correlation Matrix",
            height=500,
            width=600
        )
        
        # Find indicators with highest and lowest correlation
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                ind1 = corr_matrix.columns[i]
                ind2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((ind1, ind2, corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        insights = []
        
        # Add most correlated pair
        if corr_pairs:
            most_corr = corr_pairs[0]
            insights.append(html.Div([
                html.Strong("Most correlated indicators: "),
                f"{most_corr[0]} and {most_corr[1]} with correlation of {most_corr[2]:.2f}",
                html.Br(),
                html.Span(
                    f"These indicators tend to give {'similar' if most_corr[2] > 0 else 'opposite'} signals.",
                    className="text-sm text-gray-600"
                )
            ], className="mb-2"))
        
        # Add least correlated pair
        if len(corr_pairs) > 1:
            # Find the pair with correlation closest to 0
            least_corr = min(corr_pairs, key=lambda x: abs(x[2]))
            insights.append(html.Div([
                html.Strong("Least correlated indicators: "),
                f"{least_corr[0]} and {least_corr[1]} with correlation of {least_corr[2]:.2f}",
                html.Br(),
                html.Span(
                    "These indicators tend to give independent signals.",
                    className="text-sm text-gray-600"
                )
            ], className="mb-2"))
        
        # Add negatively correlated pair if exists
        neg_corr = [pair for pair in corr_pairs if pair[2] < -0.3]
        if neg_corr:
            neg_corr = min(neg_corr, key=lambda x: x[2])  # Get most negative
            insights.append(html.Div([
                html.Strong("Contrasting indicators: "),
                f"{neg_corr[0]} and {neg_corr[1]} with negative correlation of {neg_corr[2]:.2f}",
                html.Br(),
                html.Span(
                    "These indicators often give opposite signals and could be used as confirmation filters.",
                    className="text-sm text-gray-600"
                )
            ], className="mb-2"))
        
        # Return the correlation analysis results
        return html.Div([
            html.H3("Indicator Correlation Analysis", className="text-xl font-bold mb-3"),
            dcc.Graph(figure=fig),
            html.Div([
                html.H4("Correlation Insights", className="text-lg font-semibold mt-4 mb-2"),
                html.Div(insights, className="p-3 bg-blue-50 rounded")
            ], className="mt-4")
        ])
        
    except Exception as e:
        return html.Div([
            html.H3("Error", className="text-xl font-bold text-red-600 mb-2"),
            html.P(f"An error occurred: {str(e)}", className="text-red-500")
        ])










# Add callback untuk update analisis dropdown pada backtesting analysis
@app.callback(
    Output("analysis-dropdown-analysis", "options"),
    [Input("backtesting-title-dropdown-analysis", "value")]
)
def update_analysis_dropdown_backtesting(title):
    if not title:
        return []

    try:
        df = get_backtesting_data()
        backtesting_data = df[df['Title'] == title]
    except Exception as e:
        return []

    # Get unique analyses 
    unique_analyses = backtesting_data['Analysis_info'].unique()
    return [{'label': analysis, 'value': analysis} for analysis in unique_analyses]

# Add callback untuk update date dropdowns pada backtesting analysis
@app.callback(
    [Output('start-month-dropdown-analysis', 'options'),
     Output('end-month-dropdown-analysis', 'options')],
    [Input('start-year-dropdown-analysis', 'value'),
     Input('end-year-dropdown-analysis', 'value')]
)
def update_month_dropdowns_backtesting(start_year, end_year):
    months = [
        {'label': 'January', 'value': 1},
        {'label': 'February', 'value': 2},
        {'label': 'March', 'value': 3},
        {'label': 'April', 'value': 4},
        {'label': 'May', 'value': 5},
        {'label': 'June', 'value': 6},
        {'label': 'July', 'value': 7},
        {'label': 'August', 'value': 8},
        {'label': 'September', 'value': 9},
        {'label': 'October', 'value': 10},
        {'label': 'November', 'value': 11},
        {'label': 'December', 'value': 12}
    ]
    return months, months

# Add callback untuk update day dropdowns pada backtesting analysis
@app.callback(
    [Output('start-day-dropdown-analysis', 'options'),
     Output('end-day-dropdown-analysis', 'options')],
    [Input('start-year-dropdown-analysis', 'value'),
     Input('start-month-dropdown-analysis', 'value'),
     Input('end-year-dropdown-analysis', 'value'),
     Input('end-month-dropdown-analysis', 'value')]
)
def update_day_dropdowns_backtesting(start_year, start_month, end_year, end_month):
    def get_days_in_month(year, month):
        if year and month:
            last_day = pd.Period(year=year, month=month, freq='M').days_in_month
            return [{'label': str(day), 'value': day} for day in range(1, last_day + 1)]
        return []

    start_days = get_days_in_month(start_year, start_month)
    end_days = get_days_in_month(end_year, end_month)
    return start_days, end_days

# Add callback untuk set default date ranges pada backtesting analysis
@app.callback(
    [Output('start-year-dropdown-analysis', 'value'),
     Output('start-month-dropdown-analysis', 'value'),
     Output('start-day-dropdown-analysis', 'value'),
     Output('end-year-dropdown-analysis', 'value'),
     Output('end-month-dropdown-analysis', 'value'),
     Output('end-day-dropdown-analysis', 'value')],
    [Input('backtesting-title-dropdown-analysis', 'value')]
)
def set_default_date_ranges_backtesting(title):
    df = get_backtesting_data()
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    if title:
        filtered_df = df[df['Title'] == title]
        min_date = filtered_df['Datetime'].min()
        max_date = filtered_df['Datetime'].max()
    else:
        min_date = df['Datetime'].min()
        max_date = df['Datetime'].max()

    return (min_date.year, min_date.month, min_date.day,
            max_date.year, max_date.month, max_date.day)

















def render_backtesting_profit():
    # Ambil data dari database
    df = get_backtesting_data()

    # Konversi kolom 'Datetime' menjadi tipe datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Ambil tanggal awal dan akhir dari database
    min_date = df['Datetime'].min()
    max_date = df['Datetime'].max()

    # Buat opsi untuk dropdown tahun
    years = list(range(min_date.year, max_date.year + 1))
    year_options = [{'label': str(year), 'value': year} for year in years]

    # Layout untuk halaman backtesting
    return html.Div([
        html.H2("Backtesting Summary", className="text-2xl font-bold mb-4"),

        # Dropdown untuk memilih judul dataset
        html.Div([
            html.Label("Select Dataset Title:", className="text-sm"),
            dcc.Dropdown(
                id='backtesting-title-dropdown',
                options=[{'label': title, 'value': title} for title in df['Title'].unique()],
                placeholder="Select a dataset title",
                className="w-full p-2 border rounded"
            ),
        ], className="mt-4"),

        # Dropdown untuk memilih analisis
        html.Div([
            html.Label("Select Analysis:", className="text-sm"),
            dcc.Dropdown(
                id='analysis-dropdown',
                placeholder="Select an analysis",
                className="w-full p-2 border rounded"
            ),
        ], className="mt-4"),

        # Dropdown untuk memilih rentang tanggal
        html.Div([
            html.H2("Date Range", className="text-lg font-semibold mb-2"),
            html.Div([
                # Tanggal Mulai
                html.Div([
                    html.Label("Start Date", className="block text-sm mb-1"),
                    html.Div([
                        dcc.Dropdown(
                            id='start-year-dropdown-backtesting',
                            options=year_options,
                            value=min_date.year,
                            placeholder="Year",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='start-month-dropdown-backtesting',
                            placeholder="Month",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='start-day-dropdown-backtesting',
                            placeholder="Day",
                            className="w-full"
                        ),
                    ], className="grid grid-cols-3 gap-2")
                ], className="mb-4"),

                # Tanggal Akhir
                html.Div([
                    html.Label("End Date", className="block text-sm mb-1"),
                    html.Div([
                        dcc.Dropdown(
                            id='end-year-dropdown-backtesting',
                            options=year_options,
                            value=max_date.year,
                            placeholder="Year",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='end-month-dropdown-backtesting',
                            placeholder="Month",
                            className="w-full"
                        ),
                        dcc.Dropdown(
                            id='end-day-dropdown-backtesting',
                            placeholder="Day",
                            className="w-full"
                        ),
                    ], className="grid grid-cols-3 gap-2")
                ])
            ], className="space-y-4")
        ], className="mt-4"),

        # Tambahkan filter signal di layout backtesting
        html.Div([
            html.Label("Filter Signals:", className="text-sm"),
            dcc.Checklist(
                id='signal-filter',
                options=[
                    {'label': 'Buy', 'value': 'Buy'},
                    {'label': 'Sell', 'value': 'Sell'},
                    {'label': 'Hold', 'value': 'Hold'}
                ],
                value=['Buy', 'Sell', 'Hold'],  # Default: tampilkan semua signal
                className="flex space-x-4"
            )
        ], className="mt-4"),

        # Table to display filtered data based on the selected title
        html.Div(id='backtesting-table-container', className="mt-4"),

        html.Div(id="signal-line-chart", className="mt-6"),


        # Input untuk capital dan tombol run backtesting
        html.Div([
            html.Label("Capital Amount:", className="text-sm"),
            dcc.Input(
                id='capital-input',
                type='number',
                placeholder='Enter initial capital',
                className="w-full p-2 border rounded"
            ),
            html.Button(
                "Run Backtesting",
                id="run-backtesting", 
                className="bg-green-500 text-white px-4 py-2 rounded mt-2 flex items-center justify-center"
            )
        ], className="mt-4"),

        # Bagian hasil backtesting dengan loading indicator
        html.Div([
            dcc.Loading(
                id="loading-backtesting-results",
                type="circle",
                color="#4CAF50",  # Warna hijau untuk animasi loading
                children=[
                    html.Div(
                        id="backtesting-results", 
                        className="mt-4",
                        # Tambahkan pesan awal sebelum backtesting dijalankan
                        children=html.Div(
                            "Enter all inputs and click 'Run Backtesting' to see results.",
                            className="text-gray-500 p-4 bg-gray-50 rounded"
                        )
                    )
                ]
            )
        ], className="mt-6 mb-6"),
        

    ], className="p-4 bg-white rounded shadow")


@app.callback(
    [Output('start-month-dropdown-backtesting', 'options'),
     Output('end-month-dropdown-backtesting', 'options')],
    [Input('start-year-dropdown-backtesting', 'value'),
     Input('end-year-dropdown-backtesting', 'value')]
)
def update_month_dropdowns(start_year, end_year):
    months = [
        {'label': 'January', 'value': 1},
        {'label': 'February', 'value': 2},
        {'label': 'March', 'value': 3},
        {'label': 'April', 'value': 4},
        {'label': 'May', 'value': 5},
        {'label': 'June', 'value': 6},
        {'label': 'July', 'value': 7},
        {'label': 'August', 'value': 8},
        {'label': 'September', 'value': 9},
        {'label': 'October', 'value': 10},
        {'label': 'November', 'value': 11},
        {'label': 'December', 'value': 12}
    ]
    return months, months


@app.callback(
    [Output('start-day-dropdown-backtesting', 'options'),
     Output('end-day-dropdown-backtesting', 'options')],
    [Input('start-year-dropdown-backtesting', 'value'),
     Input('start-month-dropdown-backtesting', 'value'),
     Input('end-year-dropdown-backtesting', 'value'),
     Input('end-month-dropdown-backtesting', 'value')]
)
def update_day_dropdowns(start_year, start_month, end_year, end_month):
    def get_days_in_month(year, month):
        if year and month:
            last_day = pd.Period(year=year, month=month, freq='M').days_in_month
            return [{'label': str(day), 'value': day} for day in range(1, last_day + 1)]
        return []

    start_days = get_days_in_month(start_year, start_month)
    end_days = get_days_in_month(end_year, end_month)
    return start_days, end_days


@app.callback(
    [Output('start-year-dropdown-backtesting', 'value'),
     Output('start-month-dropdown-backtesting', 'value'),
     Output('start-day-dropdown-backtesting', 'value'),
     Output('end-year-dropdown-backtesting', 'value'),
     Output('end-month-dropdown-backtesting', 'value'),
     Output('end-day-dropdown-backtesting', 'value')],
    [Input('backtesting-title-dropdown', 'value')]
)
def set_default_date_ranges(title):
    df = get_backtesting_data()
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    if title:
        filtered_df = df[df['Title'] == title]
        min_date = filtered_df['Datetime'].min()
        max_date = filtered_df['Datetime'].max()
    else:
        min_date = df['Datetime'].min()
        max_date = df['Datetime'].max()

    return (min_date.year, min_date.month, min_date.day,
            max_date.year, max_date.month, max_date.day)



# Callback untuk menampilkan data backtesting
@app.callback(
    Output("backtesting-table-container", "children"),
    [Input("backtesting-title-dropdown", "value"),
     Input("analysis-dropdown", "value"),
     Input("start-year-dropdown-backtesting", "value"),
     Input("start-month-dropdown-backtesting", "value"),
     Input("start-day-dropdown-backtesting", "value"),
     Input("end-year-dropdown-backtesting", "value"),
     Input("end-month-dropdown-backtesting", "value"),
     Input("end-day-dropdown-backtesting", "value"),
     Input("signal-filter", "value")]
)
def display_backtesting_data(title, selected_analysis, start_year, start_month, start_day, end_year, end_month, end_day, signal_filter):
    if not title:
        return html.Div("Please select a dataset.", className="text-gray-500")

    try:
        # Ambil data backtesting
        df = get_backtesting_data()
        if df.empty:
            return html.Div("No data available in the dataset.", className="text-gray-500")

        # Filter data berdasarkan judul
        backtesting_data = df[df['Title'] == title]
        if backtesting_data.empty:
            return html.Div(f"No data found for the selected title: {title}.", className="text-gray-500")

        # Filter data berdasarkan analisis
        if selected_analysis:
            backtesting_data = backtesting_data[backtesting_data['Analysis_info'] == selected_analysis]
            if backtesting_data.empty:
                return html.Div(f"No data available for analysis: {selected_analysis}.", className="text-gray-500")

        # Validasi dan filter berdasarkan rentang tanggal
        try:
            start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
            end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
            if start_date > end_date:
                return html.Div("Invalid date range: Start date is after end date.", className="text-red-500")

            backtesting_data['Datetime'] = pd.to_datetime(backtesting_data['Datetime'])
            backtesting_data = backtesting_data[
                (backtesting_data['Datetime'] >= start_date) & (backtesting_data['Datetime'] <= end_date)
            ]
            if backtesting_data.empty:
                return html.Div("No data available for the selected date range.", className="text-gray-500")
        except Exception as e:
            return html.Div(f"Invalid date range: {str(e)}", className="text-red-500")

        # Filter data berdasarkan signal yang dipilih
        backtesting_data = backtesting_data[backtesting_data['Signal'].isin(signal_filter)]
        if backtesting_data.empty:
            return html.Div("No data available for the selected signals.", className="text-gray-500")

        # **Urutkan data berdasarkan waktu**
        backtesting_data = backtesting_data.sort_values(by='Datetime').reset_index(drop=True)

        # Hitung jumlah sinyal
        buy_count = len(backtesting_data[backtesting_data['Signal'] == 'Buy'])
        sell_count = len(backtesting_data[backtesting_data['Signal'] == 'Sell'])
        hold_count = len(backtesting_data[backtesting_data['Signal'] == 'Hold'])

        # Ringkasan sinyal
        signal_summary = html.Div([
            html.H3(f"Signal Summary for {selected_analysis or 'All Analyses'}", className="text-lg font-semibold mb-3"),
            html.Div([
                html.Div([
                    html.Span("Buy Signals: ", className="font-semibold"),
                    html.Span(str(buy_count), className="text-green-600 ml-1")
                ], className="mr-6"),
                html.Div([
                    html.Span("Sell Signals: ", className="font-semibold"),
                    html.Span(str(sell_count), className="text-red-600 ml-1")
                ], className="mr-6"),
                html.Div([
                    html.Span("Hold Signals: ", className="font-semibold"),
                    html.Span(str(hold_count), className="text-blue-600 ml-1")
                ])
            ], className="flex mb-4")
        ], className="mb-4 p-4 bg-white rounded shadow-sm")

        # Tabel backtesting
        table = dash_table.DataTable(
            data=backtesting_data.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in backtesting_data.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Signal} = "Buy"'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                    'color': 'green'
                },
                {
                    'if': {'filter_query': '{Signal} = "Sell"'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                    'color': 'red'
                },
                {
                    'if': {'filter_query': '{Signal} = "Hold"'},
                    'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                    'color': 'blue'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            },
            page_size=10
        )

        return html.Div([signal_summary, table])

    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="text-red-500")


@app.callback(
    Output("analysis-dropdown", "options"),
    [Input("backtesting-title-dropdown", "value")]
)
def update_analysis_dropdown(title):
    if not title:
        return []

    # Fetch the backtesting data based on the title
    try:
        df = get_backtesting_data()  # Fetch the complete dataset
        backtesting_data = df[df['Title'] == title]
    except Exception as e:
        return []

    # Get unique analyses from the Analysis_info column
    unique_analyses = backtesting_data['Analysis_info'].unique()
    return [{'label': analysis, 'value': analysis} for analysis in unique_analyses]


@app.callback(
    Output("signal-line-chart", "children"),
    [Input("backtesting-title-dropdown", "value"),
     Input("analysis-dropdown", "value"),
     Input("start-year-dropdown-backtesting", "value"),
     Input("start-month-dropdown-backtesting", "value"),
     Input("start-day-dropdown-backtesting", "value"),
     Input("end-year-dropdown-backtesting", "value"),
     Input("end-month-dropdown-backtesting", "value"),
     Input("end-day-dropdown-backtesting", "value"),
     Input("signal-filter", "value")]
)
def update_signal_line_chart(title, selected_analysis, start_year, start_month, start_day, end_year, end_month, end_day, signal_filter):
    if not title:
        return html.Div("", className="text-gray-500")

    try:
        # Ambil data backtesting
        df = get_backtesting_data()
        if df.empty:
            return html.Div("No data available in the dataset.", className="text-gray-500")

        # Filter data berdasarkan judul
        backtesting_data = df[df['Title'] == title]
        if backtesting_data.empty:
            return html.Div(f"No data found for the selected title: {title}.", className="text-gray-500")

        # Filter data berdasarkan analisis
        if selected_analysis:
            backtesting_data = backtesting_data[backtesting_data['Analysis_info'] == selected_analysis]
            if backtesting_data.empty:
                return html.Div(f"No data available for analysis: {selected_analysis}.", className="text-gray-500")

        # Validasi dan filter berdasarkan rentang tanggal
        try:
            start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
            end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
            if start_date > end_date:
                return html.Div("Invalid date range: Start date is after end date.", className="text-red-500")

            backtesting_data['Datetime'] = pd.to_datetime(backtesting_data['Datetime'])
            backtesting_data = backtesting_data[
                (backtesting_data['Datetime'] >= start_date) & (backtesting_data['Datetime'] <= end_date)
            ]
            if backtesting_data.empty:
                return html.Div("No data available for the selected date range.", className="text-gray-500")
        except Exception as e:
            return html.Div(f"Invalid date range: {str(e)}", className="text-red-500")

        # Filter data berdasarkan signal yang dipilih
        backtesting_data = backtesting_data[backtesting_data['Signal'].isin(signal_filter)]
        if backtesting_data.empty:
            return html.Div("No data available for the selected signals.", className="text-gray-500")

        # **Urutkan data berdasarkan waktu**
        backtesting_data = backtesting_data.sort_values(by='Datetime').reset_index(drop=True)

        # Buat grafik line chart
        fig = go.Figure()

        # Tambahkan data berdasarkan sinyal
        for signal, color in [('Buy', 'green'), ('Sell', 'red'), ('Hold', 'blue')]:
            signal_data = backtesting_data[backtesting_data['Signal'] == signal]
            fig.add_trace(go.Scatter(
                x=signal_data['Datetime'],
                y=signal_data['Price'],
                mode='lines+markers',
                name=f"{signal} Signal",
                line=dict(color=color),
                marker=dict(size=8, symbol='circle')
            ))

        # Update layout grafik
        fig.update_layout(
            title="Signal Line Chart",
            xaxis_title="Datetime",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        # Bungkus grafik dengan loading
        return dcc.Loading(
            id="loading-line-chart",
            type="circle",
            color="#4CAF50",
            children=dcc.Graph(figure=fig)
        )

    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="text-red-500")


@app.callback(
    Output("backtesting-results", "children"),
    [Input("run-backtesting", "n_clicks")],
    [State("backtesting-title-dropdown", "value"),
     State("capital-input", "value"),
     State("analysis-dropdown", "value"),
     State("start-year-dropdown-backtesting", "value"),
     State("start-month-dropdown-backtesting", "value"),
     State("start-day-dropdown-backtesting", "value"),
     State("end-year-dropdown-backtesting", "value"),
     State("end-month-dropdown-backtesting", "value"),
     State("end-day-dropdown-backtesting", "value")]
)
def run_backtesting(n_clicks, title, capital, selected_analysis, start_year, start_month, start_day, end_year, end_month, end_day):
    # Jika tombol belum ditekan atau input belum lengkap
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    start_time = time.time()
    
    if not title or not capital or not selected_analysis:
        return html.Div("Please provide all inputs.", className="text-red-500")

    # Validate and format the date range
    try:
        start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
    except Exception as e:
        return html.Div(f"Invalid date range: {str(e)}", className="text-red-500")

    # Fetch the backtesting data
    try:
        df = get_backtesting_data()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        filtered_df = df[(df['Title'] == title) & (df['Analysis_info'] == selected_analysis)]
        filtered_df = filtered_df[(filtered_df['Datetime'] >= start_date) & (filtered_df['Datetime'] <= end_date)]
        if filtered_df.empty:
            return html.Div(f"No data found for the selected analysis: {selected_analysis}.", className="text-gray-500")
    except Exception as e:
        return html.Div(f"Error fetching backtesting data: {str(e)}", className="text-red-500")

    # Sort data by date to ensure proper transaction order
    filtered_df = filtered_df.sort_values(by='Datetime').reset_index(drop=True)

    # Separate Buy and Sell signals
    buy_signals = filtered_df[filtered_df['Signal'] == 'Buy'].reset_index(drop=True)
    sell_signals = filtered_df[filtered_df['Signal'] == 'Sell'].reset_index(drop=True)

    # Initialize variables for the linear backtesting scenarios
    all_scenarios = []
    scenario_id = 1
    best_scenario = None
    max_final_capital = float('-inf')

    # Generate all possible linear scenarios
    # For each buy signal, try each sell signal that comes after it
    for buy_start_idx in range(len(buy_signals)):
        for sell_start_idx in range(len(sell_signals)):
            # Skip if the sell signal doesn't come after the buy signal
            if sell_signals.iloc[sell_start_idx]['Datetime'] <= buy_signals.iloc[buy_start_idx]['Datetime']:
                continue

            # Initialize scenario variables
            scenario_transactions = []
            current_capital = float(capital)
            current_position = None
            current_buy_idx = buy_start_idx
            current_sell_idx = sell_start_idx
            
            # Track dates to ensure we don't have multiple transactions on the same date
            transaction_dates = set()
            
            # Continue trading until we can't find more valid buy/sell pairs
            while current_buy_idx < len(buy_signals) and current_sell_idx < len(sell_signals):
                buy_signal = buy_signals.iloc[current_buy_idx]
                sell_signal = sell_signals.iloc[current_sell_idx]
                
                # Verify sell comes after buy
                if sell_signal['Datetime'] <= buy_signal['Datetime']:
                    # Find the next valid sell signal
                    found_valid_sell = False
                    for next_sell_idx in range(current_sell_idx + 1, len(sell_signals)):
                        if sell_signals.iloc[next_sell_idx]['Datetime'] > buy_signal['Datetime']:
                            current_sell_idx = next_sell_idx
                            found_valid_sell = True
                            break
                    
                    if not found_valid_sell:
                        break  # No more valid sell signals
                    
                    sell_signal = sell_signals.iloc[current_sell_idx]
                
                # Calculate transaction details for this trade
                buy_date = buy_signal['Datetime']
                buy_price = buy_signal['Price']
                sell_date = sell_signal['Datetime']
                sell_price = sell_signal['Price']
                
                # Skip if we've already traded on these dates
                if buy_date in transaction_dates or sell_date in transaction_dates:
                    # Try to find the next buy signal
                    current_buy_idx += 1
                    continue
                
                # Calculate shares and profit
                shares = int(current_capital // buy_price)
                if shares == 0:  # Not enough capital
                    break
                    
                investment = shares * buy_price
                proceeds = shares * sell_price
                profit = proceeds - investment
                percentage_profit = ((sell_price - buy_price) / buy_price) * 100
                
                # Update capital
                current_capital = current_capital - investment + proceeds
                
                # Add the transaction to our scenario
                transaction = {
                    'Buy Date': buy_date,
                    'Buy Price': buy_price,
                    'Sell Date': sell_date,
                    'Sell Price': sell_price,
                    'Shares': shares,
                    'Investment': investment,
                    'Proceeds': proceeds,
                    'Profit': profit,
                    'Percentage Profit': percentage_profit,
                    'Capital After Trade': current_capital
                }
                scenario_transactions.append(transaction)
                
                # Add dates to our set of transaction dates
                transaction_dates.add(buy_date)
                transaction_dates.add(sell_date)
                
                # Find the next buy signal that comes after this sell signal
                found_valid_buy = False
                for next_buy_idx in range(buy_start_idx, len(buy_signals)):
                    if buy_signals.iloc[next_buy_idx]['Datetime'] > sell_date:
                        current_buy_idx = next_buy_idx
                        found_valid_buy = True
                        break
                
                if not found_valid_buy:
                    break  # No more valid buy signals
            
            # Only include scenarios that had at least one transaction
            if not scenario_transactions:
                continue
                
            # Calculate scenario summary
            initial_capital = float(capital)
            final_capital = scenario_transactions[-1]['Capital After Trade']
            total_profit = final_capital - initial_capital
            total_percentage_profit = (total_profit / initial_capital) * 100
            total_trades = len(scenario_transactions)
            
            # Add a summary entry
            summary = {
                'Buy Date': "SUMMARY",
                'Buy Price': "",
                'Sell Date': "",
                'Sell Price': "",
                'Shares': "",
                'Investment': "",
                'Proceeds': "",
                'Profit': total_profit,
                'Percentage Profit': total_percentage_profit,
                'Capital After Trade': final_capital
            }
            scenario_transactions.append(summary)
            
            # Create the scenario object
            scenario = {
                'Scenario ID': scenario_id,
                'Transactions': scenario_transactions,
                'Initial Capital': initial_capital,
                'Final Capital': final_capital,
                'Total Profit': total_profit,
                'Total Percentage Profit': total_percentage_profit,
                'Number of Trades': total_trades
            }
            
            all_scenarios.append(scenario)
            scenario_id += 1
            
            # Update best scenario if needed
            if final_capital > max_final_capital:
                max_final_capital = final_capital
                best_scenario = scenario

    # Sort scenarios by profitability
    all_scenarios.sort(key=lambda x: x['Final Capital'], reverse=True)

    # Limit to top X scenarios for display purposes
    top_scenarios = all_scenarios[:10] if len(all_scenarios) > 10 else all_scenarios

    # Generate the output display
    scenario_components = []

    end_time = time.time()  # Akhiri penghitungan waktu
    execution_time = end_time - start_time

    # Add a summary of the best scenario
    if best_scenario:
        best_scenario_summary = html.Div([
            html.H2(f"Best Linear Scenario for {selected_analysis}", className="text-2xl font-bold mb-4 text-green-600"),
            html.Div([
                html.Div([
                    html.Span("Initial Capital: ", className="font-semibold"),
                    html.Span(f"{best_scenario['Initial Capital']:.2f}", className="ml-1")
                ], className="mb-2"),
                html.Div([
                    html.Span("Final Capital: ", className="font-semibold"),
                    html.Span(f"{best_scenario['Final Capital']:.2f}", className="ml-1 text-green-600")
                ], className="mb-2"),
                html.Div([
                    html.Span("Total Profit: ", className="font-semibold"),
                    html.Span(f"{best_scenario['Total Profit']:.2f}", className="ml-1 text-green-600")
                ], className="mb-2"),
                html.Div([
                    html.Span("Total Percentage Profit: ", className="font-semibold"),
                    html.Span(f"{best_scenario['Total Percentage Profit']:.2f}%", className="ml-1 text-blue-600")
                ], className="mb-2"),
                html.Div([
                    html.Span("Number of Trades: ", className="font-semibold"),
                    html.Span(str(best_scenario['Number of Trades']), className="ml-1")
                ], className="mb-2"),
                html.Div([
                    html.Span("Date Range: ", className="font-semibold"),
                    html.Span(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", className="ml-1")
                ], className="mb-2"),
                    html.Div([
                    html.Span("Execution Time: ", className="font-semibold"),
                    html.Span(f"{execution_time:.2f} seconds", className="ml-1 text-blue-600")
                ])
            ], className="p-4 bg-white rounded shadow-sm")
        ], className="mb-6")
        
        scenario_components.append(best_scenario_summary)

    # Add a table showing overall statistics for top scenarios
    top_scenario_stats = []
    for scenario in top_scenarios:
        top_scenario_stats.append({
            'Scenario ID': scenario['Scenario ID'],
            'Initial Capital': f"{scenario['Initial Capital']:.2f}",
            'Final Capital': f"{scenario['Final Capital']:.2f}",
            'Total Profit': f"{scenario['Total Profit']:.2f}",
            'Total % Profit': f"{scenario['Total Percentage Profit']:.2f}%",
            'Trades': scenario['Number of Trades']
        })

    top_scenarios_table = html.Div([
        html.H2("Top Performing Scenarios", className="text-xl font-bold mb-3"),
        dash_table.DataTable(
            data=top_scenario_stats,
            columns=[{'name': col, 'id': col} for col in top_scenario_stats[0].keys()],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                }
            ]
        )
    ], className="mb-6")
    
    if top_scenario_stats:
        scenario_components.append(top_scenarios_table)

    # Generate detailed tables for each of the top scenarios
    for scenario in top_scenarios:
        scenario_id = scenario['Scenario ID']
        transactions = scenario['Transactions']
        
        # Create a table for this scenario
        table = dash_table.DataTable(
            data=transactions,
            columns=[{'name': col, 'id': col} for col in transactions[0].keys()],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Buy Date} contains "SUMMARY"'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                    'fontWeight': 'bold'
                }
            ]
        )
        
        # Add the table to our components
        scenario_detail = html.Div([
            html.H3(f"Scenario {scenario_id} Details", className="text-lg font-semibold mb-3"),
            html.Div([
                html.Div([
                    html.Span("Initial Capital: ", className="font-semibold"),
                    html.Span(f"{scenario['Initial Capital']:.2f}", className="ml-1")
                ], className="mr-6"),
                html.Div([
                    html.Span("Final Capital: ", className="font-semibold"),
                    html.Span(f"{scenario['Final Capital']:.2f}", className="ml-1 text-green-600")
                ], className="mr-6"),
                html.Div([
                    html.Span("Total Profit: ", className="font-semibold"),
                    html.Span(f"{scenario['Total Profit']:.2f}", className="ml-1 text-green-600")
                ], className="mr-6"),
                html.Div([
                    html.Span("Total % Profit: ", className="font-semibold"),
                    html.Span(f"{scenario['Total Percentage Profit']:.2f}%", className="ml-1 text-blue-600")
                ]),
            ], className="flex mb-4"),
            table
        ], className="mb-6 p-4 bg-white rounded shadow-sm")
        
        scenario_components.append(scenario_detail)

    # If no scenarios were found
    if not scenario_components:
        return html.Div("No valid trading scenarios found for the given parameters.", className="text-red-500")

    # Return all scenario components
    return html.Div([
        html.H1("Backtesting Results", className="text-2xl font-bold mb-6"),
        html.Div(f"Total scenarios evaluated: {len(all_scenarios)}", className="mb-4"),
        html.Div(scenario_components)
    ])



def update_table_based_on_title(n_clicks, title, capital):
    df = get_backtesting_data()  # Assuming this fetches the data

    # Validate input values
    if not n_clicks or not title or not capital:
        return html.Div("Please provide all inputs.", className="text-red-500")

    # Filter the DataFrame based on the selected title
    try:
        filtered_df = df[df['Title'] == title]
        if filtered_df.empty:
            return html.Div("No data found for the selected title.", className="text-gray-500")

        # Show the relevant columns
        columns_to_show = ['Datetime', 'Price', 'Signal', 'Volume']
        filtered_df = filtered_df[columns_to_show]
        filtered_df = filtered_df.sort_values(by='Datetime')  # Ensure the data is sorted by date/time

    except Exception as e:
        return html.Div(f"Error processing data: {str(e)}", className="text-red-500")

    # Simulate backtesting with adjusted logic
    total_profit = 0
    total_percentage_profit = 0
    total_cycles = 0
    results = []

    # Separate Buy and Sell signals
    buy_signals = filtered_df[filtered_df['Signal'] == 'Buy'].reset_index(drop=True)
    sell_signals = filtered_df[filtered_df['Signal'] == 'Sell'].reset_index(drop=True)

    buy_index = 0
    sell_index = 0

    # Match Buy and Sell signals
    while buy_index < len(buy_signals) and sell_index < len(sell_signals):
        buy_row = buy_signals.iloc[buy_index]
        sell_row = sell_signals.iloc[sell_index]

        # Ensure Sell happens after Buy and only if profitable
        if sell_row['Datetime'] > buy_row['Datetime'] and sell_row['Price'] > buy_row['Price']:
            shares = capital // buy_row['Price']
            profit = (sell_row['Price'] - buy_row['Price']) * shares
            percentage_profit = ((sell_row['Price'] - buy_row['Price']) / buy_row['Price']) * 100
            total_profit += profit
            total_percentage_profit += percentage_profit
            total_cycles += 1

            results.append({
                'Buy Date': buy_row['Datetime'],
                'Buy Price': buy_row['Price'],
                'Sell Date': sell_row['Datetime'],
                'Sell Price': sell_row['Price'],
                'Shares': shares,
                'Profit': profit,
                'Percentage Profit': percentage_profit,
                'Total Capital After Trade': capital + profit
            })

            # Move to the next Buy and Sell signal
            buy_index += 1
            sell_index += 1
        else:
            # If Sell signal is before the current Buy or not profitable, skip the Sell
            sell_index += 1

    # Calculate average percentage profit
    if total_cycles > 0:
        avg_percentage_profit = total_percentage_profit / total_cycles
    else:
        avg_percentage_profit = 0

    # Add summary row
    summary = {
        'Buy Date': 'Summary',
        'Buy Price': '',
        'Sell Date': '',
        'Sell Price': '',
        'Shares': '',
        'Profit': total_profit,
        'Percentage Profit': avg_percentage_profit,
        'Total Capital After Trade': ''
    }
    results.append(summary)

    # Return backtesting results as a table
    if not results:
        return html.Div("No trades executed. Please check the signals.", className="text-gray-500")

    # Generate detailed transaction summary
    transaction_summary = html.Div([
        html.H3("Transaction Summary", className="text-lg font-semibold mb-3"),
        html.Div([
            html.Div([
                html.Span("Total Trades: ", className="font-semibold"),
                html.Span(str(total_cycles), className="ml-1")
            ], className="mr-6"),
            html.Div([
                html.Span("Total Profit: ", className="font-semibold"),
                html.Span(f"{total_profit:.2f}", className="ml-1 text-green-600")
            ], className="mr-6"),
            html.Div([
                html.Span("Average Percentage Profit: ", className="font-semibold"),
                html.Span(f"{avg_percentage_profit:.2f}%", className="ml-1 text-blue-600")
            ])
        ], className="flex mb-4")
    ], className="mb-4 p-4 bg-white rounded shadow-sm")

    # Return both the table and the transaction summary
    return html.Div([
        dash_table.DataTable(
            data=results,
            columns=[{'name': col, 'id': col} for col in results[0].keys()],
            page_size=10,  # Limit rows per page
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'}
        ),
        transaction_summary
    ])

@app.callback(
    Output("backtesting-scenarios", "children"),
    [Input("run-backtesting", "n_clicks")],
    [State("backtesting-title-dropdown", "value"), State("capital-input", "value")]
)
def calculate_backtesting_scenarios(n_clicks, title, capital):
    if not n_clicks or not title or not capital:
        return html.Div("Please provide all inputs.", className="text-red-500")

    # Fetch the backtesting data
    try:
        df = get_backtesting_data()
        filtered_df = df[df['Title'] == title]
        if filtered_df.empty:
            return html.Div("No data found for the selected title.", className="text-gray-500")
    except Exception as e:
        return html.Div(f"Error fetching backtesting data: {str(e)}", className="text-red-500")

    # Sort data by date to ensure proper transaction order
    filtered_df = filtered_df.sort_values(by='Datetime').reset_index(drop=True)

    # Separate Buy and Sell signals
    buy_signals = filtered_df[filtered_df['Signal'] == 'Buy'].reset_index(drop=True)
    sell_signals = filtered_df[filtered_df['Signal'] == 'Sell'].reset_index(drop=True)

    # Initialize variables for tracking results
    scenarios = []
    scenario_index = 1

    # Iterate through each Buy signal
    for buy_index, buy_row in buy_signals.iterrows():
        # Find all Sell signals that occur after the current Buy signal
        potential_sells = sell_signals[sell_signals['Datetime'] > buy_row['Datetime']]

        if not potential_sells.empty:
            # Create a table for this Buy signal
            scenario_results = []

            # Iterate through each potential Sell signal
            for sell_index, sell_row in potential_sells.iterrows():
                # Calculate transaction details
                shares = capital // buy_row['Price']
                profit = (sell_row['Price'] - buy_row['Price']) * shares
                percentage_profit = ((sell_row['Price'] - buy_row['Price']) / buy_row['Price']) * 100

                # Append transaction details to the scenario results
                scenario_results.append({
                    'Buy Date': buy_row['Datetime'],
                    'Buy Price': buy_row['Price'],
                    'Sell Date': sell_row['Datetime'],
                    'Sell Price': sell_row['Price'],
                    'Shares': shares,
                    'Profit': profit,
                    'Percentage Profit': percentage_profit,
                    'Total Capital After Trade': capital + profit
                })

            # Add the scenario to the list of scenarios
            scenarios.append({
                'Scenario': f"Scenario {scenario_index}",
                'Results': scenario_results
            })
            scenario_index += 1

    # Generate tables for each scenario
    scenario_tables = []
    for scenario in scenarios:
        scenario_name = scenario['Scenario']
        scenario_results = scenario['Results']

        # Create a table for this scenario
        table = dash_table.DataTable(
            data=scenario_results,
            columns=[{'name': col, 'id': col} for col in scenario_results[0].keys()],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Buy Date} = "Summary"'},
                    'backgroundColor': 'rgba(240, 240, 240, 0.8)',
                    'fontWeight': 'bold'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            }
        )

        # Add the table to the list of scenario tables
        scenario_tables.append(html.Div([
            html.H3(scenario_name, className="text-lg font-semibold mb-3"),
            table
        ], className="mb-6 p-4 bg-white rounded shadow-sm"))

    # Return all scenario tables
    return html.Div(scenario_tables)


@app.callback(
    [Output('bollinger-params', 'style'),
     Output('ma-params', 'style'),
     Output('rsi-params', 'style'),
     Output('macd-params', 'style'),
     Output('adx-params', 'style'),
     Output('volume-params', 'style'),
     Output('fibonacci-params', 'style'),
     Output('candlestick-params', 'style')],
    [Input('technical-checklist', 'value')]
)
def toggle_parameter_inputs(selected_technicals):
    if not selected_technicals:
        selected_technicals = []
        
    displays = []
    for tech in ['Bollinger_Signal', 'MA_Signal', 'RSI_Signal', 
                 'MACD_Signal', 'ADX_Signal', 'Volume_Signal', 'Fibonacci_Signal', 'Candlestick_Signal']:
        if tech in selected_technicals:
            displays.append({'display': 'block'})
        else:
            displays.append({'display': 'none'})
    
    return displays



# Callback untuk toggle sidebar
@app.callback(
    [Output('sidebar', 'className'),
     Output('sidebar-toggle', 'children'),
     Output('show-sidebar-button', 'style')],
    [Input('sidebar-toggle', 'n_clicks'),
     Input('show-sidebar-button', 'n_clicks')],
    [State('sidebar', 'className')]
)
def toggle_sidebar(hide_clicks, show_clicks, current_class):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "w-96 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300", "≪", {'display': 'none'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'sidebar-toggle':
        if "w-96" in current_class:
            return "w-0 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300", "≫", {'display': 'block'}
        else:
            return "w-96 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300", "≪", {'display': 'none'}
    elif button_id == 'show-sidebar-button':
        return "w-96 bg-white shadow-lg h-screen overflow-y-auto transition-all duration-300", "≪", {'display': 'none'}
    
    return current_class, "≪", {'display': 'none'}

# Callback untuk mengisi dropdown bulan
@app.callback(
    [Output('start-month-dropdown', 'options'),
     Output('end-month-dropdown', 'options')],
    [Input('data-period-selector', 'value')]
)
def update_month_dropdowns(selected_period):
    months = [
        {'label': 'January', 'value': 1},
        {'label': 'February', 'value': 2},
        {'label': 'March', 'value': 3},
        {'label': 'April', 'value': 4},
        {'label': 'May', 'value': 5},
        {'label': 'June', 'value': 6},
        {'label': 'July', 'value': 7},
        {'label': 'August', 'value': 8},
        {'label': 'September', 'value': 9},
        {'label': 'October', 'value': 10},
        {'label': 'November', 'value': 11},
        {'label': 'December', 'value': 12}
    ]
    return months, months

# Callback untuk update days berdasarkan bulan yang dipilih
@app.callback(
    [Output('start-day-dropdown', 'options'),
     Output('end-day-dropdown', 'options')],
    [Input('start-year-dropdown', 'value'),
     Input('start-month-dropdown', 'value'),
     Input('end-year-dropdown', 'value'),
     Input('end-month-dropdown', 'value')]
)
def update_days(start_year, start_month, end_year, end_month):
    def get_days_in_month(year, month):
        if year and month:
            last_day = pd.Period(year=year, month=month, freq='M').daysinmonth
            return [{'label': str(i).zfill(2), 'value': i} for i in range(1, last_day + 1)]
        return []
    
    start_days = get_days_in_month(start_year, start_month)
    end_days = get_days_in_month(end_year, end_month)
    
    return start_days, end_days

# Callback baru untuk inisialisasi dan update dropdown tanggal
# @app.callback(
#     [Output('start-year-dropdown', 'options'),
#      Output('start-year-dropdown', 'value'),
#      Output('end-year-dropdown', 'options'),
#      Output('end-year-dropdown', 'value'),
#      Output('start-month-dropdown', 'value'),
#      Output('end-month-dropdown', 'value'),
#      Output('start-day-dropdown', 'value'),
#      Output('end-day-dropdown', 'value')],
#     [Input('data-period-selector', 'value')]
# )
# def update_date_dropdowns(selected_period):
#     data = load_and_process_data(selected_period)
    
#     # Get min and max dates from data
#     min_date = data['Datetime'].min()
#     max_date = data['Datetime'].max()
    
#     years = list(range(min_date.year, max_date.year + 1))
#     year_options = [{'label': str(year), 'value': year} for year in years]
    
#     # Set default values
#     default_start_year = min_date.year
#     default_end_year = max_date.year
#     default_start_month = min_date.month
#     default_end_month = max_date.month
#     default_start_day = min_date.day
#     default_end_day = max_date.day
    
#     return (year_options, default_start_year,
#             year_options, default_end_year,
#             default_start_month, default_end_month,
#             default_start_day, default_end_day)


@app.callback(
    [Output('start-year-dropdown', 'options'),
     Output('start-year-dropdown', 'value'),
     Output('end-year-dropdown', 'options'),
     Output('end-year-dropdown', 'value'),
     Output('start-month-dropdown', 'value'),
     Output('end-month-dropdown', 'value'),
     Output('start-day-dropdown', 'value'),
     Output('end-day-dropdown', 'value')],
    [Input('data-period-selector', 'value'),
     Input('ticker-dropdown', 'value')]
)
def update_date_dropdowns(selected_period, selected_ticker):
    if not selected_ticker:
        return [], None, [], None, None, None, None, None
        
    data = load_and_process_data(selected_period, selected_ticker)
    
    if data.empty:
        return [], None, [], None, None, None, None, None
        
    # Get min and max dates from data
    min_date = data['Datetime'].min()
    max_date = data['Datetime'].max()
    
    years = list(range(min_date.year, max_date.year + 1))
    year_options = [{'label': str(year), 'value': year} for year in years]
    
    # Set default values
    default_start_year = min_date.year
    default_end_year = max_date.year
    default_start_month = min_date.month
    default_end_month = max_date.month
    default_start_day = min_date.day
    default_end_day = max_date.day
    
    return (year_options, default_start_year,
            year_options, default_end_year,
            default_start_month, default_end_month,
            default_start_day, default_end_day)




def save_signals_to_db(title, ticker, signals_data, selected_technicals):
    if not title or not signals_data or not ticker:
        print("Invalid data: Title, ticker, and signals data are required")
        return False

    try:
        connection = create_db_connection()
        if not connection:
            print("Failed to establish database connection")
            return False

        cursor = connection.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS data_signal_dashboard2 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            title VARCHAR(255),
            ticker VARCHAR(255),
            analysis JSON,
            signal_data JSON,
            INDEX idx_timestamp (timestamp),
            INDEX idx_title (title),
            INDEX idx_ticker (ticker)
        )
        """
        cursor.execute(create_table_query)
        
        # Clean and validate signals data
        cleaned_signals = {
            'ticker': ticker,
            'total_signals': signals_data['total_signals'],
            'signal_details': {}
        }
        
        # Clean up signal details - include analysis name in each entry
        for tech, details in signals_data['signal_details'].items():
            if isinstance(details, dict):
                cleaned_signals['signal_details'][tech] = {
                    signal_type: [
                        {**signal_entry, 'analysis': tech}  # Add 'analysis' field
                        for signal_entry in signals
                    ]
                    for signal_type, signals in details.items()
                    if signal_type in ['Buy', 'Sell', 'Hold'] and isinstance(signals, list)
                }
        
        # Convert selected technicals to JSON format
        analysis_data = json.dumps(selected_technicals)

        insert_query = """
        INSERT INTO data_signal_dashboard2 (timestamp, title, ticker, analysis, signal_data)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            datetime.now(),
            title[:255],
            ticker[:255],
            analysis_data,
            json.dumps(cleaned_signals)
        ))
        
        connection.commit()
        print(f"Data saved successfully for ticker {ticker}!")
        return True

    except pymysql.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()





@app.callback(
    [Output('charts-container', 'children'),
     Output('stock-table', 'data'), 
     Output('stock-table', 'columns'),
     Output('interactive-signal-table', 'children'),
     Output('save-status', 'children')],
    [Input('ticker-dropdown', 'value'),
     Input('technical-checklist', 'value'),
     Input('start-year-dropdown', 'value'),
     Input('start-month-dropdown', 'value'),
     Input('start-day-dropdown', 'value'),
     Input('end-year-dropdown', 'value'),
     Input('end-month-dropdown', 'value'),
     Input('end-day-dropdown', 'value'),
     Input('data-period-selector', 'value'),
     Input('realtime-interval-selector', 'value'),
     Input('save-button', 'n_clicks'),
     Input('bb-period', 'value'),
     Input('bb-std', 'value'),
     Input('bb-signal-mode', 'value'), 
     Input('ma-short', 'value'),
     Input('ma-long', 'value'),
     Input('ma-medium', 'value'), 
     Input('ma-lines-count', 'value'),  
     Input('rsi-period', 'value'),
     Input('rsi-overbought', 'value'),
     Input('rsi-oversold', 'value'),
     Input('macd-fast', 'value'),
     Input('macd-slow', 'value'),
     Input('macd-signal', 'value'),
     Input('adx-period', 'value'),
     Input('adx-threshold', 'value'),
     Input('volume-period', 'value'),
     Input('fibonacci-lookback', 'value'),
     Input('candlestick-confidence', 'value'),
     Input('candlestick-categories', 'value'),
     Input('show-explanations', 'value')],
    [State('save-title', 'value')]
)
def update_analysis(ticker, selected_technicals, 
                   start_year, start_month, start_day,
                   end_year, end_month, end_day, 
                   selected_period, selected_interval, n_clicks,
                   bb_window, bb_std, bb_signal_mode, 
                   ma_short, ma_long, ma_medium, ma_lines_count, 
                   rsi_period, rsi_ob, rsi_os,
                   macd_fast, macd_slow, macd_signal,
                   adx_period, adx_threshold,
                   volume_period,
                   fibonacci_lookback, candlestick_confidence,
                   candlestick_categories, show_explanations, 
                   save_title):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not ticker or None in [start_year, start_month, start_day,
                            end_year, end_month, end_day]:
        return [], [], [], None, ""
        
        
    try:
        # Load data only for selected ticker and period
        # filtered_df = load_and_process_data(selected_period, ticker)

        filtered_df = load_and_process_data(
            period=selected_period, 
            ticker=ticker,
            interval=selected_interval if selected_period == 'realtime' else '1min'
        )
        
        if filtered_df.empty:
            return [], [], [], None, "No data available for selected ticker"


        # Calculate indicators with custom parameters
        if 'Bollinger_Signal' in selected_technicals:
            filtered_df = calculate_bollinger_bands(
                filtered_df, 
                window=bb_window, 
                num_std=bb_std, 
                signal_mode=bb_signal_mode  
            )
        
        if 'MA_Signal' in selected_technicals:
            filtered_df = calculate_ma(
                filtered_df, 
                short_period=ma_short, 
                long_period=ma_long, 
                medium_period=ma_medium, 
                ma_lines=ma_lines_count  
            )
        
        if 'RSI_Signal' in selected_technicals:
            filtered_df = calculate_rsi(filtered_df, period=rsi_period, ob_level=rsi_ob, os_level=rsi_os)
        
        if 'MACD_Signal' in selected_technicals:
            filtered_df = calculate_macd(filtered_df, short_window=macd_fast, long_window=macd_slow, signal_window=macd_signal)
        
        if 'ADX_Signal' in selected_technicals:
            filtered_df = calculate_adx(filtered_df, period=adx_period, threshold=adx_threshold)
        
        if 'Volume_Signal' in selected_technicals:
            filtered_df = calculate_volume(filtered_df, ma_window=volume_period)
            # Safeguard for volume_reason column
            if 'volume_reason' not in filtered_df.columns:
                print("DEBUG: 'volume_reason' column not found after calculate_volume. Adding default.")
                filtered_df['volume_reason'] = "N/A"
        
        if 'Fibonacci_Signal' in selected_technicals:
            filtered_df = calculate_fibonacci_retracement(filtered_df, lookback=fibonacci_lookback)

        if 'Candlestick_Signal' in selected_technicals:
            # Apply candlestick pattern detection
            filtered_df = detect_candlestick_patterns(filtered_df)
            print("DEBUG: After detect_candlestick_patterns, head of relevant columns:")
            if 'detected_patterns' in filtered_df.columns and 'Candlestick_Signal' in filtered_df.columns:
                print(filtered_df[['Datetime', 'detected_patterns', 'Candlestick_Signal']].head())
            elif 'detected_patterns' in filtered_df.columns:
                print(filtered_df[['Datetime', 'detected_patterns']].head())
                print("DEBUG: 'Candlestick_Signal' column missing after pattern detection.")
            else:
                print("DEBUG: 'detected_patterns' column missing after pattern detection.")

            # Apply confidence threshold
            confidence_threshold_val = candlestick_confidence if 'candlestick_confidence' in locals() and candlestick_confidence is not None else 50
            if 'candlestick_confidence' in filtered_df.columns:
                mask = filtered_df['candlestick_confidence'] < confidence_threshold_val
                filtered_df.loc[mask, 'Candlestick_Signal'] = 'Hold'
            
            # Filter by pattern categories if specified
            if 'candlestick_categories' in locals() and candlestick_categories:
                # This is a simplified approach - you would need to define which patterns belong to which categories
                # and implement the filtering based on that
                pass
        
        # Convert selected dates to datetime
        start_date = pd.Timestamp(
            year=start_year, 
            month=start_month, 
            day=start_day
        ).replace(hour=0, minute=0, second=0)
        
        end_date = pd.Timestamp(
            year=end_year,
            month=end_month,
            day=end_day
        ).replace(hour=23, minute=59, second=59)
        
        # Filter data by date range
        filtered_df = filtered_df[
            (filtered_df['Datetime'] >= start_date) &
            (filtered_df['Datetime'] <= end_date)
        ].copy()
        
        if filtered_df.empty:
            return [], [], [], None, "No data available for selected date range"


        # def classify_fibonacci_trend(row, lookback=60):
        #     # Only classify if Fibonacci_Signal is present
        #     if 'Fibonacci_Signal' not in row or pd.isna(row['Fibonacci_Signal']):
        #         return 'Sideways'
        #     # Use the last available Fibonacci levels
        #     if hasattr(filtered_df, 'attrs') and 'fibonacci_levels' in filtered_df.attrs:
        #         fib = filtered_df.attrs['fibonacci_levels']
        #         levels = fib['levels']
        #         last_close = row['Close']
        #         if last_close > levels[3]:  # Above 0.618
        #             return 'Uptrend'
        #         elif last_close < levels[1]:  # Below 0.382
        #             return 'Downtrend'
        #         else:
        #             return 'Sideways'
        #     return 'Sideways'

        # # Add Fibonacci trend column if selected
        # if 'Fibonacci_Signal' in selected_technicals:
        #     filtered_df['Fibonacci_Trend'] = filtered_df.apply(lambda row: classify_fibonacci_trend(row, lookback=fibonacci_lookback), axis=1)
        # else:
        #     filtered_df['Fibonacci_Trend'] = 'Sideways'





        # Rest of your existing code remains the same...
        def calculate_signal(row):
            # Mapping indikator ke nama kolom
            signal_mapping = {
                'Bollinger_Signal': 'Signal',
                'RSI_Signal': 'RsiSignal',
                'MACD_Signal': 'MacdSignal',
                'MA_Signal': 'MaSignal',
                'ADX_Signal': 'AdxSignal',
                'Volume_Signal': 'VolumeSignal',
                'Fibonacci_Signal': 'Fibonacci_Signal',
            #   'Candlestick_Signal': 'Candlestick_Signal'
            }

            # Ambil sinyal dari semua indikator teknikal kecuali candlestick
            signals = []
            for tech in selected_technicals:
                if tech in signal_mapping and signal_mapping[tech] in row:
                    signal = row[signal_mapping[tech]]
                    if pd.notna(signal) and tech != 'Candlestick_Signal':
                        signals.append(signal)

            # Hitung jumlah sinyal
            buy_count = signals.count('Buy')
            sell_count = signals.count('Sell')
            hold_count = signals.count('Hold')

            # Tentukan sinyal dasar berdasarkan mayoritas
            if buy_count > sell_count and buy_count > hold_count:
                base_signal = 'Buy'
            elif sell_count > buy_count and sell_count > hold_count:
                base_signal = 'Sell'
            else:
                base_signal = 'Hold'

            # Dapatkan konfirmasi volume dan trend
            volume_signal = row.get('VolumeSignal', None)
            high_volume = volume_signal == 'High Volume'
            candlestick_trend = row.get('candlestick_trend', 'Neutral')

            # Logic konfirmasi baru
            final_signal = base_signal
            volume_reason = ""
            candlestick_reason = ""

            # Jika volume rendah dan bukan sinyal Hold, ubah ke Hold
            if 'Volume_Signal' in selected_technicals:
                if not high_volume and base_signal != 'Hold':
                    final_signal = 'Hold'
                    volume_reason = f"Sinyal {base_signal} diabaikan karena volume rendah"
                elif high_volume:
                    volume_reason = f"Volume tinggi mendukung sinyal {base_signal}"
                else:
                    volume_reason = "Volume rendah"
            else:
                volume_reason = "Volume tidak dipertimbangkan"

            # Konfirmasi trend candlestick
            if 'Candlestick_Signal' in selected_technicals and candlestick_trend != 'Neutral':
                if base_signal == 'Buy' and candlestick_trend == 'Downtrend':
                    final_signal = 'Hold'
                    candlestick_reason = "Sinyal Buy diabaikan karena trend menurun"
                elif base_signal == 'Sell' and candlestick_trend == 'Uptrend':
                    final_signal = 'Hold'
                    candlestick_reason = "Sinyal Sell diabaikan karena trend naik"
                elif base_signal == 'Buy' and candlestick_trend == 'Uptrend':
                    candlestick_reason = "Sinyal Buy dikonfirmasi oleh trend naik"
                elif base_signal == 'Sell' and candlestick_trend == 'Downtrend':
                    candlestick_reason = "Sinyal Sell dikonfirmasi oleh trend turun"
                elif candlestick_trend == 'Sideways':
                    candlestick_reason = f"Sinyal {base_signal} di pasar sideways"
                else:
                    candlestick_reason = f"Trend {candlestick_trend} netral terhadap sinyal {base_signal}"
            else:
                candlestick_reason = "Candlestick trend tidak tersedia atau tidak dipilih"

            # Jika sudah diubah ke Hold karena volume, update reasoning
            if final_signal == 'Hold' and base_signal != 'Hold':
                if not high_volume and (base_signal == 'Buy' and candlestick_trend == 'Downtrend'):
                    candlestick_reason = f"Sinyal {base_signal} diabaikan karena trend bertentangan dan volume rendah"
                elif not high_volume and (base_signal == 'Sell' and candlestick_trend == 'Uptrend'):
                    candlestick_reason = f"Sinyal {base_signal} diabaikan karena trend bertentangan dan volume rendah"

            # Gabungkan alasan
            combined_reason = f"{volume_reason}. {candlestick_reason}".strip()

            return final_signal, combined_reason, volume_reason, candlestick_reason

        # Terapkan fungsi ke DataFrame
        if selected_technicals:
            filtered_df['signal_result'] = filtered_df.apply(calculate_signal, axis=1)
            filtered_df['signal_akhir'] = filtered_df['signal_result'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
            filtered_df['signal_reason'] = filtered_df['signal_result'].apply(lambda x: x[1] if isinstance(x, tuple) else "")
        else:
            filtered_df['signal_akhir'] = 'Hold'
            filtered_df['signal_reason'] = ""

        # Hapus kolom sementara
        if 'signal_result' in filtered_df.columns:
            filtered_df.drop(columns=['signal_result'], inplace=True)



        # Tambahkan kolom Signal dari Bollinger Bands jika ada
        if 'Bollinger_Signal' in selected_technicals:
            filtered_df['Bollinger_Signal'] = filtered_df['Signal']
        
        if 'RSI_Signal' in selected_technicals:
            filtered_df['RSI_Signal'] = filtered_df['RsiSignal']
        
        if 'MACD_Signal' in selected_technicals:
            filtered_df['MACD_Signal'] = filtered_df['MacdSignal']
        
        if 'MA_Signal' in selected_technicals:
            filtered_df['MA_Signal'] = filtered_df['MaSignal']
        
        if 'ADX_Signal' in selected_technicals:
            filtered_df['ADX_Signal'] = filtered_df['AdxSignal']
        
        if 'Volume_Signal' in selected_technicals:
            filtered_df['Volume_Signal'] = filtered_df['VolumeSignal']
        
        if 'Fibonacci_Signal' in selected_technicals:
            filtered_df['Fibonacci_Signal'] = filtered_df['Fibonacci_Signal']
        
        if 'Candlestick_Signal' in selected_technicals:
            filtered_df['Candlestick_Signal'] = filtered_df['Candlestick_Signal']
        

        

        # Membuat grafik untuk setiap indikator teknikal
        charts = []
        for tech in selected_technicals:
            if tech == 'Bollinger_Signal' and 'Upper Band' in filtered_df.columns and 'Lower Band' in filtered_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Upper Band'], mode='lines', name='Upper Band'))
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Middle Band'], mode='lines', name='Middle Band'))
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Lower Band'], mode='lines', name='Lower Band'))
                fig.update_layout(title="Bollinger Bands", xaxis_title="Tanggal", yaxis_title="Harga")
                charts.append(dcc.Graph(figure=fig))
            
            elif tech == 'MA_Signal' and 'short_MA' in filtered_df.columns and 'long_MA' in filtered_df.columns:
                fig = go.Figure()
                
                # Tambahkan garis Close
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['Close'], 
                    mode='lines', 
                    name='Close',
                    line=dict(dash='dash', color='grey')
                ))
                
                # Tambahkan garis MA Fast
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['short_MA'], 
                    mode='lines', 
                    name=f'Fast MA ({ma_short})', 
                    line=dict(dash='dash', color='blue')
                ))
                
                # Tambahkan garis MA Medium jika mode 3 lines dan kolomnya ada
                if ma_lines_count == '3' and 'medium_MA' in filtered_df.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['medium_MA'], 
                        mode='lines', 
                        name=f'Medium MA ({ma_medium})', 
                        line=dict(dash='dash', color='orange') # Anda bisa mengganti warna jika perlu
                    ))
                
                # Tambahkan garis MA Slow
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['long_MA'], 
                    mode='lines', 
                    name=f'Slow MA ({ma_long})', 
                    line=dict(dash='dash', color='red')
                ))
                
                # Update layout judul secara dinamis
                title_text = f"Moving Average ({ma_short}, {ma_long})"
                if ma_lines_count == '3' and 'medium_MA' in filtered_df.columns:
                    title_text = f"Moving Average ({ma_short}, {ma_medium}, {ma_long})"
                elif ma_lines_count == '3': # Jika 3 lines dipilih tapi medium_MA tidak ada
                     title_text = f"Moving Average (3 Lines - Medium MA not available)"
                
                fig.update_layout(
                    title=title_text, 
                    xaxis_title="Tanggal", 
                    yaxis_title="Nilai",
                    legend=dict(
                        x=0.01, # Posisi legend di kiri atas dalam area plot
                        y=0.99,
                        traceorder="normal",
                        bgcolor='rgba(255, 255, 255, 0.5)',
                        bordercolor='rgba(0, 0, 0, 0.1)'
                    )
                )
                
                charts.append(dcc.Graph(figure=fig))




            elif tech == 'RSI_Signal' and 'RSI' in filtered_df.columns:
                fig = go.Figure()
                
                # Tambahkan garis RSI
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['RSI'], 
                    mode='lines', 
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Tambahkan garis overbought berdasarkan parameter rsi_ob
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=[rsi_ob]*len(filtered_df), 
                    mode='lines', 
                    name=f'Overbought ({rsi_ob})', 
                    line=dict(
                        dash='dash',
                        color='red',
                        width=1
                    )
                ))
                
                # Tambahkan garis oversold berdasarkan parameter rsi_os
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=[rsi_os]*len(filtered_df), 
                    mode='lines', 
                    name=f'Oversold ({rsi_os})', 
                    line=dict(
                        dash='dash',
                        color='green',
                        width=1
                    )
                ))
                
                # Update layout dengan informasi periode RSI
                fig.update_layout(
                    title=f"RSI ({rsi_period} periods)",
                    xaxis_title="Tanggal",
                    yaxis_title="RSI",
                    yaxis=dict(
                        range=[0, 100],  # Tetapkan range RSI dari 0-100
                        gridcolor='lightgrey',
                        zerolinecolor='black',
                        zerolinewidth=1
                    ),
                    plot_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01
                    ),
                    margin=dict(r=100)  # Memberikan ruang untuk legend di sebelah kanan
                )
                
                # Tambahkan area shading untuk zona overbought dan oversold
                fig.add_hrect(
                    y0=rsi_ob, 
                    y1=100, 
                    fillcolor="red", 
                    opacity=0.1, 
                    layer="below", 
                    line_width=0,
                    name="Overbought Zone"
                )
                
                fig.add_hrect(
                    y0=0, 
                    y1=rsi_os, 
                    fillcolor="green", 
                    opacity=0.1, 
                    layer="below", 
                    line_width=0,
                    name="Oversold Zone"
                )

                charts.append(dcc.Graph(figure=fig))



            # Mengecek apakah kolom yang diperlukan ada di filtered_df
            elif tech == 'MACD_Signal' and 'MACD' in filtered_df.columns and 'Signal_Line' in filtered_df.columns and 'MACD_Hist' in filtered_df.columns:
                fig = go.Figure()

                # Tambahkan histogram MACD terlebih dahulu agar menjadi background
                fig.add_trace(go.Bar(
                    x=filtered_df['Datetime'],
                    y=filtered_df['MACD_Hist'],
                    name='MACD Histogram',
                    marker=dict(
                        color=filtered_df['MACD_Hist'].apply(
                            lambda x: 'rgba(0, 255, 0, 0.3)' if x > 0 else 'rgba(255, 0, 0, 0.3)'  # Kurangi opacity
                        )
                    ),
                    width=24*60*60*1000*0.8  # 80% dari interval harian
                ))

                # Tambahkan garis MACD
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['MACD'], 
                    mode='lines', 
                    name='MACD',
                    line=dict(color='#FF9900', width=2)
                ))

                # Tambahkan garis Signal Line
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=filtered_df['Signal_Line'], 
                    mode='lines', 
                    name='Signal Line',
                    line=dict(color='#0066FF', width=2)
                ))

                # Tambahkan garis nol
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'], 
                    y=[0] * len(filtered_df), 
                    mode='lines', 
                    name='Zero Line', 
                    line=dict(color='black', dash='dash', width=1),
                    showlegend=False
                ))

                # Update layout
                fig.update_layout(
                    title="MACD (Moving Average Convergence Divergence)",
                    xaxis_title="Tanggal",
                    yaxis_title="Nilai",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=600,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01
                    ),
                    yaxis=dict(
                        gridcolor='lightgrey',
                        zerolinecolor='black',
                        zerolinewidth=1,
                        range=[
                            min(min(filtered_df['MACD']), 
                                min(filtered_df['Signal_Line']), 
                                min(filtered_df['MACD_Hist'])) * 1.1,
                            max(max(filtered_df['MACD']), 
                                max(filtered_df['Signal_Line']), 
                                max(filtered_df['MACD_Hist'])) * 1.1
                        ]
                    ),
                    xaxis=dict(
                        gridcolor='lightgrey'
                    ),
                    margin=dict(r=100)  # Memberikan ruang untuk legend di sebelah kanan
                )

                # Menambahkan grafik ke dalam charts
                charts.append(dcc.Graph(figure=fig))



            elif tech == 'ADX_Signal' and 'ADX' in filtered_df.columns and '+DI' in filtered_df.columns and '-DI' in filtered_df.columns:
                # Membuat subplots dengan dua baris
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,  # Sumbu X berbagi
                    vertical_spacing=0.05,  # Jarak antar grafik
                    subplot_titles=("Harga Close", "Indikator ADX (+DI, -DI, ADX)")
                )

                # Bagian 1: Grafik Harga Close
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['Close'], 
                        mode='lines', 
                        name='Harga Close', 
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1  # Lokasi di subplot
                )

                # Bagian 2: Grafik Indikator ADX
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['ADX'], 
                        mode='lines', 
                        name='ADX', 
                        line=dict(color='orange', width=2)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['+DI'], 
                        mode='lines', 
                        name='+DI', 
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['-DI'], 
                        mode='lines', 
                        name='-DI', 
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    title="ADX (Average Directional Index)",
                    xaxis=dict(title=None),  # Hapus label 'Tanggal' di subplot pertama
                    yaxis=dict(title="Nilai", showgrid=True),
                    yaxis2=dict(title="Nilai ADX"),  # Sumbu Y untuk subplot kedua
                    showlegend=True,
                    legend=dict(
                        orientation="v",  # Legend
                        x=1.05,
                        y=1
                    ),
                    height=700,  # Tinggi total grafik
                    margin=dict(t=50, b=50, l=50, r=100)  # Tambahkan ruang untuk legenda di kanan
                )

                # Tambahkan ke daftar grafik
                charts.append(dcc.Graph(figure=fig))


            elif tech == 'Volume_Signal' and 'VMA' in filtered_df.columns and 'Volume' in filtered_df.columns:
                # Membuat figure
                fig = go.Figure()

                # Menambahkan bar chart untuk Volume asli
                fig.add_trace(
                    go.Bar(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['Volume'], 
                        name='Volume', 
                        marker_color='blue'
                    )
                )

                # Menambahkan line chart untuk VMA
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'], 
                        y=filtered_df['VMA'], 
                        mode='lines', 
                        name='VMA', 
                        line=dict(color='orange', width=2)
                    )
                )

                # Mengatur layout
                fig.update_layout(
                    title="Volume",
                    xaxis_title="Tanggal",
                    yaxis_title="Volume",
                    legend=dict(
                        orientation="v",
                        x=1.05,
                        y=0.5,
                        xanchor="left", 
                        yanchor="middle" 
                    ),
                    height=500,
                    margin=dict(t=50, b=50, l=50, r=100),
                    barmode='group', 
                )

                # Menambahkan grafik ke list charts
                charts.append(dcc.Graph(figure=fig))
            elif tech == 'Fibonacci_Signal' and 'Fibonacci_Signal' in filtered_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close',
                    line=dict(color='black', width=2)
                ))

                # Extract all levels for each row
                levels_matrix = filtered_df.apply(
                    lambda row: row.get('fib_levels', [None]*5) if 'fib_levels' in row and row['fib_levels'] is not None else [None]*5,
                    axis=1, result_type='expand'
                )
                retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                colors = ['#FFB347', '#FFD700', '#90EE90', '#87CEEB', '#FF69B4']

                for i, level in enumerate(retracement_levels):
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=levels_matrix[i],
                        mode='lines',
                        name=f'Fibo {level*100:.1f}%',
                        line=dict(color=colors[i], dash='dash')
                    ))

                # Optionally, plot high/low as step lines too
                if 'fib_high' in filtered_df.columns and 'fib_low' in filtered_df.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['fib_high'],
                        mode='lines',
                        name='High',
                        line=dict(color='red', dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['fib_low'],
                        mode='lines',
                        name='Low',
                        line=dict(color='green', dash='dot')
                    ))

                fig.update_layout(
                    title="Fibonacci Retracement (Stepwise)",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500
                )
                charts.append(dcc.Graph(figure=fig))
        

            #candlestick pattern
            elif tech == 'Candlestick_Signal' and 'Candlestick_Signal' in filtered_df.columns:
                fig = go.Figure()
                
                # Create candlestick chart
                fig.add_trace(go.Candlestick(
                    x=filtered_df['Datetime'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name='Price'
                ))
                
                # Check if the column exists before trying to filter on it
                if 'Candlestick_Signal' in filtered_df.columns:
                    # Add buy/sell signals based on candlestick patterns
                    buy_mask = filtered_df['Candlestick_Signal'] == 'Buy'
                    sell_mask = filtered_df['Candlestick_Signal'] == 'Sell'
                    
                    if buy_mask.any():
                        fig.add_trace(go.Scatter(
                            x=filtered_df.loc[buy_mask, 'Datetime'],
                            y=filtered_df.loc[buy_mask, 'High'] + (filtered_df.loc[buy_mask, 'High'] * 0.01),  # Slightly above high
                            mode='markers',
                            name='Bullish Pattern',
                            marker=dict(color='green', symbol='triangle-up', size=10)
                        ))
                    
                    if sell_mask.any():
                        fig.add_trace(go.Scatter(
                            x=filtered_df.loc[sell_mask, 'Datetime'],
                            y=filtered_df.loc[sell_mask, 'Low'] - (filtered_df.loc[sell_mask, 'Low'] * 0.01),  # Slightly below low
                            mode='markers',
                            name='Bearish Pattern',
                            marker=dict(color='red', symbol='triangle-down', size=10)
                        ))
                    
                    # Add confidence as text annotations for selected points
                    if 'candlestick_confidence' in filtered_df.columns:
                        for i, row in filtered_df[buy_mask | sell_mask].iterrows():
                            if i % 5 == 0:  # Only show every 5th point to avoid clutter
                                fig.add_annotation(
                                    x=row['Datetime'],
                                    y=row['Close'],
                                    text=f"{int(row['candlestick_confidence'])}%",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor='black' if row['Candlestick_Signal'] == 'Buy' else 'red'
                                )
                
                # Add pattern count information to the chart
                if 'bullish_patterns' in filtered_df.columns and 'bearish_patterns' in filtered_df.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['bullish_patterns'],
                        mode='lines',
                        name='Bullish Patterns',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1),
                        yaxis='y2'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['bearish_patterns'],
                        mode='lines',
                        name='Bearish Patterns',
                        line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
                        yaxis='y2'
                    ))
                
                # Update layout with a secondary y-axis for pattern counts
                fig.update_layout(
                    title="Candlestick Pattern Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(
                        title="Pattern Count",
                        overlaying="y",
                        side="right",
                        range=[0, 5]  # Adjust based on your typical pattern counts
                    ),
                    height=600
                )
                
                charts.append(dcc.Graph(figure=fig))
                
            #     # Add an explanatory text box if show_explanations is enabled
            #     if show_explanations == 'yes':
            #         try:
            #             candlestick_explanations = get_candlestick_explanations()
            #             explanatory_text = html.Div([
            #                 html.H3("Understanding Candlestick Patterns", className="font-bold text-lg mb-2"),
            #                 html.P(candlestick_explanations['overview'], className="mb-3"),
                            
            #                 # Add tabs for bullish and bearish patterns
            #                 dcc.Tabs([
            #                     dcc.Tab(label="Bullish Patterns", children=[
            #                         html.Div([
            #                             html.Div([
            #                                 html.H4(pattern['name'], className="font-semibold text-green-700"),
            #                                 html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                                 html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                                 html.Div([
            #                                     html.Img(
            #                                         src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                                         alt=pattern['name'],
            #                                         className="w-24 h-24 object-contain"
            #                                     ) if pattern.get('image', True) else None,
            #                                 ], className="mt-2")
            #                             ], className="mb-4 p-3 border border-green-200 rounded bg-green-50")
            #                             for pattern in candlestick_explanations['patterns']['bullish']
            #                         ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #                     ]),
            #                     dcc.Tab(label="Bearish Patterns", children=[
            #                         html.Div([
            #                             html.Div([
            #                                 html.H4(pattern['name'], className="font-semibold text-red-700"),
            #                                 html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                                 html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                                 html.Div([
            #                                     html.Img(
            #                                         src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                                         alt=pattern['name'],
            #                                         className="w-24 h-24 object-contain"
            #                                     ) if pattern.get('image', True) else None,
            #                                 ], className="mt-2")
            #                             ], className="mb-4 p-3 border border-red-200 rounded bg-red-50")
            #                             for pattern in candlestick_explanations['patterns']['bearish']
            #                         ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #                     ]),
            #                     dcc.Tab(label="Neutral Patterns", children=[
            #                         html.Div([
            #                             html.Div([
            #                                 html.H4(pattern['name'], className="font-semibold text-blue-700"),
            #                                 html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                                 html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                                 html.Div([
            #                                     html.Img(
            #                                         src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                                         alt=pattern['name'],
            #                                         className="w-24 h-24 object-contain"
            #                                     ) if pattern.get('image', True) else None,
            #                                 ], className="mt-2")
            #                             ], className="mb-4 p-3 border border-blue-200 rounded bg-blue-50")
            #                             for pattern in candlestick_explanations['patterns']['neutral']
            #                         ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #                     ]),
            #                 ]),
                            
            #                 html.Div([
            #                     html.H4("Signal Interpretation", className="font-semibold mb-1 mt-4"),
            #                     html.Ul([
            #                         html.Li(f"Buy: {candlestick_explanations['signals']['Buy']}", className="ml-5 list-disc text-green-600"),
            #                         html.Li(f"Sell: {candlestick_explanations['signals']['Sell']}", className="ml-5 list-disc text-red-600"),
            #                         html.Li(f"Hold: {candlestick_explanations['signals']['Hold']}", className="ml-5 list-disc text-blue-600")
            #                     ]),
            #                     html.P(candlestick_explanations['usage'], className="mt-3 text-sm italic")
            #                 ], className="mt-4")
            #             ], className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200")
                        
            #             charts.append(explanatory_text)
            #         except Exception as e:
            #             print(f"Error displaying candlestick explanations: {str(e)}")
            #             # Add a fallback explanation if the detailed one fails
            #             charts.append(html.Div(f"Candlestick patterns help predict potential price movements based on historical patterns. Error loading details: {str(e)}",
            #                                 className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200"))
              
            #   # Add an explanatory text box
                  
            # candlestick_explanations = get_candlestick_explanations()
            # explanatory_text = html.Div([
            #     html.H3("Understanding Candlestick Patterns", className="font-bold text-lg mb-2"),
            #     html.P(candlestick_explanations['overview'], className="mb-3"),
                
            #     # Add tabs for bullish and bearish patterns
            #     dcc.Tabs([
            #         dcc.Tab(label="Bullish Patterns", children=[
            #             html.Div([
            #                 html.Div([
            #                     html.H4(pattern['name'], className="font-semibold text-green-700"),
            #                     html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                     html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                     html.Div([
            #                         html.Img(
            #                             src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                             alt=pattern['name'],
            #                             className="w-24 h-24 object-contain"
            #                         ) if pattern.get('image', True) else None,
            #                     ], className="mt-2")
            #                 ], className="mb-4 p-3 border border-green-200 rounded bg-green-50")
            #                 for pattern in candlestick_explanations['patterns']['bullish']
            #             ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #         ]),
            #         dcc.Tab(label="Bearish Patterns", children=[
            #             html.Div([
            #                 html.Div([
            #                     html.H4(pattern['name'], className="font-semibold text-red-700"),
            #                     html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                     html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                     html.Div([
            #                         html.Img(
            #                             src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                             alt=pattern['name'],
            #                             className="w-24 h-24 object-contain"
            #                         ) if pattern.get('image', True) else None,
            #                     ], className="mt-2")
            #                 ], className="mb-4 p-3 border border-red-200 rounded bg-red-50")
            #                 for pattern in candlestick_explanations['patterns']['bearish']
            #             ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #         ]),
            #         dcc.Tab(label="Neutral Patterns", children=[
            #             html.Div([
            #                 html.Div([
            #                     html.H4(pattern['name'], className="font-semibold text-blue-700"),
            #                     html.P(f"Strength: {pattern['strength']}", className="text-sm italic"),
            #                     html.P(get_pattern_description(pattern['code']), className="mt-1"),
            #                     html.Div([
            #                         html.Img(
            #                             src=f"assets/candlestick_patterns/{pattern['code'].lower()}.png" if pattern['code'] else "",
            #                             alt=pattern['name'],
            #                             className="w-24 h-24 object-contain"
            #                         ) if pattern.get('image', True) else None,
            #                     ], className="mt-2")
            #                 ], className="mb-4 p-3 border border-blue-200 rounded bg-blue-50")
            #                 for pattern in candlestick_explanations['patterns']['neutral']
            #             ], className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3")
            #         ]),
            #     ]),
                
            #     html.Div([
            #         html.H4("Signal Interpretation", className="font-semibold mb-1 mt-4"),
            #         html.Ul([
            #             html.Li(f"Buy: {candlestick_explanations['signals']['Buy']}", className="ml-5 list-disc text-green-600"),
            #             html.Li(f"Sell: {candlestick_explanations['signals']['Sell']}", className="ml-5 list-disc text-red-600"),
            #             html.Li(f"Hold: {candlestick_explanations['signals']['Hold']}", className="ml-5 list-disc text-blue-600")
            #         ]),
            #         html.P(candlestick_explanations['usage'], className="mt-3 text-sm italic")
            #     ], className="mt-4")
            # ], className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200")
            
            # if show_explanations == 'yes':
            #     charts.append(explanatory_text)



        # Menentukan kolom tabel berdasarkan pilihan analisa teknikal
        base_columns = ['Datetime', 'Close']
        technical_columns = []

        for tech in selected_technicals:
            if tech == 'Bollinger_Signal':
                technical_columns.extend(['Upper Band', 'Lower Band','Middle Band', tech])
            elif tech == 'RSI_Signal':
                technical_columns.extend(['RSI', tech])
            elif tech == 'MACD_Signal':
                technical_columns.extend(['MACD', 'Signal_Line','MACD_Hist', tech])
            elif tech == 'MA_Signal':
                technical_columns.extend(['short_MA', 'long_MA', 'medium_MA', tech])
            elif tech == 'ADX_Signal':
                technical_columns.extend(['ADX', '+DI', '-DI', tech])
            elif tech == 'Volume_Signal':
                technical_columns.extend(['Volume', 'VMA', 'VolumeSignal', tech])
            elif tech == 'Fibonacci_Signal':
                technical_columns.extend(['Fibonacci_Signal', 'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786'])
            elif tech == 'Candlestick_Signal':
                technical_columns.extend(['detected_patterns', 'candlestick_trend', 'candlestick_reason'])

        # Only add Volume to base columns if Volume_Signal is selected
        if 'Volume_Signal' in selected_technicals:
            base_columns.append('Volume')

        displayed_columns = base_columns + technical_columns + ['signal_akhir']
        displayed_columns = [col for col in displayed_columns if col in filtered_df.columns]

        # Signal summary calculation
        signal_summary = analyze_signals(filtered_df, selected_technicals)
    
        # Create interactive table
        # Create interactive signal table - UPDATED VERSION
        table_content = html.Div([
            # Summary Header
            html.Div([
                html.H2("Signal Analysis Summary", className="text-xl font-bold mb-4"),
                html.Div([
                    html.Div([
                        html.Span("Buy Signals: ", className="font-semibold"),
                        html.Span(
                            str(signal_summary['total_signals']['Buy']),
                            className="text-green-600 ml-1"
                        )
                    ], className="mr-6"),
                    html.Div([
                        html.Span("Sell Signals: ", className="font-semibold"),
                        html.Span(
                            str(signal_summary['total_signals']['Sell']),
                            className="text-red-600 ml-1"
                        )
                    ], className="mr-6"),
                    html.Div([
                        html.Span("Hold Signals: ", className="font-semibold"),
                        html.Span(
                            str(signal_summary['total_signals']['Hold']),
                            className="text-blue-600 ml-1"
                        )
                    ])
                ], className="flex mb-4"),
            ], className="border-b pb-4"),
            
            # Technical Analysis Details - FIXED STRUCTURE
            html.Div([
                # Create individual sections for each technical indicator
                *[
                    html.Div([
                        html.H3(tech.replace('_', ' '), className="text-lg font-semibold mb-3"),
                        
                        # Special handling for Candlestick_Signal - Show trends instead of signal counts
                        html.Div([
                            html.Div([
                                html.Span("Candlestick Trends: ", className="font-semibold"),
                                html.Span("Uptrend: ", className="text-green-600 font-medium ml-2"),
                                html.Span(str(signal_summary.get('candlestick_trends', {}).get('Uptrend', 0)), className="mr-3"),
                                html.Span("Downtrend: ", className="text-red-600 font-medium"),
                                html.Span(str(signal_summary.get('candlestick_trends', {}).get('Downtrend', 0)), className="mr-3"),
                                html.Span("Sideways: ", className="text-blue-600 font-medium"),
                                html.Span(str(signal_summary.get('candlestick_trends', {}).get('Sideways', 0)))
                            ], className="mb-2")
                        ], className="mb-3") if tech == 'Candlestick_Signal' else 
                        
                        # Standard signal count for other indicators
                        html.Div([
                            html.Div([
                                html.Span(f"{tech.replace('_', ' ')} Buy Signals: ", className="font-semibold"),
                                html.Span(
                                    str(len(signal_summary['signal_details'][tech]['Buy'])),
                                    className="text-green-600 ml-1"
                                )
                            ], className="mr-6"),
                            html.Div([
                                html.Span(f"{tech.replace('_', ' ')} Sell Signals: ", className="font-semibold"),
                                html.Span(
                                    str(len(signal_summary['signal_details'][tech]['Sell'])),
                                    className="text-red-600 ml-1"
                                )
                            ], className="mr-6"),
                            html.Div([
                                html.Span(f"{tech.replace('_', ' ')} Hold Signals: ", className="font-semibold"),
                                html.Span(
                                    str(len(signal_summary['signal_details'][tech]['Hold'])),
                                    className="text-blue-600 ml-1"
                                )
                            ])
                        ], className="flex mb-4"),
                        
                        # Volume signal summary for Volume_Signal
                        html.Div([
                            html.Div([
                                html.Span("Volume Analysis: ", className="font-semibold"),
                                html.Span("High Volume: ", className="text-green-600 font-medium ml-2"),
                                html.Span(str(signal_summary.get('volume_signals', {}).get('High Volume', 0)), className="mr-3"),
                                html.Span("Low Volume: ", className="text-gray-600 font-medium"),
                                html.Span(str(signal_summary.get('volume_signals', {}).get('Low Volume', 0)))
                            ], className="mb-2")
                        ], className="mb-3") if tech == 'Volume_Signal' and 'volume_signals' in signal_summary else html.Div(),
                        
                        # Create table based on indicator type
                        html.Div([
                            dash_table.DataTable(
                                data=(signal_summary['signal_details'][tech]['Buy'] + 
                                    signal_summary['signal_details'][tech]['Sell'] +
                                    signal_summary['signal_details'][tech]['Hold']),
                                columns=[
                                    {'name': 'Date/Time', 'id': 'datetime'},
                                    {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Signal', 'id': 'signal'},
                                ] + (
                                    # Add Volume column for Volume_Signal or Final_Signal with Volume_Signal
                                    [{'name': 'Volume', 'id': 'volume', 'type': 'numeric', 'format': {'specifier': ','}}] 
                                    if tech == 'Volume_Signal' or (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)
                                    else []
                                ) + (
                                    # Add volume analysis column for Volume_Signal
                                    [{'name': 'Volume Analysis', 'id': 'volume_reason'}]
                                    if tech == 'Volume_Signal' or (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals)
                                    else []
                                ) + (
                                    # Add detected patterns column for Candlestick_Signal - UPDATED
                                    [{'name': 'Detected Patterns', 'id': 'detected_patterns'}]
                                    if tech == 'Candlestick_Signal' or (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)
                                    else []
                                ) 
                                + (
                                    # Add candlestick pattern column for Candlestick_Signal
                                    [{'name': 'Pattern Summary', 'id': 'patterns'}]
                                    if tech == 'Candlestick_Signal' or (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)
                                    else []
                                ) 
                                + (
                                    # Add trend column for Candlestick_Signal
                                    [{'name': 'Trend', 'id': 'trend'}]
                                    if tech == 'Candlestick_Signal' or (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)
                                    else []
                                ) + (
                                    # Add candlestick analysis column for Candlestick_Signal
                                    [{'name': 'Candlestick Analysis', 'id': 'candlestick_reason'}]
                                    if tech == 'Candlestick_Signal' or (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals)
                                    else []
                                ) + (
                                    # Add combined analysis column for Final_Signal
                                    [{'name': 'Combined Analysis', 'id': 'reason'}] 
                                    if tech == 'Final_Signal' else []
                                ),
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{signal} = "Buy"'},
                                        'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                                        'color': 'green'
                                    },
                                    {
                                        'if': {'filter_query': '{signal} = "Sell"'},
                                        'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                        'color': 'red'
                                    },
                                    {
                                        'if': {'filter_query': '{signal} = "Hold"'},
                                        'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                                        'color': 'blue'
                                    },
                                ] + (
                                    # Add trend-based formatting for Candlestick_Signal
                                    [
                                        {
                                            'if': {'filter_query': '{trend} = "Uptrend"'},
                                            'backgroundColor': 'rgba(0, 255, 0, 0.05)',
                                        },
                                        {
                                            'if': {'filter_query': '{trend} = "Downtrend"'},
                                            'backgroundColor': 'rgba(255, 0, 0, 0.05)',
                                        },
                                        {
                                            'if': {'filter_query': '{trend} = "Sideways"'},
                                            'backgroundColor': 'rgba(0, 0, 255, 0.05)',
                                        }
                                    ] if tech == 'Candlestick_Signal' or (tech == 'Final_Signal' and 'Candlestick_Signal' in selected_technicals) else []
                                ) + (
                                    # Add volume-based formatting for Volume_Signal
                                    [
                                        {
                                            'if': {'filter_query': '{volume_reason} contains "High Volume"'},
                                            'backgroundColor': 'rgba(0, 128, 0, 0.05)',
                                        },
                                        {
                                            'if': {'filter_query': '{volume_reason} contains "Low Volume"'},
                                            'backgroundColor': 'rgba(128, 128, 128, 0.05)',
                                        }
                                    ] if tech == 'Volume_Signal' or (tech == 'Final_Signal' and 'Volume_Signal' in selected_technicals) else []
                                ),
                                style_header={
                                    'backgroundColor': 'rgb(240, 240, 240)',
                                    'fontWeight': 'bold'
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'minWidth': '120px',
                                    'maxWidth': '200px',
                                    'whiteSpace': 'normal'
                                },
                                page_size=5,
                                sort_action='native',
                                filter_action='native',
                                sort_mode='multi'
                            )
                        ], className="mt-2")
                    ], className="mb-6 p-4 bg-white rounded shadow-sm")
                    
                    # Only include indicators that have signal data
                    for tech in selected_technicals + ['Final_Signal']
                    if tech in signal_summary['signal_details'] and (
                        signal_summary['signal_details'][tech]['Buy'] or 
                        signal_summary['signal_details'][tech]['Sell'] or
                        signal_summary['signal_details'][tech]['Hold']
                    )
                ]
            ]),
            
            # Dedicated Volume Analysis Table (Enhanced Version)
            html.Div([
                html.H3("Volume Analysis Detail", className="text-lg font-semibold mb-3"),
                
                # Volume Statistics Summary
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span("Total Volume Signals: ", className="font-semibold"),
                            html.Span(str(len(signal_summary.get('volume_data', []))), className="ml-1")
                        ], className="mr-6"),
                        html.Div([
                            html.Span("High Volume Signals: ", className="font-semibold"),
                            html.Span(
                                str(signal_summary.get('volume_stats', {}).get('high_volume_count', 0)), 
                                className="text-green-600 ml-1"
                            )
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Low Volume Signals: ", className="font-semibold"),
                            html.Span(
                                str(signal_summary.get('volume_stats', {}).get('low_volume_count', 0)), 
                                className="text-gray-600 ml-1"
                            )
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Average Volume: ", className="font-semibold"),
                            html.Span(
                                f"{signal_summary.get('volume_stats', {}).get('average_volume', 0):,.0f}" if signal_summary.get('volume_stats', {}).get('average_volume', 0) > 0 else "N/A", 
                                className="text-blue-600 ml-1"
                            )
                        ])
                    ], className="flex flex-wrap mb-4"),
                    
                    # Volume Trend Analysis
                    html.Div([
                        html.Div([
                            html.Span("Volume Trend: ", className="font-semibold"),
                            html.Span(
                                signal_summary.get('volume_stats', {}).get('trend', 'Unknown'), 
                                className=f"ml-1 {'text-green-600' if signal_summary.get('volume_stats', {}).get('trend', '') == 'Increasing' else 'text-red-600' if signal_summary.get('volume_stats', {}).get('trend', '') == 'Decreasing' else 'text-gray-600'}"
                            )
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Peak Volume: ", className="font-semibold"),
                            html.Span(
                                f"{signal_summary.get('volume_stats', {}).get('peak_volume', 0):,.0f}" if signal_summary.get('volume_stats', {}).get('peak_volume', 0) > 0 else "N/A", 
                                className="text-purple-600 ml-1"
                            )
                        ], className="mr-6"),
                        html.Div([
                            html.Span("Min Volume: ", className="font-semibold"),
                            html.Span(
                                f"{signal_summary.get('volume_stats', {}).get('min_volume', 0):,.0f}" if signal_summary.get('volume_stats', {}).get('min_volume', 0) > 0 else "N/A", 
                                className="text-orange-600 ml-1"
                            )
                        ])
                    ], className="flex mb-4")
                ], className="p-3 bg-blue-50 rounded-lg mb-4"),
                
                # Enhanced Volume Table - IMPROVED VERSION
                html.Div([
                    dash_table.DataTable(
                        data=signal_summary.get('volume_data', []),
                        columns=[
                            {'name': 'Date/Time', 'id': 'datetime'},
                            {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Volume', 'id': 'volume', 'type': 'numeric', 'format': {'specifier': ','}},
                            {'name': 'VMA', 'id': 'vma', 'type': 'numeric', 'format': {'specifier': ','}},
                            # {'name': 'Volume Ratio', 'id': 'volume_ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Volume Type', 'id': 'volume_type'},
                            # {'name': 'Signal', 'id': 'signal'},
                            {'name': 'Price Change %', 'id': 'price_change', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                            {'name': 'Volume Trend', 'id': 'volume_trend'},
                            {'name': 'Analysis', 'id': 'analysis'}
                        ],
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{signal} = "Buy"'},
                                'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                                'color': 'green'
                            },
                            {
                                'if': {'filter_query': '{signal} = "Sell"'},
                                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                'color': 'red'
                            },
                            {
                                'if': {'filter_query': '{signal} = "Hold"'},
                                'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                                'color': 'blue'
                            },
                            {
                                'if': {'filter_query': '{volume_type} = "High Volume"'},
                                'backgroundColor': 'rgba(0, 128, 0, 0.1)',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {'filter_query': '{volume_type} = "Low Volume"'},
                                'backgroundColor': 'rgba(128, 128, 128, 0.05)',
                            },
                            {
                                'if': {'filter_query': '{volume_ratio} > 2'},
                                'backgroundColor': 'rgba(255, 165, 0, 0.1)',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {'filter_query': '{volume_trend} = "Spike"'},
                                'backgroundColor': 'rgba(255, 0, 255, 0.1)',
                                'fontWeight': 'bold'
                            },
                            {
                                'if': {'filter_query': '{price_change} > 2'},
                                'backgroundColor': 'rgba(0, 255, 0, 0.15)',
                            },
                            {
                                'if': {'filter_query': '{price_change} < -2'},
                                'backgroundColor': 'rgba(255, 0, 0, 0.15)',
                            }
                        ],
                        style_header={
                            'backgroundColor': 'rgb(240, 240, 240)',
                            'fontWeight': 'bold'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'minWidth': '100px',
                            'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        },
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                        sort_mode='multi',
                        tooltip_data=[
                            {
                                'analysis': {'value': row.get('analysis', '') + f"\n\nVolume: {row.get('volume', 0):,}\nVMA: {row.get('vma', 0):,}\nRatio: {row.get('volume_ratio', 0):.2f}x\nPrice Change: {row.get('price_change', 0):.2f}%", 'type': 'markdown'}
                                for column in ['analysis']
                            } for row in signal_summary.get('volume_data', [])
                        ],
                        tooltip_duration=None
                    ) if signal_summary.get('volume_data', []) else html.Div([
                        html.P("No volume data available", className="text-center text-gray-500 italic p-4"),
                        html.P("Volume analysis requires Volume_Signal to be selected and data to be processed.", 
                            className="text-center text-sm text-gray-400")
                    ])
                ], className="mt-2")
            ], className="mb-6 p-4 bg-white rounded shadow-sm") if 'Volume_Signal' in selected_technicals else html.Div()
        ])

        
        
        # Handle save button
        save_status = ""
        if triggered_id == 'save-button' and n_clicks > 0:
            if save_title:
                if save_signals_to_db(save_title, ticker, signal_summary, selected_technicals):
                    save_status = f"Signals saved successfully for {ticker}!"
                else:
                    save_status = f"Error saving signals for {ticker} to database."
            else:
                save_status = "Please enter a title before saving."

        # Format table data
        displayed_columns = base_columns + technical_columns + ['signal_akhir']
        displayed_columns = [col for col in displayed_columns if col in filtered_df.columns]
        table_data = filtered_df[displayed_columns].to_dict('records')
        table_columns = [{'name': col, 'id': col} for col in displayed_columns]

        # Add chart type selector (candlestick/line)
        chart_type_selector = dcc.RadioItems(
            id='combination-chart-type',
            options=[
                {'label': 'Candlestick', 'value': 'candlestick'},
                {'label': 'Line', 'value': 'line'}
            ],
            value='candlestick',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        )

        # Combination Graph (TradingView style)
        def get_combination_graph(chart_type='candlestick'):
            import plotly.graph_objs as go
            fig = go.Figure()
            # Price chart
            if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig.add_trace(go.Candlestick(
                    x=filtered_df['Datetime'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name='Candlestick',
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    showlegend=True
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close',
                    line=dict(color='black', width=2)
                ))
            # Overlay indicators
            if 'Bollinger_Signal' in selected_technicals and 'Upper Band' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Upper Band'], mode='lines', name='Upper Band', line=dict(color='blue', dash='dot')))
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Middle Band'], mode='lines', name='Middle Band', line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Lower Band'], mode='lines', name='Lower Band', line=dict(color='blue', dash='dot')))
            if 'MA_Signal' in selected_technicals and 'short_MA' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['short_MA'], mode='lines', name='Short MA', line=dict(color='orange')))
            if 'MA_Signal' in selected_technicals and 'long_MA' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['long_MA'], mode='lines', name='Long MA', line=dict(color='red')))
            if 'Fibonacci_Signal' in selected_technicals and 'fib_levels' in filtered_df.columns:
                retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                colors = ['#FFB347', '#FFD700', '#90EE90', '#87CEEB', '#FF69B4']
                for i, level in enumerate(retracement_levels):
                    y_vals = filtered_df['fib_levels'].apply(lambda x: x[i] if isinstance(x, list) else None)
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'],
                        y=y_vals,
                        mode='lines',
                        name=f'Fibo {level*100:.1f}%',
                        line=dict(color=colors[i], dash='dash')
                    ))
                if 'fib_high' in filtered_df.columns:
                    fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['fib_high'], mode='lines', name='Fibo High', line=dict(color='red', dash='dot')))
                if 'fib_low' in filtered_df.columns:
                    fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['fib_low'], mode='lines', name='Fibo Low', line=dict(color='green', dash='dot')))
            # Add buy/sell/hold markers for final signal
            if 'signal_akhir' in filtered_df.columns:
                buy_mask = filtered_df['signal_akhir'] == 'Buy'
                sell_mask = filtered_df['signal_akhir'] == 'Sell'
                hold_mask = filtered_df['signal_akhir'] == 'Hold'
                fig.add_trace(go.Scatter(
                    x=filtered_df.loc[buy_mask, 'Datetime'],
                    y=filtered_df.loc[buy_mask, 'Close'],
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', symbol='triangle-up', size=10)
                ))
                fig.add_trace(go.Scatter(
                    x=filtered_df.loc[sell_mask, 'Datetime'],
                    y=filtered_df.loc[sell_mask, 'Close'],
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', symbol='triangle-down', size=10)
                ))
                fig.add_trace(go.Scatter(
                    x=filtered_df.loc[hold_mask, 'Datetime'],
                    y=filtered_df.loc[hold_mask, 'Close'],
                    mode='markers',
                    name='Hold',
                    marker=dict(color='blue', symbol='circle', size=7)
                ))
            fig.update_layout(
                title="Combination Graph (All Indicators)",
                xaxis_title="Tanggal",
                yaxis_title="Harga",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            return dcc.Graph(figure=fig)

        # Tabs for charts
        indicator_names = {
            'Bollinger_Signal': 'BB Chart',
            'MA_Signal': 'MA Chart',
            'RSI_Signal': 'RSI Chart',
            'MACD_Signal': 'MACD Chart',
            'ADX_Signal': 'ADX Chart',
            'Volume_Signal': 'Volume Chart',
            'Fibonacci_Signal': 'Fibonacci Chart',
            'Candlestick_Signal': 'Candlestick Chart'
        }

        chart_tabs = [
            dcc.Tab(label='Combination Graph', children=[
                html.Div([
                    html.Label('Chart Type:'),
                    dcc.RadioItems(
                        id='shared-chart-type',
                        options=[
                            {'label': 'Candlestick', 'value': 'candlestick'},
                            {'label': 'Line', 'value': 'line'}
                        ],
                        value='candlestick',
                        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                    ),
                    html.Div(id='combination-graph-container'),
                    html.Div(indicator_explanations['combination'] if show_explanations == 'yes' else '', id='combination-explanation')
                ])
            ]),
            dcc.Tab(label='All Graph', children=[
                html.Div([
                    html.Label('Chart Type:'),
                    html.Div(id='synced-chart-type-container'),
                    html.Div(id='all-graph-container'),
                    html.Div(indicator_explanations['all_graph'] if show_explanations == 'yes' else '', id='all-graph-explanation')
                ])
            ]),
        ]

        # Conditionally add the "Candlestick Pattern Analysis" tab
        if 'Candlestick_Signal' in selected_technicals:
            candlestick_analysis_tab_content = html.Div([
                html.Div([
                    html.H2("Candlestick Pattern Analysis", className="text-2xl font-bold mb-4"),
                    html.P("""
                        Candlestick patterns help traders understand market psychology and potential price movements.
                        This analysis identifies specific patterns and provides detailed explanations of what they indicate.
                    """, className="mb-6 text-gray-700"),
                    
                    html.Div([
                        html.Div([
                            html.H3("Pattern Overview", className="text-xl font-semibold mb-2"),
                            html.Div(id="pattern-summary-stats", className="grid grid-cols-3 gap-4 mb-4")
                        ], className="mb-6 p-4 bg-white rounded-lg shadow"),
                        
                        html.Div([
                            html.H3("Latest Detected Patterns", className="text-xl font-semibold mb-2"),
                            html.Div(id="latest-patterns-display", className="overflow-x-auto")
                        ], className="mb-6 p-4 bg-white rounded-lg shadow"),
                        
                        html.Div(id="pattern-detail-report", className="p-4 bg-white rounded-lg shadow")
                    ])
                ], className="max-w-7xl mx-auto")
            ])
            chart_tabs.append(dcc.Tab(label='Candlestick Pattern Analysis', children=candlestick_analysis_tab_content))

        # Add per-indicator charts as other tabs with explanations only if enabled
        for tech, chart_component in zip(selected_technicals, charts): # charts is the list of dcc.Graph or html.Div components
            tab_label = indicator_names.get(tech, tech)
            
            # 'chart_component' can be a dcc.Graph or an html.Div (like an explanation)
            # The structure below assumes 'chart_component' is the primary content for the tab
            children_content = [chart_component]
            if show_explanations == 'yes':
                # Add specific explanation for this indicator if available
                # The 'charts' list might already contain explanations for some indicators (like candlestick)
                # This line adds a generic explanation if not already handled by chart_component
                if not (isinstance(chart_component, html.Div) and "Understanding Candlestick Patterns" in str(chart_component.children)): # Avoid double explanation
                    children_content.append(html.Div(indicator_explanations.get(tech, html.Div())))

            chart_tabs.append(dcc.Tab(label=tab_label, children=html.Div(children_content)))
            
        charts_container = dcc.Tabs(chart_tabs, id='chart-tabs')
        
        # Rest of your function continues as before
        return charts_container, table_data, table_columns, table_content, save_status
        
    except Exception as e:
        print(f"Error in update_analysis: {str(e)}")
        return [], [], [], None, f"Error: {str(e)}"



# Add callbacks for the new pattern analysis components
@app.callback(
    [Output("pattern-summary-stats", "children"),
     Output("latest-patterns-display", "children"),
     Output("pattern-detail-report", "children")],
    [Input('ticker-dropdown', 'value'),
     Input('technical-checklist', 'value'),
     Input('start-year-dropdown', 'value'),
     Input('start-month-dropdown', 'value'),
     Input('start-day-dropdown', 'value'),
     Input('end-year-dropdown', 'value'),
     Input('end-month-dropdown', 'value'),
     Input('end-day-dropdown', 'value'),
     Input('data-period-selector', 'value'),
     Input('realtime-interval-selector', 'value'),
     Input('candlestick-confidence', 'value')]
)
def update_pattern_analysis(ticker, selected_technicals, 
                          start_year, start_month, start_day,
                          end_year, end_month, end_day, 
                          selected_period, selected_interval,
                          confidence_threshold):
    
    if not selected_technicals or 'Candlestick_Signal' not in selected_technicals:
        no_data_msg = html.Div("Enable Candlestick Pattern analysis to see details.", className="text-center text-gray-500 p-4")
        return no_data_msg, no_data_msg, no_data_msg
    
    print(f"Update pattern analysis called with ticker: {ticker} and selected_technicals: {selected_technicals}")
    
    # Check if ticker is selected
    if not ticker:
        return [
            html.Div("Please select a ticker", className="text-center text-gray-500 col-span-3"),
            html.Div("No patterns to display", className="text-center text-gray-500"),
            html.Div("No pattern details to display", className="text-center text-gray-500")
        ]
    
    # Check if date inputs are valid
    if None in [start_year, start_month, start_day, end_year, end_month, end_day]:
        return [
            html.Div("Please select valid date range", className="text-center text-gray-500 col-span-3"),
            html.Div("Invalid date range", className="text-center text-gray-500"),
            html.Div("Invalid date range", className="text-center text-gray-500")
        ]
    
    try:
        # Load and process data with candlestick patterns
        filtered_df = load_and_process_data(
            period=selected_period, 
            ticker=ticker,
            interval=selected_interval if selected_period == 'realtime' else '1min'
        )
        
        if filtered_df.empty:
            return [
                html.Div("No data available for the selected ticker", 
                         className="text-center text-gray-500 col-span-3"),
                html.Div("No patterns to display", className="text-center text-gray-500"),
                html.Div("No pattern details to display", className="text-center text-gray-500")
            ]
        
        print(f"Data loaded. Shape: {filtered_df.shape}")
        print(f"Data columns: {filtered_df.columns.tolist()}")
        
        # Apply date filtering
        try:
            # Convert selected dates to datetime - adjust for different data types
            if selected_period == 'max':  # Daily data - no time component
                start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
                end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day).replace(hour=23, minute=59, second=59)
            else:  # Hourly or minute data - with time component
                start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day).replace(hour=0, minute=0, second=0)
                end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day).replace(hour=23, minute=59, second=59)
            
            print(f"Date range: {start_date} to {end_date}")
            print(f"Data datetime range: {filtered_df['Datetime'].min()} to {filtered_df['Datetime'].max()}")
            
            # Filter data by date range
            initial_count = len(filtered_df)
            filtered_df = filtered_df[
                (filtered_df['Datetime'] >= start_date) &
                (filtered_df['Datetime'] <= end_date)
            ].copy()
            
            print(f"Data after date filtering: {len(filtered_df)} rows (was {initial_count})")
            
            if filtered_df.empty:
                return [
                    html.Div("No data available for the selected date range", 
                             className="text-center text-gray-500 col-span-3"),
                    html.Div("No patterns to display", className="text-center text-gray-500"),
                    html.Div("No pattern details to display", className="text-center text-gray-500")
                ]
        except Exception as e:
            print(f"Date filtering error: {str(e)}")
            return [
                html.Div(f"Error filtering date range: {str(e)}", 
                         className="text-center text-red-500 col-span-3"),
                html.Div("Error processing date range", className="text-center text-red-500"),
                html.Div("Error processing date range", className="text-center text-red-500")
            ]
        
        # Apply candlestick pattern detection
        print("Applying candlestick pattern detection")
        
        try:
            # Ensure we have the required OHLC columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in filtered_df.columns]
            if missing_columns:
                return [
                    html.Div(f"Missing required columns: {missing_columns}", 
                             className="text-center text-red-500 col-span-3"),
                    html.Div("Cannot perform candlestick analysis", className="text-center text-red-500"),
                    html.Div("Missing OHLC data", className="text-center text-red-500")
                ]
            
            # Apply the pattern detection
            filtered_df = detect_candlestick_patterns(filtered_df)
            
            print(f"Columns after pattern detection: {filtered_df.columns.tolist()}")
            
        except Exception as e:
            print(f"Pattern detection error: {str(e)}")
            return [
                html.Div(f"Error detecting patterns: {str(e)}", 
                         className="text-center text-red-500 col-span-3"),
                html.Div("Pattern detection failed", className="text-center text-red-500"),
                html.Div("Pattern detection failed", className="text-center text-red-500")
            ]
        
        # Check if the detector created the required columns
        if 'Candlestick_Signal' not in filtered_df.columns:
            print("Candlestick_Signal column not found - calculating manually")
            
            # Get pattern columns (those starting with CDL)
            pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL')]
            
            if not pattern_columns:
                return [
                    html.Div("No candlestick patterns detected in data", 
                             className="text-center text-orange-500 col-span-3"),
                    html.Div("No pattern columns found", className="text-center text-orange-500"),
                    html.Div("Try with different data or time period", className="text-center text-orange-500")
                ]
            
            # Define pattern categories
            bullish_patterns = [
                'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
                'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
                'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
                'CDLTWEEZERBOTTOM'
            ]
            
            bearish_patterns = [
                'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
                'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
                'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
                'CDLGRAVESTONEDOJI', 'CDLTWEEZERTOP'
            ]
            
            # Create summary columns
            filtered_df['bullish_patterns'] = 0
            filtered_df['bearish_patterns'] = 0
            
            # Sum pattern counts with error handling
            for pattern in bullish_patterns:
                if pattern in filtered_df.columns:
                    filtered_df['bullish_patterns'] += filtered_df[pattern].fillna(0)
            
            for pattern in bearish_patterns:
                if pattern in filtered_df.columns:
                    filtered_df['bearish_patterns'] += filtered_df[pattern].fillna(0)
            
            # Generate signals
            filtered_df['Candlestick_Signal'] = 'Hold'
            bull_mask = filtered_df['bullish_patterns'] > filtered_df['bearish_patterns']
            bear_mask = filtered_df['bearish_patterns'] > filtered_df['bullish_patterns']
            
            filtered_df.loc[bull_mask, 'Candlestick_Signal'] = 'Buy'
            filtered_df.loc[bear_mask, 'Candlestick_Signal'] = 'Sell'
            
            # Add confidence score
            filtered_df['candlestick_confidence'] = 50  # Default
            
            # Calculate max patterns for scaling (avoid division by zero)
            max_bull = filtered_df['bullish_patterns'].max() if len(filtered_df) > 0 else 0
            max_bear = filtered_df['bearish_patterns'].max() if len(filtered_df) > 0 else 0
            max_patterns = max(max_bull, max_bear, 1)
            
            # Calculate confidence
            if bull_mask.any():
                filtered_df.loc[bull_mask, 'candlestick_confidence'] = (
                    50 + 50 * (filtered_df.loc[bull_mask, 'bullish_patterns'] / max_patterns)
                ).clip(0, 100)
            
            if bear_mask.any():
                filtered_df.loc[bear_mask, 'candlestick_confidence'] = (
                    50 + 50 * (filtered_df.loc[bear_mask, 'bearish_patterns'] / max_patterns)
                ).clip(0, 100)
        
        # Apply confidence threshold
        if confidence_threshold and 'candlestick_confidence' in filtered_df.columns:
            print(f"Applying confidence threshold: {confidence_threshold}")
            low_confidence_mask = filtered_df['candlestick_confidence'] < confidence_threshold
            filtered_df.loc[low_confidence_mask, 'Candlestick_Signal'] = 'Hold'
        
        # Get pattern columns for analysis
        pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL') and 
                          col not in ['Candlestick_Signal', 'candlestick_confidence']]
        
        print(f"Pattern columns found: {pattern_columns}")
        
        # Check if any patterns were actually detected
        if not pattern_columns:
            return [
                html.Div("No candlestick pattern columns found in the data", 
                         className="text-center text-orange-500 col-span-3"),
                html.Div("Pattern detection may have failed", className="text-center text-orange-500"),
                html.Div("Try with different parameters", className="text-center text-orange-500")
            ]
        
        # Calculate total patterns with safe operations
        try:
            total_patterns = sum(filtered_df[col].sum() for col in pattern_columns if col in filtered_df.columns)
            print(f"Total patterns detected: {total_patterns}")
            
            if total_patterns == 0:
                return [
                    html.Div("No candlestick patterns detected in the selected data range", 
                             className="text-center text-orange-500 col-span-3"),
                    html.Div("Try adjusting the date range or selecting a different ticker", 
                             className="text-center text-orange-500"),
                    html.Div("Candlestick patterns work best with daily or hourly data", 
                             className="text-center text-orange-500")
                ]
        except Exception as e:
            print(f"Error calculating total patterns: {str(e)}")
            return [
                html.Div(f"Error calculating patterns: {str(e)}", 
                         className="text-center text-red-500 col-span-3"),
                html.Div("Pattern calculation failed", className="text-center text-red-500"),
                html.Div("Pattern calculation failed", className="text-center text-red-500")
            ]
        
        # Define pattern categories for summary
        bullish_patterns_list = [
            'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
            'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
            'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
            'CDLTWEEZERBOTTOM'
        ]
        
        bearish_patterns_list = [
            'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
            'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
            'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
            'CDLGRAVESTONEDOJI', 'CDLTWEEZERTOP'
        ]
        
        neutral_patterns_list = [
            'CDLDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU', 'CDLTAKURI', 
            'CDLRICKSHAWMAN', 'CDLLONGLEGGEDDOJI', 'CDLDRAGONFLYDOJI'
        ]
        
        # Calculate pattern totals safely
        total_bullish = sum(filtered_df[col].sum() for col in bullish_patterns_list if col in filtered_df.columns)
        total_bearish = sum(filtered_df[col].sum() for col in bearish_patterns_list if col in filtered_df.columns)
        total_neutral = sum(filtered_df[col].sum() for col in neutral_patterns_list if col in filtered_df.columns)
        
        print(f"Pattern totals - Bullish: {total_bullish}, Bearish: {total_bearish}, Neutral: {total_neutral}")
        
        # 1. Create Summary Stats
        summary_stats = []
        
        # Pattern totals card
        summary_stats.append(html.Div([
            html.H4("Pattern Totals", className="text-lg font-medium mb-2"),
            html.Div([
                html.Div([
                    html.Span("Total Patterns: ", className="font-semibold"),
                    html.Span(str(total_patterns))
                ], className="mb-1"),
                html.Div([
                    html.Span("Bullish Patterns: ", className="font-semibold text-green-700"),
                    html.Span(str(total_bullish))
                ], className="mb-1"),
                html.Div([
                    html.Span("Bearish Patterns: ", className="font-semibold text-red-700"),
                    html.Span(str(total_bearish))
                ], className="mb-1"),
                html.Div([
                    html.Span("Neutral Patterns: ", className="font-semibold text-blue-700"),
                    html.Span(str(total_neutral))
                ])
            ])
        ], className="bg-gray-50 p-4 rounded"))
        
        # Trend analysis with safe operations
        trend_data = [0, 0, 0]  # Default values for Uptrend, Downtrend, Sideways
        if 'candlestick_trend' in filtered_df.columns:
            uptrend_count = (filtered_df['candlestick_trend'] == 'Uptrend').sum()
            downtrend_count = (filtered_df['candlestick_trend'] == 'Downtrend').sum()
            sideways_count = (filtered_df['candlestick_trend'] == 'Sideways').sum()
            trend_data = [uptrend_count, downtrend_count, sideways_count]
        
        summary_stats.append(html.Div([
            html.H4("Trend Analysis", className="text-lg font-medium mb-2"),
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Pie(
                            labels=['Uptrend', 'Downtrend', 'Sideways'],
                            values=trend_data,
                            marker=dict(colors=['green', 'red', 'blue']),
                            textinfo='label+percent'
                        )
                    ],
                    layout=go.Layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=200,
                        showlegend=False
                    )
                )
            )
        ], className="bg-gray-50 p-4 rounded"))
        
        # 2. Create Latest Patterns Display - FIXED VERSION
        pattern_rows = []
        
        # Create a list of all pattern occurrences with their dates
        all_pattern_occurrences = []
        
        for pattern_col_name in pattern_columns:
            if pattern_col_name not in filtered_df.columns:
                continue
                
            # Find all rows where this pattern occurs
            pattern_mask = filtered_df[pattern_col_name] == 1
            pattern_occurrences = filtered_df[pattern_mask]
            
            for idx, row in pattern_occurrences.iterrows():
                # Safe datetime formatting
                datetime_val = row['Datetime']
                if pd.isna(datetime_val):
                    continue  # Skip invalid dates
                    
                try:
                    if selected_period == 'max':  # Daily data
                        formatted_datetime = datetime_val.strftime("%Y-%m-%d")
                    else:  # Hourly or minute data
                        formatted_datetime = datetime_val.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_datetime = str(datetime_val)
                
                # Determine pattern type
                pattern_type = 'Neutral'
                if pattern_col_name in bullish_patterns_list:
                    pattern_type = 'Bullish'
                elif pattern_col_name in bearish_patterns_list:
                    pattern_type = 'Bearish'
                
                # Safe price formatting
                try:
                    price = float(row['Close'])
                except:
                    price = 0.0
                
                all_pattern_occurrences.append({
                    'Datetime': formatted_datetime,
                    'datetime_obj': datetime_val,  # Keep original for sorting
                    'Pattern': pattern_col_name.replace('CDL', ''),
                    'Price': price,
                    'Signal': pattern_type
                })
        
        # Sort all occurrences by datetime (most recent first) and take only the latest 10
        if all_pattern_occurrences:
            # Sort by datetime_obj in descending order (most recent first)
            all_pattern_occurrences.sort(key=lambda x: x['datetime_obj'], reverse=True)
            
            # Take the 10 most recent patterns
            pattern_rows = []
            for occurrence in all_pattern_occurrences[:10]:
                pattern_rows.append({
                    'Datetime': occurrence['Datetime'],
                    'Pattern': occurrence['Pattern'],
                    'Price': occurrence['Price'],
                    'Signal': occurrence['Signal']
                })
        
        # Create latest patterns table
        if not pattern_rows:
            latest_table = html.Div("No specific patterns detected in the selected time range.",
                                    className="text-center text-gray-500 italic")
        else:
            latest_table = dash_table.DataTable(
                data=pattern_rows,
                columns=[
                    {'name': 'Date/Time', 'id': 'Datetime'},
                    {'name': 'Pattern', 'id': 'Pattern'},
                    {'name': 'Price', 'id': 'Price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Signal Type', 'id': 'Signal'}
                ],
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Signal} = "Bullish"'},
                        'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                        'color': 'green'
                    },
                    {
                        'if': {'filter_query': '{Signal} = "Bearish"'},
                        'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                        'color': 'red'
                    },
                    {
                        'if': {'filter_query': '{Signal} = "Neutral"'},
                        'backgroundColor': 'rgba(0, 0, 255, 0.1)',
                        'color': 'blue'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(240, 240, 240)',
                    'fontWeight': 'bold'
                },
                style_cell={'textAlign': 'left', 'padding': '10px'},
                page_size=10
            )
        
        # 3. Generate the detailed pattern report
        try:
            pattern_report = create_candlestick_analysis_report(filtered_df)
        except Exception as e:
            print(f"Error creating pattern report: {str(e)}")
            pattern_report = html.Div(f"Error generating detailed report: {str(e)}", 
                                    className="text-red-500")
        
        return summary_stats, latest_table, pattern_report
        
    except Exception as e:
        print(f"Error in update_pattern_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return [
            html.Div(f"Error: {str(e)}", className="text-center text-red-500 col-span-3"),
            html.Div("Error loading pattern data", className="text-center text-red-500"),
            html.Div("Error generating pattern report", className="text-center text-red-500")
        ]




# Also make sure your create_candlestick_analysis_report function uses the imported get_pattern_description
def create_candlestick_analysis_report(filtered_df):
    """Generate a detailed report of candlestick patterns found in the data"""
    
    # Check if DataFrame is empty first
    if filtered_df.empty:
        return html.Div("No data available for candlestick pattern analysis.",
                      className="text-gray-500 italic")
    
    # Get all pattern columns
    pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL') and 
                     col not in ['Candlestick_Signal', 'candlestick_confidence',
                               'bullish_patterns', 'bearish_patterns', 'neutral_patterns']]
    
    # Check if any pattern columns exist
    if not pattern_columns:
        return html.Div("No candlestick pattern columns found in the data.",
                      className="text-gray-500 italic")
    
    # Define pattern categories
    bullish_patterns_list = [
        'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
        'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
        'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
        'CDLTWEEZERBOTTOM'
    ]
    
    bearish_patterns_list = [
        'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
        'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
        'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
        'CDLGRAVESTONEDOJI', 'CDLTWEEZERTOP'
    ]
    
    neutral_patterns_list = [
        'CDLDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU', 'CDLTAKURI', 
        'CDLRICKSHAWMAN', 'CDLLONGLEGGEDDOJI', 'CDLDRAGONFLYDOJI'
    ]
    
    # Create a summary of detected patterns
    detected_patterns = {}
    for pattern_col_name in pattern_columns:
        if pattern_col_name in filtered_df.columns:
            try:
                if filtered_df[pattern_col_name].isna().all():
                    continue
                    
                count = int(filtered_df[pattern_col_name].sum())
                if count > 0:
                    try:
                        pattern_mask = (filtered_df[pattern_col_name] == 1) & (filtered_df[pattern_col_name].notna())
                        if not pattern_mask.any():
                            continue
                            
                        pattern_dates = filtered_df.loc[pattern_mask, 'Datetime'].tolist()
                        
                        if not pattern_dates:
                            continue
                            
                    except Exception as date_error:
                        print(f"Error getting dates for pattern {pattern_col_name}: {str(date_error)}")
                        continue
                    
                    try:
                        description = get_pattern_description(pattern_col_name)
                    except Exception as e:
                        print(f"Error getting description for {pattern_col_name}: {str(e)}")
                        description = "Description not available."
                    
                    detected_patterns[pattern_col_name] = {
                        'count': count,
                        'dates': pattern_dates,
                        'description': description,
                        'type': 'bullish' if pattern_col_name in bullish_patterns_list else 
                              'bearish' if pattern_col_name in bearish_patterns_list else 
                              'neutral' if pattern_col_name in neutral_patterns_list else 'unknown'
                    }
            except Exception as e:
                print(f"Error processing pattern {pattern_col_name}: {str(e)}")
                continue
    
    if not detected_patterns:
        return html.Div("No specific candlestick patterns detected in the selected time range.",
                      className="text-gray-500 italic")
    
    report_items = []
    
    pattern_by_type = {
        'bullish': {k: v for k, v in detected_patterns.items() if v['type'] == 'bullish'},
        'bearish': {k: v for k, v in detected_patterns.items() if v['type'] == 'bearish'},
        'neutral': {k: v for k, v in detected_patterns.items() if v['type'] == 'neutral'}
    }
    
    for pattern_type, color_class, bg_class in [
        ('bullish', 'text-green-600', 'bg-green-50'),
        ('bearish', 'text-red-600', 'bg-red-50'),
        ('neutral', 'text-blue-600', 'bg-blue-50')
    ]:
        if pattern_by_type[pattern_type]:
            report_items.append(html.H3(f"{pattern_type.title()} Patterns", 
                                      className=f"text-xl font-bold {color_class} mt-4 mb-2"))
            
            for pattern, data in pattern_by_type[pattern_type].items():
                card_children = [
                    html.H4(pattern.replace('CDL', ''), className=f"text-lg font-semibold {color_class.replace('600', '700')}"),
                    html.Div([
                        html.P(f"Occurrences: {data['count']}", className="font-medium"),
                        html.P(data['description'], className="mt-2 text-sm"),
                    ])
                ]

                # --- START OF MODIFIED SECTION ---
                # Logic to create an expandable list for dates
                if data['dates']:
                    # If there are more than 5 dates, make the list expandable
                    if len(data['dates']) > 5:
                        dates_component = html.Details([
                            html.Summary(
                                f"Dates Detected ({len(data['dates'])}): Click to expand",
                                className="mt-2 font-medium cursor-pointer text-sm"
                            ),
                            html.Ul([
                                html.Li(
                                    date.strftime("%Y-%m-%d %H:%M") if pd.notna(date) and hasattr(date, 'strftime') else str(date), 
                                    className="text-sm"
                                )
                                for date in data['dates'] # Loop through all dates
                            ], className="list-disc ml-5 mt-1")
                        ])
                    # Otherwise, just show the list directly
                    else:
                        dates_component = html.Div([
                            html.P("Dates Detected:", className="mt-2 font-medium"),
                            html.Ul([
                                html.Li(
                                    date.strftime("%Y-%m-%d %H:%M") if pd.notna(date) and hasattr(date, 'strftime') else str(date), 
                                    className="text-sm"
                                )
                                for date in data['dates']
                            ], className="list-disc ml-5")
                        ])
                    
                    # Find the correct place to insert the dates component
                    # It should be inside the second Div in card_children
                    card_children[1].children.append(dates_component)
                # --- END OF MODIFIED SECTION ---

                # Attempt to add a mini-chart for the first occurrence with comprehensive safety checks
                if data['dates'] and len(data['dates']) > 0:
                    try:
                        example_date = data['dates'][0]
                        if pd.notna(example_date):
                            try:
                                date_match_mask = filtered_df['Datetime'] == example_date
                                date_match_indices = filtered_df[date_match_mask].index
                                
                                if len(date_match_indices) == 0:
                                    # No need to show error, chart will just not appear
                                    continue
                                
                                matched_index = date_match_indices[0]
                                
                                try:
                                    index_list = filtered_df.index.tolist()
                                    if matched_index not in index_list:
                                        continue
                                        
                                    actual_index_pos = index_list.index(matched_index)
                                    
                                except (ValueError, KeyError) as e:
                                    print(f"Error finding index position for {pattern}: {str(e)}")
                                    continue
                                
                                total_rows = len(filtered_df)
                                if total_rows == 0:
                                    continue
                                
                                context_size = min(5, total_rows // 2)
                                start_idx = max(0, actual_index_pos - context_size)
                                end_idx = min(total_rows, actual_index_pos + context_size + 1)
                                
                                if start_idx >= end_idx:
                                    start_idx = max(0, actual_index_pos - 1)
                                    end_idx = min(total_rows, actual_index_pos + 2)
                                
                                if start_idx >= total_rows or end_idx <= start_idx:
                                    continue
                                
                                try:
                                    chart_df_mini = filtered_df.iloc[start_idx:end_idx].copy()
                                except (IndexError, ValueError) as slice_error:
                                    print(f"Slicing error for {pattern}: {str(slice_error)}")
                                    continue

                                if (chart_df_mini.empty or 
                                    len(chart_df_mini) == 0 or
                                    not all(col in chart_df_mini.columns for col in ['Open', 'High', 'Low', 'Close', 'Datetime'])):
                                    continue

                                if (chart_df_mini[['Open', 'High', 'Low', 'Close']].isna().all().all() or
                                    (chart_df_mini[['Open', 'High', 'Low', 'Close']] <= 0).all().all()):
                                    continue

                                try:
                                    mini_fig = go.Figure(data=[go.Candlestick(
                                        x=chart_df_mini['Datetime'],
                                        open=chart_df_mini['Open'],
                                        high=chart_df_mini['High'],
                                        low=chart_df_mini['Low'],
                                        close=chart_df_mini['Close'],
                                        name=pattern.replace('CDL', ''),
                                        showlegend=False
                                    )])
                                    
                                    try:
                                        title_date = example_date.strftime('%Y-%m-%d') if hasattr(example_date, 'strftime') else str(example_date)[:10]
                                    except:
                                        title_date = "Unknown Date"
                                    
                                    mini_fig.update_layout(
                                        title=f"Example: {pattern.replace('CDL', '')} on {title_date}",
                                        xaxis_title=None, 
                                        yaxis_title=None,
                                        showlegend=False,
                                        height=200, 
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        xaxis_rangeslider_visible=False,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    
                                    try:
                                        mini_fig.add_vline(x=example_date, line_width=1, line_dash="dash", line_color="purple")
                                    except Exception as vline_error:
                                        print(f"Could not add vertical line for {pattern}: {str(vline_error)}")
                                    
                                    # Append the chart to the main card_children list
                                    card_children.append(dcc.Graph(
                                        figure=mini_fig, 
                                        config={'displayModeBar': False}, 
                                        className="mt-2"
                                    ))
                                    
                                except Exception as chart_error:
                                    print(f"Error creating mini-chart figure for {pattern}: {str(chart_error)}")
                                    
                            except Exception as date_match_error:
                                print(f"Error matching date for {pattern}: {str(date_match_error)}")
                        else:
                            # Silently ignore if date is invalid
                            pass
                    except Exception as e:
                        print(f"Error creating mini-chart for {pattern}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        card_children.append(html.P("Mini-chart creation failed due to data constraints.", 
                                                   className="text-xs text-red-500 mt-1"))
                
                border_color = 'border-green-200' if pattern_type == 'bullish' else \
                              'border-red-200' if pattern_type == 'bearish' else 'border-blue-200'
                
                card = html.Div(card_children, 
                               className=f"mb-5 p-4 border {border_color} rounded-lg {bg_class}")
                report_items.append(card)
    
    report_items.append(html.Div([
        html.H3("How to Use Candlestick Patterns", className="text-xl font-bold mt-6 mb-2"),
        html.Ul([
            html.Li("Look for patterns at key support/resistance levels for stronger signals", className="mb-1"),
            html.Li("Patterns are more reliable when confirmed by other technical indicators", className="mb-1"),
            html.Li("Volume often increases during pattern formation in reliable signals", className="mb-1"),
            html.Li("Multiple patterns occurring together provide stronger confirmation", className="mb-1"),
            html.Li("Always wait for candle completion before acting on a pattern", className="mb-1"),
        ], className="list-disc ml-5 mb-3"),
        html.P("Remember that no pattern guarantees future price movement. Use candlestick patterns as part of a complete trading strategy with proper risk management.", 
              className="text-sm italic")
    ], className="mt-6 p-4 bg-gray-50 rounded-lg"))
    
    return html.Div(report_items, className="mt-4")






def get_pattern_examples():
    """Returns HTML examples of important candlestick patterns with explanations"""
    pattern_examples = html.Div([
        html.H3("Common Candlestick Patterns", className="text-xl font-bold mb-4"),
        
        # Doji Patterns
        html.Div([
            html.H4("Doji Patterns", className="text-lg font-semibold text-blue-600 mb-2"),
            html.Div([
                # Regular Doji
                html.Div([
                    html.H5("Doji", className="font-medium"),
                    html.Img(src="assets/patterns/doji.png", className="h-24 w-auto"),
                    html.P("Opening and closing prices are virtually equal, with varying shadow lengths.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border rounded"),
                
                # Dragonfly Doji
                html.Div([
                    html.H5("Dragonfly Doji", className="font-medium"),
                    html.Img(src="assets/patterns/dragonfly_doji.png", className="h-24 w-auto"),
                    html.P("Opening and closing prices at the high with a long lower shadow.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border rounded"),
                
                # Gravestone Doji
                html.Div([
                    html.H5("Gravestone Doji", className="font-medium"),
                    html.Img(src="assets/patterns/gravestone_doji.png", className="h-24 w-auto"),
                    html.P("Opening and closing prices at the low with a long upper shadow.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border rounded"),
                
                # Long-legged Doji
                html.Div([
                    html.H5("Long-legged Doji", className="font-medium"),
                    html.Img(src="assets/patterns/long_legged_doji.png", className="h-24 w-auto"),
                    html.P("Opening and closing prices are equal with long upper and lower shadows.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border rounded"),
            ], className="grid grid-cols-4 gap-4 mb-6")
        ]),
        
        # Single Candle Patterns
        html.Div([
            html.H4("Single Candle Patterns", className="text-lg font-semibold mb-2"),
            html.Div([
                # Hammer
                html.Div([
                    html.H5("Hammer", className="font-medium text-green-600"),
                    html.Img(src="assets/patterns/hammer.png", className="h-24 w-auto"),
                    html.P("Small body at the top with a long lower shadow. Bullish reversal at bottom of downtrend.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-green-200 rounded bg-green-50"),
                
                # Inverted Hammer
                html.Div([
                    html.H5("Inverted Hammer", className="font-medium text-green-600"),
                    html.Img(src="assets/patterns/inverted_hammer.png", className="h-24 w-auto"),
                    html.P("Small body at the bottom with a long upper shadow. Bullish reversal at bottom of downtrend.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-green-200 rounded bg-green-50"),
                
                # Hanging Man
                html.Div([
                    html.H5("Hanging Man", className="font-medium text-red-600"),
                    html.Img(src="assets/patterns/hanging_man.png", className="h-24 w-auto"),
                    html.P("Small body at the top with a long lower shadow. Bearish reversal at top of uptrend.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-red-200 rounded bg-red-50"),
                
                # Shooting Star
                html.Div([
                    html.H5("Shooting Star", className="font-medium text-red-600"),
                    html.Img(src="assets/patterns/shooting_star.png", className="h-24 w-auto"),
                    html.P("Small body at the bottom with a long upper shadow. Bearish reversal at top of uptrend.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-red-200 rounded bg-red-50"),
            ], className="grid grid-cols-4 gap-4 mb-6")
        ]),
        
        # Multi-Candle Patterns
        html.Div([
            html.H4("Multi-Candle Patterns", className="text-lg font-semibold mb-2"),
            html.Div([
                # Bullish Engulfing
                html.Div([
                    html.H5("Bullish Engulfing", className="font-medium text-green-600"),
                    html.Img(src="assets/patterns/bullish_engulfing.png", className="h-24 w-auto"),
                    html.P("A bullish candle that completely engulfs the previous bearish candle.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-green-200 rounded bg-green-50"),
                
                # Morning Star
                html.Div([
                    html.H5("Morning Star", className="font-medium text-green-600"),
                    html.Img(src="assets/patterns/morning_star.png", className="h-24 w-auto"),
                    html.P("Three-candle pattern: bearish, small indecision, and bullish candle. Powerful reversal.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-green-200 rounded bg-green-50"),
                
                # Bearish Engulfing
                html.Div([
                    html.H5("Bearish Engulfing", className="font-medium text-red-600"),
                    html.Img(src="assets/patterns/bearish_engulfing.png", className="h-24 w-auto"),
                    html.P("A bearish candle that completely engulfs the previous bullish candle.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-red-200 rounded bg-red-50"),
                
                # Evening Star
                html.Div([
                    html.H5("Evening Star", className="font-medium text-red-600"),
                    html.Img(src="assets/patterns/evening_star.png", className="h-24 w-auto"),
                    html.P("Three-candle pattern: bullish, small indecision, and bearish candle. Powerful reversal.", 
                           className="text-sm mt-2")
                ], className="text-center p-2 border border-red-200 rounded bg-red-50"),
            ], className="grid grid-cols-4 gap-4 mb-4")
        ]),
        
        # Interactive pattern examples that users can click on for more information
        html.Div([
            html.H4("Interactive Examples", className="text-lg font-semibold mb-2"),
            html.P("Click on any pattern to see real examples and detailed explanations.", 
                   className="text-sm italic mb-4"),
            
            dcc.Dropdown(
                id="pattern-example-selector",
                options=[
                    {'label': 'Doji', 'value': 'CDLDOJI'},
                    {'label': 'Hammer', 'value': 'CDLHAMMER'},
                    {'label': 'Shooting Star', 'value': 'CDLSHOOTINGSTAR'},
                    {'label': 'Engulfing Patterns', 'value': 'CDLENGULFING'},
                    {'label': 'Morning/Evening Star', 'value': 'CDLSTAR'},
                    {'label': 'Harami Patterns', 'value': 'CDLHARAMI'},
                    # Add more patterns as needed
                ],
                value='CDLDOJI',
                clearable=False
            ),
            
            html.Div(id="interactive-pattern-example", className="mt-4 p-4 border rounded")
        ])
    ])
    
    return pattern_examples

@app.callback(
    Output("interactive-pattern-example", "children"),
    [Input("pattern-example-selector", "value")]
)
def update_interactive_example(selected_pattern):
    """Display interactive examples based on user selection"""
    if not selected_pattern:
        return html.Div("Please select a pattern", className="text-gray-500")
    
    # Define examples with images and explanations
    example_content = {
        'CDLDOJI': {
            'title': 'Doji Examples',
            'description': 'Doji candles form when the opening and closing prices are virtually equal, creating a very small body. They indicate indecision in the market.',
            'examples': [
                {
                    'title': 'Doji at Market Top',
                    'image': 'assets/examples/doji_top.png',
                    'explanation': 'This Doji formed at the top of an uptrend, signaling potential reversal as buyers lost momentum.'
                },
                {
                    'title': 'Doji at Market Bottom',
                    'image': 'assets/examples/doji_bottom.png',
                    'explanation': 'This Doji formed at the bottom of a downtrend, indicating possible exhaustion of selling pressure.'
                }
            ],
            'trading_tips': [
                'Look for confirmation from the next candle',
                'More significant when appearing after a strong trend',
                'Combine with support/resistance levels for better signals'
            ]
        },
        # Include similar detailed examples for other patterns
    }
    
    # Default content if specific pattern not found
    if selected_pattern not in example_content:
        return html.Div(f"Examples for {selected_pattern.replace('CDL', '')} are being prepared.", 
                        className="text-gray-500")
    
    content = example_content[selected_pattern]
    
    return html.Div([
        html.H3(content['title'], className="text-xl font-semibold mb-2"),
        html.P(content['description'], className="mb-4"),
        
        # Examples with images
        html.Div([
            html.Div([
                html.H4(example['title'], className="font-medium mb-2"),
                html.Div([
                    html.Img(src=example['image'], className="w-full h-auto"),
                    html.P(example['explanation'], className="mt-2 text-sm")
                ], className="border p-3 rounded")
            ], className="w-1/2 p-2")
            for example in content['examples']
        ], className="flex flex-wrap -mx-2 mb-4"),
        
        # Trading tips
        html.Div([
            html.H4("Trading Tips", className="font-medium mb-2"),
            html.Ul([
                html.Li(tip, className="text-sm mb-1") for tip in content['trading_tips']
            ], className="list-disc pl-5")
        ], className="bg-blue-50 p-3 rounded")
    ])








@app.callback(
    [Output('synced-chart-type-container', 'children'),
     Output('all-graph-container', 'children', allow_duplicate=True)],  # Add this line
    [Input('shared-chart-type', 'value')],
    prevent_initial_call=True  # Add this parameter
)
def sync_chart_type_to_all_graph(chart_type):
    """Creates a synchronized version of the chart type radio buttons in All Graph tab"""
    radio_buttons = dcc.RadioItems(
        id='synced-chart-type',
        options=[
            {'label': 'Candlestick', 'value': 'candlestick'},
            {'label': 'Line', 'value': 'line'}
        ],
        value=chart_type,
        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
    )
    
    # Also return a placeholder that will be updated when the real callback runs
    return radio_buttons, dash.no_update

# Add this callback to sync from All Graph tab back to Combination Graph tab
@app.callback(
    Output('shared-chart-type', 'value'),
    [Input('synced-chart-type', 'value')],
    [State('shared-chart-type', 'value')],
    prevent_initial_call=True  # Add this parameter
)
def sync_chart_type_from_all_graph(all_graph_value, current_value):
    """Syncs the chart type selection from All Graph tab back to Combination Graph tab"""
    if all_graph_value is not None and all_graph_value != current_value:
        return all_graph_value
    return dash.no_update




# Fix for line chart and Fibonacci retracement issues in update_combination_graph
@app.callback(
    Output('combination-graph-container', 'children'),
    [Input('shared-chart-type', 'value'),
     Input('ticker-dropdown', 'value'),
     Input('technical-checklist', 'value'),
     Input('start-year-dropdown', 'value'),
     Input('start-month-dropdown', 'value'),
     Input('start-day-dropdown', 'value'),
     Input('end-year-dropdown', 'value'),
     Input('end-month-dropdown', 'value'),
     Input('end-day-dropdown', 'value'),
     Input('data-period-selector', 'value'),
     Input('realtime-interval-selector', 'value'),
     Input('bb-period', 'value'),
     Input('bb-std', 'value'),
     Input('ma-short', 'value'),
     Input('ma-long', 'value'),
     Input('ma-medium', 'value'),
     Input('ma-lines-count', 'value'),
     Input('rsi-period', 'value'),
     Input('rsi-overbought', 'value'),
     Input('rsi-oversold', 'value'),
     Input('macd-fast', 'value'),
     Input('macd-slow', 'value'),
     Input('macd-signal', 'value'),
     Input('adx-period', 'value'),
     Input('adx-threshold', 'value'),
     Input('volume-period', 'value'),
     Input('fibonacci-lookback', 'value'),
     Input('candlestick-confidence', 'value'),
     Input('show-explanations', 'value')]  # Add the new input
)
def update_combination_graph(chart_type, ticker, selected_technicals, start_year, start_month, start_day,
                            end_year, end_month, end_day, selected_period, selected_interval,
                            bb_window, bb_std, ma_short, ma_long, ma_medium, ma_lines_count,
                            rsi_period, rsi_ob, rsi_os,
                            macd_fast, macd_slow, macd_signal,
                            adx_period, adx_threshold,
                            volume_period, fibonacci_lookback, candlestick_confidence, show_explanations):
    
    if not ticker:
        return html.Div("Please select a ticker", className="text-center text-gray-500 mt-10")
    
    try:
        # Load and process data (similar to your existing code)
        filtered_df = load_and_process_data(
            period=selected_period, 
            ticker=ticker,
            interval=selected_interval if selected_period == 'realtime' else '1min'
        )
        
        if filtered_df.empty:
            return html.Div([
                html.H3("No data available for the selected ticker", className="text-red-500"),
                html.P("Please try a different ticker or check data availability.")
            ], className="p-4 bg-gray-50 rounded-lg border border-red-200")
        
        # Convert selected dates to datetime
        try:
            start_date = pd.Timestamp(
                year=start_year, 
                month=start_month, 
                day=start_day
            ).replace(hour=0, minute=0, second=0)
            
            end_date = pd.Timestamp(
                year=end_year,
                month=end_month,
                day=end_day
            ).replace(hour=23, minute=59, second=59)
            
            # Filter data by date range
            filtered_df = filtered_df[
                (filtered_df['Datetime'] >= start_date) &
                (filtered_df['Datetime'] <= end_date)
            ].copy()
            
            if filtered_df.empty:
                return html.Div([
                    html.H3("No data available for the selected date range", className="text-red-500"),
                    html.P("Please try a different date range for this ticker.")
                ], className="p-4 bg-gray-50 rounded-lg border border-red-200")
                
        except Exception as e:
            print(f"Date filtering error: {str(e)}")
            # If date filtering fails, just use all available data
            pass
        
        # Apply technical indicators with custom parameters
        if selected_technicals:
            if 'Bollinger_Signal' in selected_technicals:
                filtered_df = calculate_bollinger_bands(filtered_df, window=bb_window, num_std=bb_std)
            
            if 'MA_Signal' in selected_technicals:
                filtered_df = calculate_ma(
                    filtered_df, 
                    short_period=ma_short, 
                    long_period=ma_long,
                    medium_period=ma_medium,
                    ma_lines=ma_lines_count 
                )
            
            if 'RSI_Signal' in selected_technicals:
                filtered_df = calculate_rsi(filtered_df, period=rsi_period, ob_level=rsi_ob, os_level=rsi_os)
            
            if 'MACD_Signal' in selected_technicals:
                filtered_df = calculate_macd(filtered_df, short_window=macd_fast, long_window=macd_slow, signal_window=macd_signal)
            
            if 'ADX_Signal' in selected_technicals:
                filtered_df = calculate_adx(filtered_df, period=adx_period, threshold=adx_threshold)
            
            if 'Volume_Signal' in selected_technicals:
                filtered_df = calculate_volume(filtered_df, ma_window=volume_period)
            
            if 'Fibonacci_Signal' in selected_technicals:
                filtered_df = calculate_fibonacci_retracement(filtered_df, lookback=fibonacci_lookback)
            
            # For candlestick patterns, enhance the visualization
            if 'Candlestick_Signal' in selected_technicals:
                # Apply candlestick pattern detection
                filtered_df = detect_candlestick_patterns(filtered_df)
                
                # Check if Candlestick_Signal column exists
                if 'Candlestick_Signal' not in filtered_df.columns:
                    # Create the Candlestick_Signal column explicitly if it doesn't exist
                    filtered_df['Candlestick_Signal'] = 'Hold'  # Default value
                    
                    # Define pattern categories
                    bullish_patterns = [
                        'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
                        'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
                        'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
                        'CDLTWEEZERBOTTOM'
                    ]
                    
                    bearish_patterns = [
                        'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
                        'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
                        'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
                        'CDLGRAVESTONEDOJI', 'CDLTWEEZERTOP'
                    ]
                    
                    # Create summary columns for pattern counts
                    filtered_df['bullish_patterns'] = 0
                    filtered_df['bearish_patterns'] = 0
                    filtered_df['neutral_patterns'] = 0
                    
                    # Sum pattern counts for bullish patterns
                    for pattern in bullish_patterns:
                        if pattern in filtered_df.columns:
                            filtered_df['bullish_patterns'] += filtered_df[pattern].fillna(0)
                    
                    # Sum pattern counts for bearish patterns
                    for pattern in bearish_patterns:
                        if pattern in filtered_df.columns:
                            filtered_df['bearish_patterns'] += filtered_df[pattern].fillna(0)
                    
                    # Generate signals based on pattern counts
                    bull_mask = filtered_df['bullish_patterns'] > filtered_df['bearish_patterns']
                    bear_mask = filtered_df['bearish_patterns'] > filtered_df['bullish_patterns']
                    
                    filtered_df.loc[bull_mask, 'Candlestick_Signal'] = 'Buy'
                    filtered_df.loc[bear_mask, 'Candlestick_Signal'] = 'Sell'
                    
                    # Add confidence score (default to 50% if not already calculated)
                    if 'candlestick_confidence' not in filtered_df.columns:
                        filtered_df['candlestick_confidence'] = 50
                        
                        # Calculate confidence based on relative pattern strength
                        max_patterns = max(
                            filtered_df['bullish_patterns'].max() or 0,
                            filtered_df['bearish_patterns'].max() or 0,
                            1  # Avoid division by zero
                        )
                        
                        filtered_df.loc[bull_mask, 'candlestick_confidence'] = (
                            50 + 50 * (filtered_df.loc[bull_mask, 'bullish_patterns'] / max_patterns)
                        ).clip(0, 100)
                        
                        filtered_df.loc[bear_mask, 'candlestick_confidence'] = (
                            50 + 50 * (filtered_df.loc[bear_mask, 'bearish_patterns'] / max_patterns)
                        ).clip(0, 100)
                
                # Apply confidence threshold after ensuring the column exists
                if 'candlestick_confidence' in filtered_df.columns and candlestick_confidence is not None:
                    low_confidence_mask = filtered_df['candlestick_confidence'] < candlestick_confidence
                    filtered_df.loc[low_confidence_mask, 'Candlestick_Signal'] = 'Hold'
        
        # Create the combination graph
        fig = go.Figure()
        
        # Add price data based on chart type
        if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Create candlestick chart
            fig.add_trace(go.Candlestick(
                x=filtered_df['Datetime'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                name=ticker,
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        else:  
            # Create line chart - use the SAME figure object
            fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ))
        
        # Add technical indicator overlays
        if 'Bollinger_Signal' in selected_technicals:
            if all(band in filtered_df.columns for band in ['Upper Band', 'Lower Band', 'Middle Band']):
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Upper Band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Middle Band'],
                    mode='lines',
                    name='Middle Band',
                    line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Lower Band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot'),
                    fill='tonexty', 
                    fillcolor='rgba(173, 216, 230, 0.1)'
                ))
        
        if 'MA_Signal' in selected_technicals:
            if 'short_MA' in filtered_df.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['short_MA'],
                    mode='lines',
                    name=f'Short MA ({ma_short})',
                    line=dict(color='blue', width=1.5)
                ))
            if ma_lines_count == '3' and 'medium_MA' in filtered_df.columns: # Check for 3 lines and column existence
                 fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['medium_MA'],
                    mode='lines',
                    name=f'Medium MA ({ma_medium})',
                    line=dict(color='orange', width=1.5) 
                ))
            if 'long_MA' in filtered_df.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['long_MA'],
                    mode='lines',
                    name=f'Long MA ({ma_long})',
                    line=dict(color='red', width=1.5)
                ))
        
        # Add Fibonacci Retracement levels if selected
        if 'Fibonacci_Signal' in selected_technicals:
            if 'fib_levels' in filtered_df.columns and not filtered_df['fib_levels'].isnull().all():
                # Process fib_levels data in chunks for more accurate historical representation
                retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                colors = ['#FFB347', '#FFD700', '#90EE90', '#87CEEB', '#FF69B4']
                
                # Process data in chunks to avoid too many line segments
                for i in range(0, len(filtered_df), max(1, min(fibonacci_lookback, len(filtered_df) // 10))):
                    chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                    if chunk.empty or 'fib_levels' not in chunk.columns:
                        continue
                        
                    # Find first valid fib_levels in this chunk
                    valid_fib_levels = None
                    for idx, row in chunk.iterrows():
                        if isinstance(row['fib_levels'], list) and len(row['fib_levels']) > 0:
                            valid_fib_levels = row['fib_levels']
                            break
                            
                    if valid_fib_levels is None:
                        continue
                        
                    # Draw Fibonacci levels for this chunk
                    for level_idx, (level, color) in enumerate(zip(retracement_levels, colors)):
                        if level_idx < len(valid_fib_levels) and valid_fib_levels[level_idx] is not None:
                            fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[valid_fib_levels[level_idx]] * len(chunk),
                                mode='lines',
                                name=f'Fib {level*100:.1f}%',
                                line=dict(color=color, dash='dash'),
                                showlegend=(i == 0)  # Only show in legend once
                            ))
                    
                # Add high/low from fib calculation
                if 'fib_high' in filtered_df.columns and 'fib_low' in filtered_df.columns:
                    for i in range(0, len(filtered_df), max(1, min(fibonacci_lookback, len(filtered_df) // 10))):
                        chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                        if chunk.empty:
                            continue
                            
                        high_val = chunk['fib_high'].iloc[0] if pd.notna(chunk['fib_high'].iloc[0]) else None
                        low_val = chunk['fib_low'].iloc[0] if pd.notna(chunk['fib_low'].iloc[0]) else None
                        
                        if high_val is not None:
                            fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[high_val] * len(chunk),
                                mode='lines',
                                name='Fib High',
                                line=dict(color='red', dash='dot', width=1),
                                showlegend=(i == 0)
                            ))
                        
                        if low_val is not None:
                            fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[low_val] * len(chunk),
                                mode='lines',
                                name='Fib Low',
                                line=dict(color='green', dash='dot', width=1),
                                showlegend=(i == 0)
                            ))
        
        # Add buy/sell signals
        if 'signal_akhir' in filtered_df.columns:
            buy_mask = filtered_df['signal_akhir'] == 'Buy'
            sell_mask = filtered_df['signal_akhir'] == 'Sell'
            
            if buy_mask.any():
                fig.add_trace(go.Scatter(
                    x=filtered_df.loc[buy_mask, 'Datetime'],
                    y=filtered_df.loc[buy_mask, 'Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', symbol='triangle-up', size=10)
                ))
            
            if sell_mask.any():
                fig.add_trace(go.Scatter(
                    x=filtered_df.loc[sell_mask, 'Datetime'],
                    y=filtered_df.loc[sell_mask, 'Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', symbol='triangle-down', size=10)
                ))
        
        # Create subplots for separate indicators
        if any(indicator in selected_technicals for indicator in ['RSI_Signal', 'MACD_Signal', 'ADX_Signal', 'Volume_Signal']):
            # Count how many separate indicators we have
            subplot_count = 1  # Start with 1 for main price chart
            
            # Add additional subplots based on selected indicators
            has_rsi = 'RSI_Signal' in selected_technicals and 'RSI' in filtered_df.columns
            has_macd = 'MACD_Signal' in selected_technicals and all(col in filtered_df.columns for col in ['MACD', 'Signal_Line'])
            has_adx = 'ADX_Signal' in selected_technicals and all(col in filtered_df.columns for col in ['ADX', '+DI', '-DI'])
            has_volume = 'Volume_Signal' in selected_technicals and 'Volume' in filtered_df.columns
            
            if has_rsi: subplot_count += 1
            if has_macd: subplot_count += 1
            if has_adx: subplot_count += 1
            if has_volume: subplot_count += 1
            
            # Create subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=subplot_count,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1]
            )
            
            # Re-add main price chart to the first subplot
            if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig.add_trace(
                    go.Candlestick(
                        x=filtered_df['Datetime'],
                        open=filtered_df['Open'],
                        high=filtered_df['High'],
                        low=filtered_df['Low'],
                        close=filtered_df['Close'],
                        name=ticker,
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ),
                    row=1, col=1
                )
            else:  # Line chart
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )
            
            # Re-add technical overlays to the main price chart
            if 'Bollinger_Signal' in selected_technicals:
                if all(band in filtered_df.columns for band in ['Upper Band', 'Lower Band', 'Middle Band']):
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['Upper Band'],
                            mode='lines',
                            name='Upper Band',
                            line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['Middle Band'],
                            mode='lines',
                            name='Middle Band',
                            line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['Lower Band'],
                            mode='lines',
                            name='Lower Band',
                            line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot')
                        ),
                        row=1, col=1
                    )
            
            if 'MA_Signal' in selected_technicals:
                if 'short_MA' in filtered_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['short_MA'],
                            mode='lines',
                            name=f'Short MA ({ma_short})',
                            line=dict(color='blue', width=1.5)
                        ),
                        row=1, col=1
                    )
                if ma_lines_count == '3' and 'medium_MA' in filtered_df.columns: # Check for 3 lines and column existence
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['medium_MA'],
                            mode='lines',
                            name=f'Medium MA ({ma_medium})',
                            line=dict(color='orange', width=1.5) 
                        ),
                        row=1, col=1
                    )
                if 'long_MA' in filtered_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['long_MA'],
                            mode='lines',
                            name=f'Long MA ({ma_long})',
                            line=dict(color='red', width=1.5)
                        ),
                        row=1, col=1
                    )
            
            # Add Fibonacci Retracement to main chart
            if 'Fibonacci_Signal' in selected_technicals:
                if 'fib_levels' in filtered_df.columns and not filtered_df['fib_levels'].isnull().all():
                    # Process fib_levels data in chunks for more accurate historical representation
                    retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    colors = ['#FFB347', '#FFD700', '#90EE90', '#87CEEB', '#FF69B4']
                    
                    # Process data in chunks to avoid too many line segments
                    for i in range(0, len(filtered_df), max(1, min(fibonacci_lookback, len(filtered_df) // 10))):
                        chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                        if chunk.empty or 'fib_levels' not in chunk.columns:
                            continue
                            
                        # Find first valid fib_levels in this chunk
                        valid_fib_levels = None
                        for idx, row in chunk.iterrows():
                            if isinstance(row['fib_levels'], list) and len(row['fib_levels']) > 0:
                                valid_fib_levels = row['fib_levels']
                                break
                                
                        if valid_fib_levels is None:
                            continue
                            
                        # Draw Fibonacci levels for this chunk
                        for level_idx, (level, color) in enumerate(zip(retracement_levels, colors)):
                            if level_idx < len(valid_fib_levels) and valid_fib_levels[level_idx] is not None:
                                fig.add_trace(go.Scatter(
                                    x=chunk['Datetime'],
                                    y=[valid_fib_levels[level_idx]] * len(chunk),
                                    mode='lines',
                                    name=f'Fib {level*100:.1f}%',
                                    line=dict(color=color, dash='dash'),
                                    showlegend=(i == 0)  # Only show in legend once
                                ))
                        
                    # Add high/low from fib calculation
                    if 'fib_high' in filtered_df.columns and 'fib_low' in filtered_df.columns:
                        for i in range(0, len(filtered_df), max(1, min(fibonacci_lookback, len(filtered_df) // 10))):
                            chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                            if chunk.empty:
                                continue
                                
                            high_val = chunk['fib_high'].iloc[0] if pd.notna(chunk['fib_high'].iloc[0]) else None
                            low_val = chunk['fib_low'].iloc[0] if pd.notna(chunk['fib_low'].iloc[0]) else None
                            
                            if high_val is not None:
                                fig.add_trace(go.Scatter(
                                    x=chunk['Datetime'],
                                    y=[high_val] * len(chunk),
                                    mode='lines',
                                    name='Fib High',
                                    line=dict(color='red', dash='dot', width=1),
                                    showlegend=(i == 0)
                                ))
                            
                            if low_val is not None:
                                fig.add_trace(go.Scatter(
                                    x=chunk['Datetime'],
                                    y=[low_val] * len(chunk),
                                    mode='lines',
                                    name='Fib Low',
                                    line=dict(color='green', dash='dot', width=1),
                                    showlegend=(i == 0)
                                ))
            
            # Add buy/sell signals to the main price chart
            if 'signal_akhir' in filtered_df.columns:
                buy_mask = filtered_df['signal_akhir'] == 'Buy'
                sell_mask = filtered_df['signal_akhir'] == 'Sell'
                
                if buy_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df.loc[buy_mask, 'Datetime'],
                            y=filtered_df.loc[buy_mask, 'Close'],
                            mode='markers',
                            name='Buy Signal',
                            marker=dict(color='green', symbol='triangle-up', size=10)
                        ),
                        row=1, col=1
                    )
                
                if sell_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df.loc[sell_mask, 'Datetime'],
                            y=filtered_df.loc[sell_mask, 'Close'],
                            mode='markers',
                            name='Sell Signal',
                            marker=dict(color='red', symbol='triangle-down', size=10)
                        ),
                        row=1, col=1
                    )
            
            # Add separate indicators to dedicated subplots
            current_row = 2
            
            # Add RSI subplot
            if has_rsi:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                # Add overbought/oversold lines
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=[rsi_ob] * len(filtered_df),
                        mode='lines',
                        name=f'Overbought ({rsi_ob})',
                        line=dict(color='red', dash='dash', width=1)
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=[rsi_os] * len(filtered_df),
                        mode='lines',
                        name=f'Oversold ({rsi_os})',
                        line=dict(color='green', dash='dash', width=1)
                    ),
                    row=current_row, col=1
                )
                
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1
            
            # Add MACD subplot
            if has_macd:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['Signal_Line'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='red', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                if 'MACD_Hist' in filtered_df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=filtered_df['Datetime'],
                            y=filtered_df['MACD_Hist'],
                            name='MACD Histogram',
                            marker_color=filtered_df['MACD_Hist'].apply(
                                lambda x: 'rgba(0, 255, 0, 0.5)' if x > 0 else 'rgba(255, 0, 0, 0.5)'
                            )
                        ),
                        row=current_row, col=1
                    )
                
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                current_row += 1
            
            # Add ADX subplot
            if has_adx:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['ADX'],
                        mode='lines',
                        name='ADX',
                        line=dict(color='purple', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['+DI'],
                        mode='lines',
                        name='+DI',
                        line=dict(color='green', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=filtered_df['-DI'],
                        mode='lines',
                        name='-DI',
                        line=dict(color='red', width=1.5)
                    ),
                    row=current_row, col=1
                )
                
                # Add threshold line
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Datetime'],
                        y=[adx_threshold] * len(filtered_df),
                        mode='lines',
                        name=f'ADX Threshold ({adx_threshold})',
                        line=dict(color='gray', dash='dash', width=1)
                    ),
                    row=current_row, col=1
                )
                
                fig.update_yaxes(title_text="ADX", row=current_row, col=1)
                current_row += 1
            
            # Add Volume subplot
            if has_volume:
                fig.add_trace(
                    go.Bar(
                        x=filtered_df['Datetime'],
                        y=filtered_df['Volume'],
                        name='Volume',
                        marker_color='blue'
                    ),
                    row=current_row, col=1
                )
                
                if 'VMA' in filtered_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Datetime'],
                            y=filtered_df['VMA'],
                            mode='lines',
                            name='Volume MA',
                            line=dict(color='orange', width=1.5)
                        ),
                        row=current_row, col=1
                    )
                
                fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        
        # Update layout for the final figure
        fig.update_layout(
            title=f"{ticker} - Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            height=max(600, 200 * subplot_count if 'subplot_count' in locals() else 600),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update x-axis settings
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ]
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            # html.Div(indicator_explanations['combination'] if show_explanations == 'yes' else '')
        ])
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in update_combination_graph: {traceback_str}")
        return html.Div([
            html.H3("Error creating combination chart", className="text-red-500"),
            html.P(f"Error: {str(e)}", className="mb-2"),
            html.P("Please check your input parameters and try again.", className="italic")
        ], className="p-4 bg-gray-50 rounded-lg border border-red-200")




# def create_candlestick_analysis_report(filtered_df):
#     """Generate a detailed report of candlestick patterns found in the data"""
    
#     # Get all pattern columns
#     pattern_columns = [col for col in filtered_df.columns if col.startswith('CDL')]
    
#     # Create a summary of detected patterns
#     detected_patterns = {}
#     for pattern in pattern_columns:
#         if pattern not in ['Candlestick_Signal', 'candlestick_confidence', 
#                           'bullish_patterns', 'bearish_patterns', 'neutral_patterns']:
#             count = filtered_df[pattern].sum()
#             if count > 0:
#                 detected_patterns[pattern] = {
#                     'count': count,
#                     'dates': filtered_df.loc[filtered_df[pattern] == 1, 'Datetime'].tolist(),
#                     'description': get_pattern_description(pattern)
#                 }
    
#     # No patterns detected
#     if not detected_patterns:
#         return html.Div("No specific candlestick patterns detected in the selected time range.",
#                         className="text-gray-500 italic")
    
#     # Create the report components
#     report_items = []
    
#     # Group patterns by type
#     bullish_patterns = {k: v for k, v in detected_patterns.items() 
#                        if k in ['CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
#                                'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
#                                'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING']}
    
#     bearish_patterns = {k: v for k, v in detected_patterns.items() 
#                        if k in ['CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
#                                'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
#                                'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
#                                'CDLGRAVESTONEDOJI']}
    
#     neutral_patterns = {k: v for k, v in detected_patterns.items() 
#                        if k in ['CDLDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU', 'CDLTAKURI', 'CDLRICKSHAWMAN']}
    
#     # Create sections for each type
#     if bullish_patterns:
#         report_items.append(html.H3("Bullish Patterns", className="text-xl font-bold text-green-600 mt-4 mb-2"))
#         for pattern, data in bullish_patterns.items():
#             # Create mini-chart for this pattern
#             pattern_df = filtered_df[filtered_df[pattern] == 1]
#             if not pattern_df.empty:
#                 # Take the first occurrence for the example
#                 example_date = pattern_df['Datetime'].iloc[0]
#                 # Get 5 days before and after for context
#                 start_idx = max(0, filtered_df[filtered_df['Datetime'] == example_date].index[0] - 5)
#                 end_idx = min(len(filtered_df) - 1, filtered_df[filtered_df['Datetime'] == example_date].index[0] + 5)
#                 example_df = filtered_df.iloc[start_idx:end_idx+1]
                
#                 # Create example figure
#                 example_fig = go.Figure()
#                 example_fig.add_trace(go.Candlestick(
#                     x=example_df['Datetime'],
#                     open=example_df['Open'],
#                     high=example_df['High'],
#                     low=example_df['Low'],
#                     close=example_df['Close'],
#                     name='Price'
#                 ))
                
#                 # Highlight the pattern candle(s)
#                 pattern_idx = example_df[example_df[pattern] == 1].index
#                 for idx in pattern_idx:
#                     example_fig.add_shape(
#                         type="rect",
#                         x0=example_df.loc[idx, 'Datetime'],
#                         y0=example_df.loc[idx, 'Low'] * 0.99,
#                         x1=example_df.loc[idx, 'Datetime'],
#                         y1=example_df.loc[idx, 'High'] * 1.01,
#                         line=dict(width=2, color="green"),
#                         fillcolor="rgba(0, 255, 0, 0.2)"
#                     )
                
#                 example_fig.update_layout(
#                     title=f"Example of {pattern.replace('CDL', '')}",
#                     xaxis_title="Date",
#                     yaxis_title="Price",
#                     height=300,
#                     width=400,
#                     showlegend=False,
#                     margin=dict(l=40, r=40, t=40, b=40)
#                 )
                
#                 # Add the pattern details
#                 report_items.append(html.Div([
#                     html.H4(pattern.replace('CDL', ''), className="text-lg font-semibold text-green-700"),
#                     html.Div([
#                         html.Div([
#                             html.P(f"Occurrences: {data['count']}", className="font-medium"),
#                             html.P(data['description'], className="mt-2 text-sm"),
#                             html.P("Dates Detected:", className="mt-2 font-medium"),
#                             html.Ul([
#                                 html.Li(date.strftime("%Y-%m-%d %H:%M") if hasattr(date, 'strftime') else str(date), 
#                                        className="text-sm")
#                                 for date in data['dates'][:5]  # Show first 5 occurrences
#                             ], className="list-disc ml-5"),
#                             html.P(f"...and {len(data['dates']) - 5} more" if len(data['dates']) > 5 else "",
#                                    className="text-sm text-gray-500 mt-1")
#                         ], className="w-2/3"),
#                         html.Div([
#                             dcc.Graph(figure=example_fig)
#                         ], className="w-1/3")
#                     ], className="flex flex-row")
#                 ], className="mb-5 p-4 border border-green-200 rounded-lg bg-green-50"))
    
#     # Add bearish patterns with charts
#     if bearish_patterns:
#         report_items.append(html.H3("Bearish Patterns", className="text-xl font-bold text-red-600 mt-4 mb-2"))
#         for pattern, data in bearish_patterns.items():
#             # Create mini-chart for this pattern
#             pattern_df = filtered_df[filtered_df[pattern] == 1]
#             if not pattern_df.empty:
#                 # Take the first occurrence for the example
#                 example_date = pattern_df['Datetime'].iloc[0]
#                 # Get 5 days before and after for context
#                 start_idx = max(0, filtered_df[filtered_df['Datetime'] == example_date].index[0] - 5)
#                 end_idx = min(len(filtered_df) - 1, filtered_df[filtered_df['Datetime'] == example_date].index[0] + 5)
#                 example_df = filtered_df.iloc[start_idx:end_idx+1]
                
#                 # Create example figure
#                 example_fig = go.Figure()
#                 example_fig.add_trace(go.Candlestick(
#                     x=example_df['Datetime'],
#                     open=example_df['Open'],
#                     high=example_df['High'],
#                     low=example_df['Low'],
#                     close=example_df['Close'],
#                     name='Price'
#                 ))
                
#                 # Highlight the pattern candle(s)
#                 pattern_idx = example_df[example_df[pattern] == 1].index
#                 for idx in pattern_idx:
#                     example_fig.add_shape(
#                         type="rect",
#                         x0=example_df.loc[idx, 'Datetime'],
#                         y0=example_df.loc[idx, 'Low'] * 0.99,
#                         x1=example_df.loc[idx, 'Datetime'],
#                         y1=example_df.loc[idx, 'High'] * 1.01,
#                         line=dict(width=2, color="red"),
#                         fillcolor="rgba(255, 0, 0, 0.2)"
#                     )
                
#                 example_fig.update_layout(
#                     title=f"Example of {pattern.replace('CDL', '')}",
#                     xaxis_title="Date",
#                     yaxis_title="Price",
#                     height=300,
#                     width=400,
#                     showlegend=False,
#                     margin=dict(l=40, r=40, t=40, b=40)
#                 )
                
#                 # Add the pattern details
#                 report_items.append(html.Div([
#                     html.H4(pattern.replace('CDL', ''), className="text-lg font-semibold text-red-700"),
#                     html.Div([
#                         html.Div([
#                             html.P(f"Occurrences: {data['count']}", className="font-medium"),
#                             html.P(data['description'], className="mt-2 text-sm"),
#                             html.P("Dates Detected:", className="mt-2 font-medium"),
#                             html.Ul([
#                                 html.Li(date.strftime("%Y-%m-%d %H:%M") if hasattr(date, 'strftime') else str(date), 
#                                        className="text-sm")
#                                 for date in data['dates'][:5]  # Show first 5 occurrences
#                             ], className="list-disc ml-5"),
#                             html.P(f"...and {len(data['dates']) - 5} more" if len(data['dates']) > 5 else "",
#                                    className="text-sm text-gray-500 mt-1")
#                         ], className="w-2/3"),
#                         html.Div([
#                             dcc.Graph(figure=example_fig)
#                         ], className="w-1/3")
#                     ], className="flex flex-row")
#                 ], className="mb-5 p-4 border border-red-200 rounded-lg bg-red-50"))

#     # Add neutral patterns with charts
#     if neutral_patterns:
#         report_items.append(html.H3("Neutral Patterns", className="text-xl font-bold text-blue-600 mt-4 mb-2"))
#         for pattern, data in neutral_patterns.items():
#             # Create mini-chart for this pattern
#             pattern_df = filtered_df[filtered_df[pattern] == 1]
#             if not pattern_df.empty:
#                 # Take the first occurrence for the example
#                 example_date = pattern_df['Datetime'].iloc[0]
#                 # Get 5 days before and after for context
#                 start_idx = max(0, filtered_df[filtered_df['Datetime'] == example_date].index[0] - 5)
#                 end_idx = min(len(filtered_df) - 1, filtered_df[filtered_df['Datetime'] == example_date].index[0] + 5)
#                 example_df = filtered_df.iloc[start_idx:end_idx+1]
                
#                 # Create example figure
#                 example_fig = go.Figure()
#                 example_fig.add_trace(go.Candlestick(
#                     x=example_df['Datetime'],
#                     open=example_df['Open'],
#                     high=example_df['High'],
#                     low=example_df['Low'],
#                     close=example_df['Close'],
#                     name='Price'
#                 ))
                
#                 # Highlight the pattern candle(s)
#                 pattern_idx = example_df[example_df[pattern] == 1].index
#                 for idx in pattern_idx:
#                     example_fig.add_shape(
#                         type="rect",
#                         x0=example_df.loc[idx, 'Datetime'],
#                         y0=example_df.loc[idx, 'Low'] * 0.99,
#                         x1=example_df.loc[idx, 'Datetime'],
#                         y1=example_df.loc[idx, 'High'] * 1.01,
#                         line=dict(width=2, color="blue"),
#                         fillcolor="rgba(0, 0, 255, 0.2)"
#                     )
                
#                 example_fig.update_layout(
#                     title=f"Example of {pattern.replace('CDL', '')}",
#                     xaxis_title="Date",
#                     yaxis_title="Price",
#                     height=300,
#                     width=400,
#                     showlegend=False,
#                     margin=dict(l=40, r=40, t=40, b=40)
#                 )
                
#                 # Add the pattern details
#                 report_items.append(html.Div([
#                     html.H4(pattern.replace('CDL', ''), className="text-lg font-semibold text-blue-700"),
#                     html.Div([
#                         html.Div([
#                             html.P(f"Occurrences: {data['count']}", className="font-medium"),
#                             html.P(data['description'], className="mt-2 text-sm"),
#                             html.P("Dates Detected:", className="mt-2 font-medium"),
#                             html.Ul([
#                                 html.Li(date.strftime("%Y-%m-%d %H:%M") if hasattr(date, 'strftime') else str(date), 
#                                        className="text-sm")
#                                 for date in data['dates'][:5]  # Show first 5 occurrences
#                             ], className="list-disc ml-5"),
#                             html.P(f"...and {len(data['dates']) - 5} more" if len(data['dates']) > 5 else "",
#                                    className="text-sm text-gray-500 mt-1")
#                         ], className="w-2/3"),
#                         html.Div([
#                             dcc.Graph(figure=example_fig)
#                         ], className="w-1/3")
#                     ], className="flex flex-row")
#                 ], className="mb-5 p-4 border border-blue-200 rounded-lg bg-blue-50"))
    
#     return html.Div(report_items, className="mt-4")




# Add a new callback for the all-graph-container
@app.callback(
    Output('all-graph-container', 'children'),
    [Input('shared-chart-type', 'value'),
     Input('ticker-dropdown', 'value'),
     Input('technical-checklist', 'value'),
     Input('start-year-dropdown', 'value'),
     Input('start-month-dropdown', 'value'),
     Input('start-day-dropdown', 'value'),
     Input('end-year-dropdown', 'value'),
     Input('end-month-dropdown', 'value'),
     Input('end-day-dropdown', 'value'),
     Input('data-period-selector', 'value'),
     Input('realtime-interval-selector', 'value'),
     Input('bb-period', 'value'),
     Input('bb-std', 'value'),
     Input('ma-short', 'value'),
     Input('ma-long', 'value'),
     Input('ma-medium', 'value'),
     Input('ma-lines-count', 'value'),
     Input('rsi-period', 'value'),
     Input('rsi-overbought', 'value'),
     Input('rsi-oversold', 'value'),
     Input('macd-fast', 'value'),
     Input('macd-slow', 'value'),
     Input('macd-signal', 'value'),
     Input('adx-period', 'value'),
     Input('adx-threshold', 'value'),
     Input('volume-period', 'value'),
     Input('fibonacci-lookback', 'value'),
     Input('show-explanations', 'value')]  # Add the new input
)
def update_all_graphs(chart_type, ticker, selected_technicals, 
                     start_year, start_month, start_day,
                     end_year, end_month, end_day, 
                     selected_period, selected_interval,
                     bb_window, bb_std,
                     ma_short, ma_long, ma_medium, ma_lines_count,
                     rsi_period, rsi_ob, rsi_os,
                     macd_fast, macd_slow, macd_signal,
                     adx_period, adx_threshold,
                     volume_period,
                     fibonacci_lookback,
                     show_explanations):
    
    if not ticker or not selected_technicals:
        return html.Div("Please select a ticker and at least one technical indicator")
    
    try:
        # Load data and filter by date range
        filtered_df = load_and_process_data(
            period=selected_period,
            ticker=ticker,
            interval=selected_interval if selected_period == 'realtime' else '1min'
        )
        
        if filtered_df.empty:
            return html.Div("No data available for the selected ticker")
        
        # Filter by date range
        try:
            start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
            end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
            filtered_df = filtered_df[(filtered_df['Datetime'] >= start_date) & 
                                     (filtered_df['Datetime'] <= end_date)]
        except Exception as e:
            print(f"Date filtering error: {str(e)}")
        
        # Calculate indicators with custom parameters
        if 'Bollinger_Signal' in selected_technicals:
            filtered_df = calculate_bollinger_bands(filtered_df, window=bb_window, num_std=bb_std)
        
        if 'MA_Signal' in selected_technicals:
        # PERBAIKAN: Pass semua parameter MA
            filtered_df = calculate_ma(
                filtered_df, 
                short_period=ma_short, 
                long_period=ma_long,
                medium_period=ma_medium, 
                ma_lines=ma_lines_count   
            )
        
        if 'RSI_Signal' in selected_technicals:
            filtered_df = calculate_rsi(filtered_df, period=rsi_period, ob_level=rsi_ob, os_level=rsi_os)
        
        if 'MACD_Signal' in selected_technicals:
            filtered_df = calculate_macd(filtered_df, short_window=macd_fast, long_window=macd_slow, signal_window=macd_signal)
        
        if 'ADX_Signal' in selected_technicals:
            filtered_df = calculate_adx(filtered_df, period=adx_period, threshold=adx_threshold)
        
        if 'Volume_Signal' in selected_technicals:
            filtered_df = calculate_volume(filtered_df, ma_window=volume_period)
        
        if 'Fibonacci_Signal' in selected_technicals:
            filtered_df = calculate_fibonacci_retracement(filtered_df, lookback=fibonacci_lookback)
        
        # Initialize list to store all the charts
        all_charts = []
        
        # Create main price chart (always first)
        main_fig = go.Figure()
        
        # Add price data based on chart type
        if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            main_fig.add_trace(go.Candlestick(
                x=filtered_df['Datetime'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                name=ticker,
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
        else:  # Explicitly use line chart
            main_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ))
        
        # Add buy/sell signals to main chart
        if 'signal_akhir' in filtered_df.columns:
            buy_mask = filtered_df['signal_akhir'] == 'Buy'
            sell_mask = filtered_df['signal_akhir'] == 'Sell'
            
            if buy_mask.any():
                main_fig.add_trace(go.Scatter(
                    x=filtered_df.loc[buy_mask, 'Datetime'],
                    y=filtered_df.loc[buy_mask, 'Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', symbol='triangle-up', size=10)
                ))
            
            if sell_mask.any():
                main_fig.add_trace(go.Scatter(
                    x=filtered_df.loc[sell_mask, 'Datetime'],
                    y=filtered_df.loc[sell_mask, 'Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', symbol='triangle-down', size=10)
                ))
        
        main_fig.update_layout(
            title=f"{ticker} - Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update x-axis settings for all charts
        main_fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[dict(bounds=["sat", "mon"])]  # Hide weekends
        )
        
        all_charts.append(dcc.Graph(figure=main_fig))
        
        # Create individual charts for each technical indicator
        
        # 1. Bollinger Bands Chart (always show first)
        if 'Bollinger_Signal' in selected_technicals:
            bb_fig = go.Figure()
            
            # Add price data
            if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                bb_fig.add_trace(go.Candlestick(
                    x=filtered_df['Datetime'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name=ticker,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
            else:
                bb_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=2)
                ))
            
            # Add Bollinger Bands
            if all(band in filtered_df.columns for band in ['Upper Band', 'Lower Band', 'Middle Band']):
                bb_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Upper Band'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Middle Band'],
                    mode='lines',
                    name='Middle Band',
                    line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Lower Band'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot'),
                    fill='tonexty', 
                    fillcolor='rgba(173, 216, 230, 0.1)'
                ))
            
            # Add signals
            if 'Bollinger_Signal' in filtered_df.columns:
                buy_mask = filtered_df['Bollinger_Signal'] == 'Buy'
                sell_mask = filtered_df['Bollinger_Signal'] == 'Sell'
                
                if buy_mask.any():
                    bb_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[buy_mask, 'Datetime'],
                        y=filtered_df.loc[buy_mask, 'Close'],
                        mode='markers',
                        name='BB Buy Signal',
                        marker=dict(color='green', symbol='triangle-up', size=10)
                    ))
                
                if sell_mask.any():
                    bb_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[sell_mask, 'Datetime'],
                        y=filtered_df.loc[sell_mask, 'Close'],
                        mode='markers',
                        name='BB Sell Signal',
                        marker=dict(color='red', symbol='triangle-down', size=10)
                    ))
            
            bb_fig.update_layout(
                title=f"Bollinger Bands (window: {bb_window}, std: {bb_std})",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            bb_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=bb_fig))
        
        # 2. Moving Average Chart
        if 'MA_Signal' in selected_technicals:
            ma_fig = go.Figure()
            
            # Add price data
            if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                ma_fig.add_trace(go.Candlestick(
                    x=filtered_df['Datetime'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name=ticker,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
            else:
                ma_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=2)
                ))
            
            # Add Moving Averages
            if 'short_MA' in filtered_df.columns:
                ma_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['short_MA'],
                    mode='lines',
                    name=f'Fast MA ({ma_short})',
                    line=dict(color='blue', width=1.5)
                ))
            
            # TAMBAHKAN: Medium MA untuk mode 3 lines
            if ma_lines_count == '3' and 'medium_MA' in filtered_df.columns:
                ma_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['medium_MA'],
                    mode='lines',
                    name=f'Medium MA ({ma_medium})',
                    line=dict(color='orange', width=1.5)
                ))
            
            if 'long_MA' in filtered_df.columns:
                ma_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['long_MA'],
                    mode='lines',
                    name=f'Slow MA ({ma_long})',
                    line=dict(color='red', width=1.5)
                ))
            
            # Update title berdasarkan mode
            if ma_lines_count == '3':
                title_text = f"Moving Averages (3 Lines) - Fast: {ma_short}, Medium: {ma_medium}, Slow: {ma_long}"
            else:
                title_text = f"Moving Averages (2 Lines) - Fast: {ma_short}, Slow: {ma_long}"
            
            ma_fig.update_layout(
                title=title_text,
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            ma_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=ma_fig))


        
        # 3. RSI Chart
        if 'RSI_Signal' in selected_technicals and 'RSI' in filtered_df.columns:
            rsi_fig = go.Figure()
            
            # Add RSI line
            rsi_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # Add overbought/oversold lines
            rsi_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=[rsi_ob] * len(filtered_df),
                mode='lines',
                name=f'Overbought ({rsi_ob})',
                line=dict(color='red', dash='dash', width=1)
            ))
            
            rsi_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=[rsi_os] * len(filtered_df),
                mode='lines',
                name=f'Oversold ({rsi_os})',
                line=dict(color='green', dash='dash', width=1)
            ))
            
            # Add midline at 50
            rsi_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=[50] * len(filtered_df),
                mode='lines',
                name='Midline (50)',
                line=dict(color='gray', dash='dot', width=1)
            ))
            
            # Add signals
            if 'RSI_Signal' in filtered_df.columns:
                buy_mask = filtered_df['RSI_Signal'] == 'Buy'
                sell_mask = filtered_df['RSI_Signal'] == 'Sell'
                
                if buy_mask.any():
                    # Mark the buy signals on the RSI line
                    rsi_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[buy_mask, 'Datetime'],
                        y=filtered_df.loc[buy_mask, 'RSI'],
                        mode='markers',
                        name='RSI Buy Signal',
                        marker=dict(color='green', symbol='triangle-up', size=10)
                    ))
                
                if sell_mask.any():
                    # Mark the sell signals on the RSI line
                    rsi_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[sell_mask, 'Datetime'],
                        y=filtered_df.loc[sell_mask, 'RSI'],
                        mode='markers',
                        name='RSI Sell Signal',
                        marker=dict(color='red', symbol='triangle-down', size=10)
                    ))
            
            rsi_fig.update_layout(
                title=f"RSI (period: {rsi_period}, ob: {rsi_ob}, os: {rsi_os})",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100]),  # RSI is always between 0 and 100
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            rsi_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=rsi_fig))
        
        # 4. MACD Chart
        if 'MACD_Signal' in selected_technicals and all(col in filtered_df.columns for col in ['MACD', 'Signal_Line']):
            macd_fig = go.Figure()
            
            # Add MACD components
            macd_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ))
            
            macd_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['Signal_Line'],
                mode='lines',
                name='Signal Line',
                line=dict(color='red', width=2)
            ))
            
            if 'MACD_Hist' in filtered_df.columns:
                macd_fig.add_trace(go.Bar(
                    x=filtered_df['Datetime'],
                    y=filtered_df['MACD_Hist'],
                    name='MACD Histogram',
                    marker_color=filtered_df['MACD_Hist'].apply(
                        lambda x: 'rgba(0, 255, 0, 0.5)' if x > 0 else 'rgba(255, 0, 0, 0.5)'
                    )
                ))
            
            # Add zero line
            macd_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=[0] * len(filtered_df),
                mode='lines',
                name='Zero Line',
                line=dict(color='gray', dash='dot', width=1)
            ))
            
            # Add signals
            if 'MACD_Signal' in filtered_df.columns:
                # We would normally mark buy/sell signals at the MACD line crossing points
                # For simplicity, we'll just show them as markers on the MACD line
                buy_mask = filtered_df['MACD_Signal'] == 'Buy'
                sell_mask = filtered_df['MACD_Signal'] == 'Sell'
                
                if buy_mask.any():
                    macd_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[buy_mask, 'Datetime'],
                        y=filtered_df.loc[buy_mask, 'MACD'],
                        mode='markers',
                        name='MACD Buy Signal',
                        marker=dict(color='green', symbol='triangle-up', size=10)
                    ))
                
                if sell_mask.any():
                    macd_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[sell_mask, 'Datetime'],
                        y=filtered_df.loc[sell_mask, 'MACD'],
                        mode='markers',
                        name='MACD Sell Signal',
                        marker=dict(color='red', symbol='triangle-down', size=10)
                    ))
            
            macd_fig.update_layout(
                title=f"MACD (fast: {macd_fast}, slow: {macd_slow}, signal: {macd_signal})",
                xaxis_title="Date",
                yaxis_title="MACD",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            macd_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=macd_fig))
        
        # 5. ADX Chart
        if 'ADX_Signal' in selected_technicals and all(col in filtered_df.columns for col in ['ADX', '+DI', '-DI']):
            adx_fig = go.Figure()
            
            # Add ADX components
            adx_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['ADX'],
                mode='lines',
                name='ADX',
                line=dict(color='purple', width=2)
            ))
            
            adx_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['+DI'],
                mode='lines',
                name='+DI',
                line=dict(color='green', width=2)
            ))
            
            adx_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['-DI'],
                mode='lines',
                name='-DI',
                line=dict(color='red', width=2)
            ))
            
            # Add threshold line
            adx_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=[adx_threshold] * len(filtered_df),
                mode='lines',
                name=f'ADX Threshold ({adx_threshold})',
                line=dict(color='gray', dash='dash', width=1)
            ))
            
            # Add signals
            if 'ADX_Signal' in filtered_df.columns:
                buy_mask = filtered_df['ADX_Signal'] == 'Buy'
                sell_mask = filtered_df['ADX_Signal'] == 'Sell'
                
                if buy_mask.any():
                    adx_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[buy_mask, 'Datetime'],
                        y=filtered_df.loc[buy_mask, 'ADX'],
                        mode='markers',
                        name='ADX Buy Signal',
                        marker=dict(color='green', symbol='triangle-up', size=10)
                    ))
                
                if sell_mask.any():
                    adx_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[sell_mask, 'Datetime'],
                        y=filtered_df.loc[sell_mask, 'ADX'],
                        mode='markers',
                        name='ADX Sell Signal',
                        marker=dict(color='red', symbol='triangle-down', size=10)
                    ))
            
            adx_fig.update_layout(
                title=f"ADX (period: {adx_period}, threshold: {adx_threshold})",
                xaxis_title="Date",
                yaxis_title="ADX",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            adx_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=adx_fig))
        
        # 6. Volume Chart
        if 'Volume_Signal' in selected_technicals and 'Volume' in filtered_df.columns:
            volume_fig = go.Figure()
            
            # Add Volume data
            volume_fig.add_trace(go.Bar(
                x=filtered_df['Datetime'],
                y=filtered_df['Volume'],
                name='Volume',
                marker_color='blue'
            ))
            
            # Add Volume MA if available
            if 'VMA' in filtered_df.columns:
                volume_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['VMA'],
                    mode='lines',
                    name=f'Volume MA ({volume_period})',
                    line=dict(color='orange', width=2)
                ))
            
            # Add signals - Volume signals are typically "High Volume" or "Low Volume"
            # We could mark points where volume is significantly above or below average
            if 'Volume_Signal' in filtered_df.columns:
                high_volume_mask = filtered_df['Volume_Signal'] == 'High Volume'
                
                if high_volume_mask.any():
                    volume_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[high_volume_mask, 'Datetime'],
                        y=filtered_df.loc[high_volume_mask, 'Volume'],
                        mode='markers',
                        name='High Volume',
                        marker=dict(color='green', symbol='circle', size=8)
                    ))
            
            volume_fig.update_layout(
                title=f"Volume Analysis (MA period: {volume_period})",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            volume_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=volume_fig))
        
        # 7. Fibonacci Retracement Chart
        if 'Fibonacci_Signal' in selected_technicals:
            fib_fig = go.Figure()
            
            # Add price data
            if chart_type == 'candlestick' and all(col in filtered_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                fib_fig.add_trace(go.Candlestick(
                    x=filtered_df['Datetime'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name=ticker,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
            else:
                fib_fig.add_trace(go.Scatter(
                    x=filtered_df['Datetime'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=2)
                ))
            
            # Add Fibonacci levels with improved historical handling
            if 'fib_levels' in filtered_df.columns and not filtered_df['fib_levels'].isnull().all():
                retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                colors = ['#FFB347', '#FFD700', '#90EE90', '#87CEEB', '#FF69B4']
                
                # Process data in chunks based on fibonacci_lookback
                chunk_size = max(1, min(fibonacci_lookback, len(filtered_df) // 10))
                for i in range(0, len(filtered_df), chunk_size):
                    chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                    if chunk.empty:
                        continue
                        
                    # Find first valid fib_levels in this chunk
                    valid_fib_levels = None
                    valid_idx = None
                    
                    for idx, row in chunk.iterrows():
                        if isinstance(row['fib_levels'], list) and len(row['fib_levels']) > 0:
                            valid_fib_levels = row['fib_levels']
                            valid_idx = idx
                            break
                            
                    if valid_fib_levels is None:
                        continue
                        
                    # Draw Fibonacci levels for this chunk
                    for level_idx, (level, color) in enumerate(zip(retracement_levels, colors)):
                        if level_idx < len(valid_fib_levels) and valid_fib_levels[level_idx] is not None:
                            fib_fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[valid_fib_levels[level_idx]] * len(chunk),
                                mode='lines',
                                name=f'Fib {level*100:.1f}%',
                                line=dict(color=color, dash='dash'),
                                showlegend=(i == 0)  # Only show in legend once
                            ))
                
                # Add high and low points used for Fibonacci calculation
                if 'fib_high' in filtered_df.columns and 'fib_low' in filtered_df.columns:
                    for i in range(0, len(filtered_df), chunk_size):
                        chunk = filtered_df.iloc[i:min(i + fibonacci_lookback, len(filtered_df))]
                        if chunk.empty:
                            continue
                            
                        high_val = chunk['fib_high'].iloc[0] if pd.notna(chunk['fib_high'].iloc[0]) else None
                        low_val = chunk['fib_low'].iloc[0] if pd.notna(chunk['fib_low'].iloc[0]) else None
                        
                        if high_val is not None:
                            fib_fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[high_val] * len(chunk),
                                mode='lines',
                                name='Fib High',
                                line=dict(color='red', dash='dot', width=1),
                                showlegend=(i == 0)
                            ))
                        
                        if low_val is not None:
                            fib_fig.add_trace(go.Scatter(
                                x=chunk['Datetime'],
                                y=[low_val] * len(chunk),
                                mode='lines',
                                name='Fib Low',
                                line=dict(color='green', dash='dot', width=1),
                                showlegend=(i == 0)
                            ))
            
            # Add signals
            if 'Fibonacci_Signal' in filtered_df.columns:
                buy_mask = filtered_df['Fibonacci_Signal'] == 'Buy'
                sell_mask = filtered_df['Fibonacci_Signal'] == 'Sell'
                
                if buy_mask.any():
                    fib_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[buy_mask, 'Datetime'],
                        y=filtered_df.loc[buy_mask, 'Close'],
                        mode='markers',
                        name='Fib Buy Signal',
                        marker=dict(color='green', symbol='triangle-up', size=10)
                    ))
                
                if sell_mask.any():
                    fib_fig.add_trace(go.Scatter(
                        x=filtered_df.loc[sell_mask, 'Datetime'],
                        y=filtered_df.loc[sell_mask, 'Close'],
                        mode='markers',
                        name='Fib Sell Signal',
                        marker=dict(color='red', symbol='triangle-down', size=10)
                    ))
            
            fib_fig.update_layout(
                title=f"Fibonacci Retracement (lookback: {fibonacci_lookback})",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            fib_fig.update_xaxes(
                rangeslider_visible=False,
                rangebreaks=[dict(bounds=["sat", "mon"])]
            )
            
            all_charts.append(dcc.Graph(figure=fib_fig))
        


        if show_explanations == 'yes':
            main_chart_with_explanation = html.Div([
                all_charts[0],  # Main price chart
                indicator_explanations['combination']
            ])

            charts_with_explanations = [main_chart_with_explanation]

            # Add explanations to individual indicator charts
            indicator_mapping = {
                'Bollinger_Signal': 1 if 'Bollinger_Signal' in selected_technicals else None,
                'MA_Signal': 2 if 'MA_Signal' in selected_technicals else None,
                'RSI_Signal': 3 if 'RSI_Signal' in selected_technicals else None,
                'MACD_Signal': 4 if 'MACD_Signal' in selected_technicals else None,
                'ADX_Signal': 5 if 'ADX_Signal' in selected_technicals else None,
                'Volume_Signal': 6 if 'Volume_Signal' in selected_technicals else None,
                'Fibonacci_Signal': 7 if 'Fibonacci_Signal' in selected_technicals else None,
            }

            for tech in selected_technicals:
                idx = indicator_mapping.get(tech)
                if idx is not None and idx < len(all_charts):
                    charts_with_explanations.append(html.Div([
                        all_charts[idx],
                        indicator_explanations.get(tech, html.Div())
                    ], className="mb-10 pb-4 border-b border-gray-200"))
        else:
            # If explanations are disabled, just use the charts without explanations
            main_chart_with_explanation = all_charts[0]
            
            charts_with_explanations = [main_chart_with_explanation]
            
            indicator_mapping = {
                'Bollinger_Signal': 1 if 'Bollinger_Signal' in selected_technicals else None,
                'MA_Signal': 2 if 'MA_Signal' in selected_technicals else None,
                'RSI_Signal': 3 if 'RSI_Signal' in selected_technicals else None,
                'MACD_Signal': 4 if 'MACD_Signal' in selected_technicals else None,
                'ADX_Signal': 5 if 'ADX_Signal' in selected_technicals else None,
                'Volume_Signal': 6 if 'Volume_Signal' in selected_technicals else None,
                'Fibonacci_Signal': 7 if 'Fibonacci_Signal' in selected_technicals else None,
            }
            
            for tech in selected_technicals:
                idx = indicator_mapping.get(tech)
                if idx is not None and idx < len(all_charts):
                    charts_with_explanations.append(html.Div([
                        all_charts[idx]
                    ], className="mb-10 pb-4 border-b border-gray-200"))

        # Return the charts with or without explanations
        return html.Div([
            html.H2(f"All Technical Analysis Charts for {ticker}", className="text-2xl font-bold mb-6 text-center"),
            html.Div(charts_with_explanations, className="space-y-8 pb-10")
        ])


        
        # # Return all charts in a container with spacing
        # return html.Div([
        #     html.H2(f"All Technical Analysis Charts for {ticker}", className="text-2xl font-bold mb-6 text-center"),
        #     html.Div(all_charts, className="space-y-8 pb-10")
        # ])
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in update_all_graphs: {traceback_str}")
        return html.Div([
            html.H3("Error creating charts", className="text-red-500"),
            html.P(f"Error: {str(e)}", className="mb-2"),
            html.P("Please check your input parameters and try again.", className="italic")
        ], className="p-4 bg-gray-50 rounded-lg border border-red-200")





def create_interactive_combination_graph(filtered_df, selected_technicals, chart_type, ticker, rsi_ob=70, rsi_os=30):
    """
    Creates an interactive combination chart with controls for technical indicators
    """
    try:
        # Create controls for technical indicators
        overlay_indicators = []
        separate_indicators = []
        
        # Categorize indicators
        overlay_types = ['Bollinger Bands', 'Moving Average', 'Fibonacci']
        separate_types = ['RSI', 'MACD', 'Volume', 'ADX']
        
        for tech in selected_technicals:
            if tech in overlay_types:
                overlay_indicators.append(tech)
            elif tech in separate_types:
                separate_indicators.append(tech)
        
        # Create controls section
        controls = html.Div([
            html.H4("Chart Controls", className="text-lg font-semibold mb-4"),
            
            # Chart Type Selection
            html.Div([
                html.Label("Chart Type:", className="font-semibold mb-2"),
                dcc.RadioItems(
                    id='chart-type-radio',
                    options=[
                        {'label': 'Candlestick', 'value': 'candlestick'},
                        {'label': 'Line', 'value': 'line'}
                    ],
                    value=chart_type,
                    className="mb-4"
                )
            ]),
            
            # Overlay Indicators - Changed to individual radio buttons for more control
            html.Div([
                html.Label("Overlay Indicators:", className="font-semibold mb-2"),
                html.Div([
                    # Individual overlay indicators with radio buttons for each
                    *[html.Div([
                        html.Label(f"{ind}:", className="ml-2"),
                        dcc.RadioItems(
                            id=f'overlay-{ind.lower().replace(" ", "-")}-radio',
                            options=[
                                {'label': 'Show', 'value': 'show'},
                                {'label': 'Hide', 'value': 'hide'}
                            ],
                            value='show' if ind in overlay_indicators else 'hide',
                            inline=True,
                            className="ml-2"
                        )
                    ], className="flex items-center mb-2") for ind in overlay_types if ind in selected_technicals]
                ], className="ml-4 mb-4"),
            ]),
            
            # Separate Indicators - Changed to individual radio buttons for more control
            html.Div([
                html.Label("Technical Indicators (Separate Charts):", className="font-semibold mb-2"),
                html.Div([
                    # Individual separate indicators with radio buttons for each
                    *[html.Div([
                        html.Label(f"{ind}:", className="ml-2"),
                        dcc.RadioItems(
                            id=f'separate-{ind.lower().replace(" ", "-")}-radio',
                            options=[
                                {'label': 'Show', 'value': 'show'},
                                {'label': 'Hide', 'value': 'hide'}
                            ],
                            value='show' if ind in separate_indicators else 'hide',
                            inline=True,
                            className="ml-2"
                        )
                    ], className="flex items-center mb-2") for ind in separate_types if ind in selected_technicals]
                ], className="ml-4 mb-4"),
            ]),
            
            # Hidden fields to store the current state of indicators
            html.Div([
                dcc.Checklist(
                    id='overlay-indicators-checklist',
                    options=[{'label': ind, 'value': ind} for ind in overlay_indicators],
                    value=overlay_indicators,
                    style={'display': 'none'}
                ),
                dcc.Checklist(
                    id='separate-indicators-checklist',
                    options=[{'label': ind, 'value': ind} for ind in separate_indicators],
                    value=separate_indicators,
                    style={'display': 'none'}
                )
            ]),
            
            # Update Button
            html.Button("Update Chart", id="update-combination-btn", 
                       className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600")
            ], className="bg-gray-50 p-4 rounded-lg mb-4")
        
        # Chart container
        chart_container = html.Div(id="combination-chart-display")
        
        return html.Div([controls, chart_container])
        
    except Exception as e:
        return html.Div([
            html.H3("Error creating interactive chart"),
            html.P(f"Error: {str(e)}")
        ])

# Add new callback for interactive chart updates
@app.callback(
    Output('combination-chart-display', 'children'),
    [Input('update-combination-btn', 'n_clicks'),
     Input('chart-type-radio', 'value'),
     Input('overlay-indicators-checklist', 'value'),
     Input('separate-indicators-checklist', 'value')],
    [State('ticker-dropdown', 'value'),
     State('start-year-dropdown', 'value'),
     State('start-month-dropdown', 'value'),
     State('start-day-dropdown', 'value'),
     State('end-year-dropdown', 'value'),
     State('end-month-dropdown', 'value'),
     State('end-day-dropdown', 'value'),
     State('data-period-selector', 'value'),
     State('realtime-interval-selector', 'value'),
     State('bb-period', 'value'),
     State('bb-std', 'value'),
     State('ma-short', 'value'),
     State('ma-long', 'value'),
     State('rsi-period', 'value'),
     State('rsi-overbought', 'value'),
     State('rsi-oversold', 'value'),
     State('macd-fast', 'value'),
     State('macd-slow', 'value'),
     State('macd-signal', 'value'),
     State('adx-period', 'value'),
     State('adx-threshold', 'value'),
     State('volume-period', 'value'),
     State('fibonacci-lookback', 'value')]
)
def update_interactive_combination_chart(n_clicks, chart_type, overlay_indicators, separate_indicators,
                                       ticker, start_year, start_month, start_day,
                                       end_year, end_month, end_day, selected_period, selected_interval,
                                       bb_window, bb_std, ma_short, ma_long,
                                       rsi_period, rsi_ob, rsi_os,
                                       macd_fast, macd_slow, macd_signal,
                                       adx_period, adx_threshold,
                                       volume_period, fibonacci_lookback):
    
    if not ticker:
        return html.Div("Please select a ticker first")
    
    try:
        # Get combined technical indicators
        all_selected = (overlay_indicators or []) + (separate_indicators or [])
        
        # Load and process data
        if selected_period == 'custom':
            # Handle custom date range
            start_date = datetime(int(start_year), int(start_month), int(start_day))
            end_date = datetime(int(end_year), int(end_month), int(end_day)) + timedelta(days=1) - timedelta(seconds=1)
            
            # Load data for custom period
            stock_data = load_and_process_data('max', ticker, selected_interval)
            filtered_df = stock_data[(stock_data['Datetime'] >= start_date) & 
                                   (stock_data['Datetime'] <= end_date)].copy()
        else:
            filtered_df, _ = update_data_based_on_ticker(ticker, selected_period)
        
        if filtered_df.empty:
            return html.Div("No data available for the selected period")
        
        # Apply technical analysis with custom parameters
        if 'Bollinger Bands' in all_selected:
            filtered_df = calculate_bollinger_bands(filtered_df, bb_window, bb_std)
        if 'Moving Average' in all_selected:
            filtered_df = calculate_ma(filtered_df, ma_short, ma_long)
        if 'RSI' in all_selected:
            filtered_df = calculate_rsi(filtered_df, rsi_period, rsi_ob, rsi_os)
        if 'MACD' in all_selected:
            filtered_df = calculate_macd(filtered_df, macd_fast, macd_slow, macd_signal)
        if 'ADX' in all_selected:
            filtered_df = calculate_adx(filtered_df, adx_period, adx_threshold)
        if 'Volume' in all_selected:
            filtered_df = calculate_volume(filtered_df, volume_period)
        if 'Fibonacci' in all_selected:
            filtered_df = calculate_fibonacci_retracement(filtered_df, fibonacci_lookback)
        
        # Create the charts
        charts = create_combination_charts_with_controls(
            filtered_df, overlay_indicators, separate_indicators, chart_type, ticker, rsi_ob, rsi_os
        )
        
        return charts
        
    except Exception as e:
        return html.Div(f"Error updating chart: {str(e)}")

def create_combination_charts_with_controls(filtered_df, overlay_indicators, separate_indicators, 
                                          chart_type, ticker, rsi_ob=70, rsi_os=30):
    """
    Create combination charts with separate handling for overlay and separate indicators
    """
    charts = []
    
    try:
        # Create main chart (candlestick or line)
        if chart_type == 'line':
            main_fig = go.Figure()
            main_fig.add_trace(go.Scatter(
                x=filtered_df['Datetime'],
                y=filtered_df['Close'],
                mode='lines',
                name=f'{ticker} Close Price',
                line=dict(color='blue', width=2)
            ))
        else:
            main_fig = go.Figure(data=go.Candlestick(
                x=filtered_df['Datetime'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                name=ticker
            ))
        
        # Add overlay indicators to main chart
        if overlay_indicators:
            for indicator in overlay_indicators:
                if indicator == 'Bollinger Bands':
                    if 'BB_Upper' in filtered_df.columns:
                        main_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['BB_Upper'],
                            mode='lines', name='BB Upper', line=dict(color='gray', dash='dash')
                        ))
                        main_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['BB_Lower'],
                            mode='lines', name='BB Lower', line=dict(color='gray', dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                        ))
                        main_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['BB_Middle'],
                            mode='lines', name='BB Middle', line=dict(color='orange')
                        ))
                
                elif indicator == 'Moving Average':
                    if 'MA_Short' in filtered_df.columns:
                        main_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['MA_Short'],
                            mode='lines', name='MA Short', line=dict(color='green')
                        ))
                    if 'MA_Long' in filtered_df.columns:
                        main_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['MA_Long'],
                            mode='lines', name='MA Long', line=dict(color='red')
                        ))
                
                elif indicator == 'Fibonacci' and 'Fib_0.618' in filtered_df.columns:
                    fib_levels = ['Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786']
                    colors = ['purple', 'blue', 'green', 'orange', 'red']
                    for level, color in zip(fib_levels, colors):
                        if level in filtered_df.columns:
                            main_fig.add_trace(go.Scatter(
                                x=filtered_df['Datetime'], y=filtered_df[level],
                                mode='lines', name=level, line=dict(color=color, dash='dot')
                            ))
        
        main_fig.update_layout(
            title=f"{ticker} - {chart_type.title()} Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            template="plotly_white",
            xaxis_rangeslider_visible=False
        )
        
        charts.append(dcc.Graph(figure=main_fig))
        
        # Create separate charts for indicators that need their own subplots
        if separate_indicators:
            for indicator in separate_indicators:
                if indicator == 'RSI' and 'RSI' in filtered_df.columns:
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'], y=filtered_df['RSI'],
                        mode='lines', name='RSI', line=dict(color='purple')
                    ))
                    rsi_fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red", annotation_text=f"Overbought ({rsi_ob})")
                    rsi_fig.add_hline(y=rsi_os, line_dash="dash", line_color="green", annotation_text=f"Oversold ({rsi_os})")
                    rsi_fig.update_layout(
                        title="RSI (Relative Strength Index)",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=300,
                        template="plotly_white"
                    )
                    charts.append(dcc.Graph(figure=rsi_fig))
                
                elif indicator == 'MACD' and 'MACD' in filtered_df.columns:
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'], y=filtered_df['MACD'],
                        mode='lines', name='MACD', line=dict(color='blue')
                    ))
                    if 'MACD_Signal' in filtered_df.columns:
                        macd_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['MACD_Signal'],
                            mode='lines', name='Signal', line=dict(color='red')
                        ))
                    if 'MACD_Histogram' in filtered_df.columns:
                        macd_fig.add_trace(go.Bar(
                            x=filtered_df['Datetime'], y=filtered_df['MACD_Histogram'],
                            name='Histogram', marker_color='gray'
                        ))
                    macd_fig.update_layout(
                        title="MACD",
                        xaxis_title="Date",
                        yaxis_title="MACD",
                        height=300,
                        template="plotly_white"
                    )
                    charts.append(dcc.Graph(figure=macd_fig))
                
                elif indicator == 'Volume' and 'Volume' in filtered_df.columns:
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=filtered_df['Datetime'], y=filtered_df['Volume'],
                        name='Volume', marker_color='lightblue'
                    ))
                    if 'Volume_MA' in filtered_df.columns:
                        volume_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['Volume_MA'],
                            mode='lines', name='Volume MA', line=dict(color='orange')
                        ))
                    volume_fig.update_layout(
                        title="Volume Analysis",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=300,
                        template="plotly_white"
                    )
                    charts.append(dcc.Graph(figure=volume_fig))
                
                elif indicator == 'ADX' and 'ADX' in filtered_df.columns:
                    adx_fig = go.Figure()
                    adx_fig.add_trace(go.Scatter(
                        x=filtered_df['Datetime'], y=filtered_df['ADX'],
                        mode='lines', name='ADX', line=dict(color='purple')
                    ))
                    if 'DI_Plus' in filtered_df.columns:
                        adx_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['DI_Plus'],
                            mode='lines', name='+DI', line=dict(color='green')
                        ))
                    if 'DI_Minus' in filtered_df.columns:
                        adx_fig.add_trace(go.Scatter(
                            x=filtered_df['Datetime'], y=filtered_df['DI_Minus'],
                            mode='lines', name='-DI', line=dict(color='red')
                        ))
                    adx_fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Threshold (25)")
                    adx_fig.update_layout(
                        title="ADX (Average Directional Index)",
                        xaxis_title="Date",
                        yaxis_title="ADX",
                        height=300,
                        template="plotly_white"
                    )
                    charts.append(dcc.Graph(figure=adx_fig))
        
        return html.Div(charts)
    
    except Exception as e:
        return html.Div([
            html.H4("Error creating charts"),
            html.P(f"Error: {str(e)}")
        ])

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run_server(debug=True)