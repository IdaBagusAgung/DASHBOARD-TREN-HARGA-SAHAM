import pandas as pd
import numpy as np

def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns and determine overall market trend from patterns.
    Focus on trend direction (Uptrend, Downtrend, Sideways) rather than buy/sell signals.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with OHLC data
    
    Returns:
    pandas.DataFrame: Original dataframe with added pattern columns and trend information
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        print(f"🔍 CANDLESTICK DEBUG (Candlestick_pattern.py): ERROR - Missing one or more required columns: {required_columns}. Available columns: {df.columns.tolist()}")
        raise ValueError(f"DataFrame must contain all of these columns: {required_columns}")
    
    # Store the original column names to handle case sensitivity
    column_mapping = {}
    for col in required_columns:
        for df_col in df.columns:
            if df_col.lower() == col.lower():
                column_mapping[col] = df_col
    
    # Extract columns for pattern recognition (handling case sensitivity)
    open_col = column_mapping.get('Open', 'Open')
    high_col = column_mapping.get('High', 'High')
    low_col = column_mapping.get('Low', 'Low')
    close_col = column_mapping.get('Close', 'Close')
    
    # Add useful technical values
    result_df['body_size'] = abs(result_df[close_col] - result_df[open_col])
    result_df['range'] = result_df[high_col] - result_df[low_col]
    
    # Handle division by zero
    result_df['range'] = result_df['range'].replace(0, 1e-10)
    
    result_df['upper_shadow'] = result_df[high_col] - result_df[[open_col, close_col]].max(axis=1)
    result_df['lower_shadow'] = result_df[[open_col, close_col]].min(axis=1) - result_df[low_col]
    result_df['is_bullish'] = (result_df[close_col] > result_df[open_col]).astype(int)
    result_df['is_bearish'] = (result_df[close_col] < result_df[open_col]).astype(int)
    
    # Calculate previous values
    result_df['prev_open'] = result_df[open_col].shift(1)
    result_df['prev_high'] = result_df[high_col].shift(1)
    result_df['prev_low'] = result_df[low_col].shift(1)
    result_df['prev_close'] = result_df[close_col].shift(1)
    result_df['prev_body_size'] = result_df['body_size'].shift(1)
    result_df['prev_range'] = result_df['range'].shift(1)
    result_df['prev_is_bullish'] = result_df['is_bullish'].shift(1)
    result_df['prev_is_bearish'] = result_df['is_bearish'].shift(1)
    
    # Calculate values for 2 periods ago
    result_df['prev2_open'] = result_df[open_col].shift(2)
    result_df['prev2_high'] = result_df[high_col].shift(2)
    result_df['prev2_low'] = result_df[low_col].shift(2)
    result_df['prev2_close'] = result_df[close_col].shift(2)
    result_df['prev2_body_size'] = result_df['body_size'].shift(2)
    result_df['prev2_is_bullish'] = result_df['is_bullish'].shift(2)
    result_df['prev2_is_bearish'] = result_df['is_bearish'].shift(2)
    
    # Technical trend detection using moving averages - ADJUSTED FOR DAILY DATA
    min_window = min(5, len(result_df))
    result_df['sma5'] = result_df[close_col].rolling(window=min_window).mean()
    
    if len(result_df) >= 20:
        result_df['sma20'] = result_df[close_col].rolling(window=20).mean()
        result_df['uptrend'] = (result_df['sma5'] > result_df['sma20']).astype(int)
        result_df['downtrend'] = (result_df['sma5'] < result_df['sma20']).astype(int)
    else:
        # For small datasets, use price momentum
        result_df['uptrend'] = (result_df[close_col] > result_df[close_col].shift(1)).astype(int)
        result_df['downtrend'] = (result_df[close_col] < result_df[close_col].shift(1)).astype(int)
    
    # Initialize detected_patterns column properly
    result_df['detected_patterns'] = ''
    
    # Initialize pattern columns with zeros
    pattern_list = [
        'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
        'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
        'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
        'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
        'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
        'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
        'CDLGRAVESTONEDOJI', 'CDLDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU',
        'CDLTAKURI', 'CDLRICKSHAWMAN'
    ]
    
    for pattern in pattern_list:
        result_df[pattern] = 0
    
    # Pattern name mapping for readable output
    pattern_names = {
        'CDL3WHITESOLDIERS': 'Three White Soldiers',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLPIERCING': 'Piercing Pattern',
        'CDLHAMMER': 'Hammer',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLENGULFING_bullish': 'Bullish Engulfing',
        'CDLHARAMI_bullish': 'Bullish Harami',
        'CDLHIKKAKE_bullish': 'Bullish Hikkake',
        'CDLMORNINGDOJISTAR': 'Morning Doji Star',
        'CDLONNECK': 'On Neck Pattern',
        'CDLTHRUSTING': 'Thrusting Pattern',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDLEVENINGSTAR': 'Evening Star',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDLHANGINGMAN': 'Hanging Man',
        'CDLENGULFING_bearish': 'Bearish Engulfing',
        'CDLHARAMI_bearish': 'Bearish Harami',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDLHIKKAKE_bearish': 'Bearish Hikkake',
        'CDLEVENINGDOJISTAR': 'Evening Doji Star',
        'CDLSTALLEDPATTERN': 'Stalled Pattern',
        'CDLGRAVESTONEDOJI': 'Gravestone Doji',
        'CDLDOJI': 'Doji',
        'CDLSPINNINGTOP': 'Spinning Top',
        'CDLMARUBOZU': 'Marubozu',
        'CDLTAKURI': 'Takuri Line',
        'CDLRICKSHAWMAN': 'Rickshaw Man'
    }
    
    # ============================ START: FIX 1 ============================
    # The `add_pattern_name` function is completely replaced with this robust version.
    def add_pattern_name(df, idx, pattern_code):
        """Add pattern name to detected_patterns column safely."""
        try:
            pattern_name = pattern_names.get(pattern_code, pattern_code.replace('CDL', ''))
            
            # Retrieve the current value, which might be '', a string, or NaN
            current_patterns_val = df.at[df.index[idx], 'detected_patterns']
            
            # Check if it's already a non-empty string
            if isinstance(current_patterns_val, str) and current_patterns_val:
                # Append only if the pattern isn't already listed
                if pattern_name not in current_patterns_val:
                    df.at[df.index[idx], 'detected_patterns'] = current_patterns_val + ', ' + pattern_name
            else:
                # If it's empty or NaN, just set the new pattern name
                df.at[df.index[idx], 'detected_patterns'] = pattern_name

        except Exception as e:
            # This will catch any other unexpected errors, e.g., if idx is out of bounds
            print(f"DEBUG: Error adding pattern {pattern_code} at index {idx}: {e}")
    # ============================= END: FIX 1 =============================
    
    # ADJUSTED THRESHOLDS FOR DAILY DATA
    # 1. Detect Doji Patterns - MORE LENIENT
    doji_threshold = 0.15
    result_df['CDLDOJI'] = ((result_df['body_size'] / result_df['range']) < doji_threshold).astype(int)
    
    # ... (the rest of your vectorized pattern detections are fine)
    # 2. Detect Spinning Top
    spinning_threshold = 0.4
    shadow_min_ratio = 0.2
    result_df['CDLSPINNINGTOP'] = (
        ((result_df['body_size'] / result_df['range']) < spinning_threshold) & 
        (result_df['upper_shadow'] > shadow_min_ratio * result_df['body_size']) & 
        (result_df['lower_shadow'] > shadow_min_ratio * result_df['body_size'])
    ).astype(int)
    
    # 3. Detect Marubozu
    marubozu_body_threshold = 0.7
    marubozu_shadow_threshold = 0.15
    result_df['CDLMARUBOZU'] = (
        (result_df['body_size'] > marubozu_body_threshold * result_df['range']) & 
        (result_df['upper_shadow'] < marubozu_shadow_threshold * result_df['range']) & 
        (result_df['lower_shadow'] < marubozu_shadow_threshold * result_df['range'])
    ).astype(int)
    
    # 4. Detect Hammer
    hammer_shadow_ratio = 1.5
    hammer_upper_threshold = 0.3
    result_df['CDLHAMMER'] = (
        (result_df['downtrend'] == 1) &
        (result_df['lower_shadow'] > hammer_shadow_ratio * result_df['body_size']) & 
        (result_df['upper_shadow'] < hammer_upper_threshold * result_df['body_size']) &
        (result_df['body_size'] > 0)
    ).astype(int)
    
    # 5. Detect Inverted Hammer
    result_df['CDLINVERTEDHAMMER'] = (
        (result_df['downtrend'] == 1) &
        (result_df['upper_shadow'] > hammer_shadow_ratio * result_df['body_size']) & 
        (result_df['lower_shadow'] < hammer_upper_threshold * result_df['body_size']) &
        (result_df['body_size'] > 0)
    ).astype(int)
    
    # 6. Detect Hanging Man
    result_df['CDLHANGINGMAN'] = (
        (result_df['uptrend'] == 1) &
        (result_df['lower_shadow'] > hammer_shadow_ratio * result_df['body_size']) & 
        (result_df['upper_shadow'] < hammer_upper_threshold * result_df['body_size']) &
        (result_df['body_size'] > 0) &
        (result_df['is_bearish'] == 1)
    ).astype(int)
    
    # 7. Detect Shooting Star
    result_df['CDLSHOOTINGSTAR'] = (
        (result_df['uptrend'] == 1) &
        (result_df['upper_shadow'] > hammer_shadow_ratio * result_df['body_size']) & 
        (result_df['lower_shadow'] < hammer_upper_threshold * result_df['body_size']) &
        (result_df['body_size'] > 0) &
        (result_df['is_bearish'] == 1)
    ).astype(int)
    
    # 8. Detect Bullish Engulfing
    engulfing_tolerance = 0.05
    result_df['CDLENGULFING_bullish'] = (
        (result_df['prev_is_bearish'] == 1) &
        (result_df['is_bullish'] == 1) &
        (result_df[open_col] <= result_df['prev_close'] * (1 + engulfing_tolerance)) &
        (result_df[close_col] >= result_df['prev_open'] * (1 - engulfing_tolerance)) &
        (result_df['body_size'] > result_df['prev_body_size'] * 0.8)
    ).astype(int)
    
    # 9. Detect Bearish Engulfing
    result_df['CDLENGULFING_bearish'] = (
        (result_df['prev_is_bullish'] == 1) &
        (result_df['is_bearish'] == 1) &
        (result_df[open_col] >= result_df['prev_close'] * (1 - engulfing_tolerance)) &
        (result_df[close_col] <= result_df['prev_open'] * (1 + engulfing_tolerance)) &
        (result_df['body_size'] > result_df['prev_body_size'] * 0.8)
    ).astype(int)
    
    # ... [Other vectorized patterns remain the same]
    result_df['CDLHARAMI_bullish'] = (
        (result_df['prev_is_bearish'] == 1) & (result_df['is_bullish'] == 1) &
        (result_df[open_col] > result_df['prev_close']) & (result_df[open_col] < result_df['prev_open']) &
        (result_df[close_col] < result_df['prev_open']) & (result_df[close_col] > result_df['prev_close'])
    ).astype(int)
    result_df['CDLHARAMI_bearish'] = (
        (result_df['prev_is_bullish'] == 1) & (result_df['is_bearish'] == 1) &
        (result_df[open_col] < result_df['prev_close']) & (result_df[open_col] > result_df['prev_open']) &
        (result_df[close_col] > result_df['prev_open']) & (result_df[close_col] < result_df['prev_close'])
    ).astype(int)
    result_df['CDLMORNINGSTAR'] = (
        (result_df['downtrend'] == 1) & (result_df['prev2_is_bearish'] == 1) &
        (result_df['prev_body_size'] < result_df['prev2_body_size'] * 0.5) & (result_df['is_bullish'] == 1) &
        (result_df[close_col] > (result_df['prev2_open'] + result_df['prev2_close']) / 2)
    ).astype(int)
    result_df['CDLEVENINGSTAR'] = (
        (result_df['uptrend'] == 1) & (result_df['prev2_is_bullish'] == 1) &
        (result_df['prev_body_size'] < result_df['prev2_body_size'] * 0.5) & (result_df['is_bearish'] == 1) &
        (result_df[close_col] < (result_df['prev2_open'] + result_df['prev2_close']) / 2)
    ).astype(int)
    result_df['CDLMORNINGDOJISTAR'] = (result_df['CDLMORNINGSTAR'] & (result_df['prev_body_size'] / result_df['prev_range'] < 0.1)).astype(int)
    result_df['CDLEVENINGDOJISTAR'] = (result_df['CDLEVENINGSTAR'] & (result_df['prev_body_size'] / result_df['prev_range'] < 0.1)).astype(int)
    result_df['CDLPIERCING'] = (
        (result_df['downtrend'] == 1) & (result_df['prev_is_bearish'] == 1) & (result_df['is_bullish'] == 1) &
        (result_df[open_col] < result_df['prev_low']) & (result_df[close_col] > (result_df['prev_open'] + result_df['prev_close']) / 2) &
        (result_df[close_col] < result_df['prev_open'])
    ).astype(int)
    result_df['CDLDARKCLOUDCOVER'] = (
        (result_df['uptrend'] == 1) & (result_df['prev_is_bullish'] == 1) & (result_df['is_bearish'] == 1) &
        (result_df[open_col] > result_df['prev_high']) & (result_df[close_col] < (result_df['prev_open'] + result_df['prev_close']) / 2) &
        (result_df[close_col] > result_df['prev_open'])
    ).astype(int)
    result_df['CDLGRAVESTONEDOJI'] = (
        (result_df['uptrend'] == 1) & (abs(result_df[close_col] - result_df[open_col]) < 0.1 * result_df['range']) &
        (result_df['upper_shadow'] > 2 * result_df['body_size']) & (result_df['lower_shadow'] < 0.1 * result_df['range'])
    ).astype(int)

    # ============================ START: FIX 2 ============================
    # The `add_pattern_name` calls are removed from these functions. They now
    # only set the flag (0 or 1), and the final loop will build the string.
    # This avoids redundancy and potential bugs.
    def detect_three_white_soldiers(df):
        for i in range(2, len(df)):
            # Use .iloc for position-based access and check for NaN in shifted data
            if not np.isnan(df['prev2_close'].iloc[i]) and \
               (df['is_bullish'].iloc[i-2:i+1] == 1).all() and \
               (df['body_size'].iloc[i] > 0.7 * df['range'].iloc[i]) and \
               (df[open_col].iloc[i] > df[open_col].iloc[i-1]) and \
               (df[close_col].iloc[i] > df[close_col].iloc[i-1]) and \
               (df['downtrend'].iloc[i-2] == 1):
                df.at[df.index[i], 'CDL3WHITESOLDIERS'] = 1
        return df
    
    def detect_three_black_crows(df):
        for i in range(2, len(df)):
            if not np.isnan(df['prev2_close'].iloc[i]) and \
               (df['is_bearish'].iloc[i-2:i+1] == 1).all() and \
               (df['body_size'].iloc[i] > 0.7 * df['range'].iloc[i]) and \
               (df[open_col].iloc[i] < df[open_col].iloc[i-1]) and \
               (df[close_col].iloc[i] < df[close_col].iloc[i-1]) and \
               (df['uptrend'].iloc[i-2] == 1):
                df.at[df.index[i], 'CDL3BLACKCROWS'] = 1
        return df

    # Apply functions that need to iterate through the dataframe
    result_df = detect_three_white_soldiers(result_df)
    result_df = detect_three_black_crows(result_df)
    
    # ... [other vectorized patterns]
    result_df['CDLTAKURI'] = ((result_df['downtrend'] == 1) & (result_df['lower_shadow'] > 3 * result_df['body_size']) & (result_df['upper_shadow'] < 0.1 * result_df['body_size']) & (result_df['body_size'] > 0)).astype(int)
    result_df['CDLRICKSHAWMAN'] = ((result_df['body_size'] / result_df['range'] < 0.1) & (result_df['upper_shadow'] > 2 * result_df['body_size']) & (result_df['lower_shadow'] > 2 * result_df['body_size'])).astype(int)
    result_df['CDLONNECK'] = ((result_df['downtrend'] == 1) & (result_df['prev_is_bearish'] == 1) & (result_df['is_bullish'] == 1) & (abs(result_df[close_col] - result_df['prev_low']) < 0.1 * result_df['prev_body_size'])).astype(int)
    result_df['CDLTHRUSTING'] = ((result_df['downtrend'] == 1) & (result_df['prev_is_bearish'] == 1) & (result_df['is_bullish'] == 1) & (result_df[close_col] > result_df['prev_close']) & (result_df[close_col] < (result_df['prev_open'] + result_df['prev_close']) / 2)).astype(int)
    result_df['CDLSTALLEDPATTERN'] = ((result_df['uptrend'] == 1) & (result_df['prev2_is_bullish'] == 1) & (result_df['prev_is_bullish'] == 1) & (result_df['is_bullish'] == 1) & (result_df['body_size'] < 0.5 * result_df['prev_body_size']) & (result_df['prev_body_size'] < 0.7 * result_df['prev2_body_size'])).astype(int)
    
    def detect_hikkake_patterns(df):
        for i in range(3, len(df)):
            # Bullish Hikkake
            if not np.isnan(df['prev_close'].iloc[i-2]) and \
               (df['downtrend'].iloc[i-3] == 1) and \
               (df[high_col].iloc[i-2] < df[high_col].iloc[i-3]) and \
               (df[low_col].iloc[i-2] > df[low_col].iloc[i-3]) and \
               (df[low_col].iloc[i-1] < df[low_col].iloc[i-2]) and \
               (df[high_col].iloc[i] > df[high_col].iloc[i-2]):
                df.at[df.index[i], 'CDLHIKKAKE_bullish'] = 1
                
            # Bearish Hikkake
            if not np.isnan(df['prev_close'].iloc[i-2]) and \
               (df['uptrend'].iloc[i-3] == 1) and \
               (df[high_col].iloc[i-2] < df[high_col].iloc[i-3]) and \
               (df[low_col].iloc[i-2] > df[low_col].iloc[i-3]) and \
               (df[high_col].iloc[i-1] > df[high_col].iloc[i-2]) and \
               (df[low_col].iloc[i] < df[low_col].iloc[i-2]):
                df.at[df.index[i], 'CDLHIKKAKE_bearish'] = 1
        return df
    
    result_df = detect_hikkake_patterns(result_df)
    # ============================= END: FIX 2 =============================
    
    # Add pattern names to detected_patterns column for all patterns
    # This final loop is now the single source for creating the text strings
    for i in range(len(result_df)):
        for pattern_code in pattern_list:
            # Use .iloc for positional access which is faster in loops
            if result_df[pattern_code].iloc[i] == 1:
                add_pattern_name(result_df, i, pattern_code)
    
    # Define pattern categories
    pattern_categories = {
        'bullish': [
            'CDL3WHITESOLDIERS', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLHAMMER',
            'CDLINVERTEDHAMMER', 'CDLENGULFING_bullish', 'CDLHARAMI_bullish',
            'CDLHIKKAKE_bullish', 'CDLMORNINGDOJISTAR', 'CDLONNECK', 'CDLTHRUSTING',
        ],
        'bearish': [
            'CDL3BLACKCROWS', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN',
            'CDLENGULFING_bearish', 'CDLHARAMI_bearish', 'CDLDARKCLOUDCOVER',
            'CDLHIKKAKE_bearish', 'CDLEVENINGDOJISTAR', 'CDLSTALLEDPATTERN',
            'CDLGRAVESTONEDOJI'
        ],
        'neutral': [
            'CDLDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU',
            'CDLTAKURI', 'CDLRICKSHAWMAN'
        ]
    }
    
    # Create pattern summary columns
    result_df['bullish_patterns'] = result_df[pattern_categories['bullish']].sum(axis=1)
    result_df['bearish_patterns'] = result_df[pattern_categories['bearish']].sum(axis=1)
    result_df['neutral_patterns'] = result_df[pattern_categories['neutral']].sum(axis=1)
    
    # Calculate candlestick trend
    result_df['candlestick_trend'] = 'Sideways'
    
    # Determine trend
    result_df.loc[(result_df['uptrend'] == 1) & (result_df['bullish_patterns'] > result_df['bearish_patterns']), 'candlestick_trend'] = 'Uptrend'
    result_df.loc[(result_df['downtrend'] == 1) & (result_df['bearish_patterns'] > result_df['bullish_patterns']), 'candlestick_trend'] = 'Downtrend'
    
    # Strong pattern signals override MA trend
    result_df.loc[(result_df['bullish_patterns'] > result_df['bearish_patterns'] * 1.5), 'candlestick_trend'] = 'Uptrend'
    result_df.loc[(result_df['bearish_patterns'] > result_df['bullish_patterns'] * 1.5), 'candlestick_trend'] = 'Downtrend'
    
    # Generate confidence
    result_df['candlestick_confidence'] = 50
    
    # Adjust confidence
    result_df.loc[(result_df['candlestick_trend'] == 'Uptrend') & (result_df['uptrend'] == 1), 'candlestick_confidence'] += 20
    result_df.loc[(result_df['candlestick_trend'] == 'Downtrend') & (result_df['downtrend'] == 1), 'candlestick_confidence'] += 20
    
    # Adjust based on pattern strength
    uptrend_mask = result_df['candlestick_trend'] == 'Uptrend'
    downtrend_mask = result_df['candlestick_trend'] == 'Downtrend'
    
    max_bullish = result_df['bullish_patterns'].max() if len(result_df) > 0 and result_df['bullish_patterns'].max() > 0 else 1
    max_bearish = result_df['bearish_patterns'].max() if len(result_df) > 0 and result_df['bearish_patterns'].max() > 0 else 1
    
    result_df.loc[uptrend_mask, 'candlestick_confidence'] += 30 * (result_df.loc[uptrend_mask, 'bullish_patterns'] / max_bullish)
    result_df.loc[downtrend_mask, 'candlestick_confidence'] += 30 * (result_df.loc[downtrend_mask, 'bearish_patterns'] / max_bearish)
    
    result_df['candlestick_confidence'] = result_df['candlestick_confidence'].clip(0, 100).round(2)
    
    # Set signals
    result_df['Candlestick_Signal'] = 'Hold'
    
    # Final cleanup of temporary columns
    temp_cols = [
        'body_size', 'range', 'upper_shadow', 'lower_shadow', 'is_bullish', 'is_bearish',
        'prev_open', 'prev_high', 'prev_low', 'prev_close', 'prev_body_size', 'prev_range',
        'prev_is_bullish', 'prev_is_bearish', 'prev2_open', 'prev2_high', 'prev2_low',
        'prev2_close', 'prev2_body_size', 'prev2_is_bullish', 'prev2_is_bearish',
        'sma5', 'uptrend', 'downtrend'
    ]
    if 'sma20' in result_df.columns:
        temp_cols.append('sma20')
        
    # result_df.drop(columns=temp_cols, inplace=True, errors='ignore')

    print(result_df)  # Debugging output to check the result
    
    return result_df


def get_pattern_description(pattern_name):
    """
    Returns a detailed description of the given candlestick pattern.
    
    Parameters:
    pattern_name (str): Name of the candlestick pattern
    
    Returns:
    str: Description of the pattern
    """
    pattern_descriptions = {
        'CDLDOJI': '''
            Doji: Forms when the opening and closing prices are virtually equal. The length of the upper and lower shadows can vary.
            
            Interpretation: Represents indecision in the market. After an uptrend, it can signal a potential reversal. 
            After a downtrend, it may indicate a bottom formation.
            
            Look for: Confirmation from the next candle's direction.
        ''',
        
        'CDLSPINNINGTOP': '''
            Spinning Top: A candlestick with a small real body and long upper and lower shadows.
            
            Interpretation: Indicates indecision in the market similar to a Doji but with slightly less intensity.
            Often appears during consolidation before a significant move.
            
            Look for: The direction of the breakout after a series of spinning tops.
        ''',
        
        'CDLMARUBOZU': '''
            Marubozu: A candlestick with no shadows (or very small ones). The opening price equals the low and the closing price equals the high (for a bullish Marubozu), or the opening equals the high and closing equals the low (for a bearish Marubozu).
            
            Interpretation: Shows strong conviction in the market. A bullish Marubozu indicates strong buying pressure, while a bearish one indicates strong selling pressure.
            
            Look for: Potential continuation in the direction of the Marubozu.
        ''',
        
        'CDLHAMMER': '''
            Hammer: A bullish reversal pattern with a small real body, a long lower shadow, and little or no upper shadow. Found at the bottom of a downtrend.
            
            Interpretation: Indicates that sellers drove prices lower during the session, but buyers were able to push the price back up by closing, showing rejection of lower prices.
            
            Look for: Confirmation with a bullish candle the next day and increased volume.
        ''',
        
        'CDLINVERTEDHAMMER': '''
            Inverted Hammer: A bullish reversal pattern with a small real body, long upper shadow, and little or no lower shadow. Found at the bottom of a downtrend.
            
            Interpretation: Shows that buyers attempted to push the price up but met resistance. However, the fact that the price closed near the open without much selling pressure is seen as positive.
            
            Look for: Confirmation with a strong bullish candle the next day.
        ''',
        
        'CDLENGULFING_bullish': '''
            Bullish Engulfing: A two-candle pattern where a bearish candle is followed by a larger bullish candle that completely "engulfs" the first candle's body.
            
            Interpretation: Indicates a potential trend reversal from bearish to bullish, showing buyers have overwhelmed sellers.
            
            Look for: The pattern at the bottom of a downtrend, with the second candle showing strong volume.
        ''',
        
        'CDLENGULFING_bearish': '''
            Bearish Engulfing: A two-candle pattern where a bullish candle is followed by a larger bearish candle that completely "engulfs" the first candle's body.
            
            Interpretation: Indicates a potential trend reversal from bullish to bearish, showing sellers have overwhelmed buyers.
            
            Look for: The pattern at the top of an uptrend, with the second candle showing strong volume.
        ''',
        
        'CDLHARAMI_bullish': '''
            Bullish Harami: A two-candle pattern where a large bearish candle is followed by a smaller bullish candle that is completely contained within the body of the first candle.
            
            Interpretation: Suggests a weakening of the downtrend and potential reversal. The small bullish candle shows indecision after a strong bearish move.
            
            Look for: Following candles to confirm the reversal with upward momentum.
        ''',
        
        # Continue with additional detailed descriptions for all patterns...
        
        'CDLMORNINGSTAR': '''
            Morning Star: A three-candle bullish reversal pattern. First is a large bearish candle, followed by a small-bodied candle (star) that gaps down, and then a large bullish candle that closes well into the first candle's body.
            
            Interpretation: Shows a shift from bearish to bullish sentiment. The star represents indecision, and the third candle confirms the reversal.
            
            Look for: This pattern at the bottom of a downtrend, with the third candle showing strong volume.
        ''',
        
        'CDLEVENINGSTAR': '''
            Evening Star: A three-candle bearish reversal pattern. First is a large bullish candle, followed by a small-bodied candle (star) that gaps up, and then a large bearish candle that closes well into the first candle's body.
            
            Interpretation: Shows a shift from bullish to bearish sentiment. The star represents indecision, and the third candle confirms the reversal.
            
            Look for: This pattern at the top of an uptrend, with the third candle showing strong volume.
        ''',
        
        'CDLSHOOTINGSTAR': '''
            Shooting Star: A single bearish reversal candle with a small real body, long upper shadow, and little or no lower shadow. Found at the top of an uptrend.
            
            Interpretation: Indicates that buyers pushed the price up significantly during the session but sellers took control by the close, rejecting higher prices.
            
            Look for: Confirmation with a bearish candle the next day and potentially increased volume.
        ''',
    }
    
    # Additional patterns would continue with similar detailed descriptions
    
    # Default description for patterns not explicitly defined
    default_description = f"A technical candlestick pattern used to predict potential price movements. Check technical analysis resources for more information on {pattern_name.replace('CDL', '')}."
    
    return pattern_descriptions.get(pattern_name, default_description)

# def get_pattern_description(pattern_name):
#     """
#     Returns a description of the given candlestick pattern.
    
#     Parameters:
#     pattern_name (str): Name of the candlestick pattern
    
#     Returns:
#     str: Description of the pattern
#     """
#     pattern_descriptions = {
#         'CDL3WHITESOLDIERS': 'Three White Soldiers: Three consecutive long-bodied bullish candles, each opening within the previous candle\'s body and closing near its high. A strong bullish reversal pattern.',
#         'CDLMORNINGSTAR': 'Morning Star: A three-candle bullish reversal pattern consisting of a large bearish candle, followed by a small-bodied candle, and completed by a large bullish candle.',
#         'CDLPIERCING': 'Piercing Pattern: A two-candle bullish reversal pattern where a bearish candle is followed by a bullish candle that opens below the prior close but closes above the midpoint of the bearish candle.',
#         'CDLHAMMER': 'Hammer: A single-candle bullish reversal pattern with a small body, little or no upper shadow, and a long lower shadow. Found in downtrends.',
#         'CDLINVERTEDHAMMER': 'Inverted Hammer: A single-candle bullish reversal pattern with a small body, long upper shadow, and little or no lower shadow. Found in downtrends.',
        
#         'CDL3BLACKCROWS': 'Three Black Crows: Three consecutive long-bodied bearish candles, each opening within the previous candle\'s body and closing near its low. A strong bearish reversal pattern.',
#         'CDLEVENINGSTAR': 'Evening Star: A three-candle bearish reversal pattern consisting of a large bullish candle, followed by a small-bodied candle, and completed by a large bearish candle.',
#         'CDLSHOOTINGSTAR': 'Shooting Star: A single-candle bearish reversal pattern with a small body, long upper shadow, and little or no lower shadow. Found in uptrends.',
#         'CDLHANGINGMAN': 'Hanging Man: A single-candle bearish reversal pattern with a small body, little or no upper shadow, and a long lower shadow. Found in uptrends.',
#         'CDLDARKCLOUDCOVER': 'Dark Cloud Cover: A two-candle bearish reversal pattern where a bullish candle is followed by a bearish candle that opens above the prior high but closes below the midpoint of the bullish candle.',
        
#         'CDLHARAMI_bullish': 'Bullish Harami: A two-candle pattern where a small-bodied bullish candle is contained within the body of the preceding larger bearish candle. Signals potential trend reversal.',
#         'CDLHARAMI_bearish': 'Bearish Harami: A two-candle pattern where a small-bodied bearish candle is contained within the body of the preceding larger bullish candle. Signals potential trend reversal.',
#         'CDLENGULFING_bullish': 'Bullish Engulfing: A two-candle pattern where a bullish candle completely engulfs the body of the prior bearish candle. Strong bullish reversal signal.',
#         'CDLENGULFING_bearish': 'Bearish Engulfing: A two-candle pattern where a bearish candle completely engulfs the body of the prior bullish candle. Strong bearish reversal signal.',
        
#         'CDLDOJI': 'Doji: A candle with a very small body where opening and closing prices are very close. Indicates indecision in the market.',
#         'CDLSPINNINGTOP': 'Spinning Top: A candle with a small body and long upper and lower shadows, showing indecision between buyers and sellers.',
#         'CDLMARUBOZU': 'Marubozu: A candle with no shadows (or very small ones), indicating strong conviction. Bullish when white/green, bearish when black/red.',
        
#         'CDLHIKKAKE_bullish': 'Bullish Hikkake: A candlestick pattern that indicates a false breakout and potential reversal to the upside.',
#         'CDLHIKKAKE_bearish': 'Bearish Hikkake: A candlestick pattern that indicates a false breakout and potential reversal to the downside.',
#         'CDLMORNINGDOJISTAR': 'Morning Doji Star: A variation of the Morning Star pattern where the middle candle is a Doji, indicating an even stronger potential reversal.',
#         'CDLEVENINGDOJISTAR': 'Evening Doji Star: A variation of the Evening Star pattern where the middle candle is a Doji, indicating an even stronger potential reversal.',
#         'CDLTAKURI': 'Takuri Line: Similar to a Hammer, with an even longer lower shadow, indicating strong buying pressure after a decline.',
#         'CDLRICKSHAWMAN': 'Rickshaw Man: A type of Doji with very long upper and lower shadows, showing extreme indecision in the market.',
#         'CDLONNECK': 'On Neck Pattern: A bearish continuation pattern where a long bearish candle is followed by a small bullish candle that opens with a gap down.',
#         'CDLTHRUSTING': 'Thrusting Pattern: Similar to the Piercing pattern but the bullish candle closes below the midpoint of the bearish candle. Considered bearish continuation.',
#         'CDLSTALLEDPATTERN': 'Stalled Pattern: Three white candles with progressively smaller bodies, indicating diminishing buying pressure.',
#         'CDLGRAVESTONEDOJI': 'Gravestone Doji: A Doji with a long upper shadow and no lower shadow, often a bearish reversal signal at the top of an uptrend.'
#     }
    
#     return pattern_descriptions.get(pattern_name, f"No description available for {pattern_name}")

def get_candlestick_explanations():
    """
    Returns explanations of various candlestick patterns for the dashboard.
    
    Returns:
    dict: Dictionary of pattern explanations
    """
    explanations = {
        'title': 'Candlestick Pattern Analysis',
        'overview': '''
Candlestick patterns are visual formations created by price movements on a chart.
They can help identify potential reversals, continuations, or periods of indecision in the market.
This indicator analyzes the OHLC data to detect common candlestick patterns and generates signals.
        ''',
        'signals': {
            'Buy': 'Generated when bullish patterns outnumber bearish patterns, suggesting potential upward price movement.',
            'Sell': 'Generated when bearish patterns outnumber bullish patterns, suggesting potential downward price movement.',
            'Hold': 'Generated when bullish and bearish patterns are balanced or few patterns are detected, suggesting no clear direction.'
        },
        'patterns': {
            'bullish': [
                {'name': 'Three White Soldiers', 'code': 'CDL3WHITESOLDIERS', 'strength': 'Strong'},
                {'name': 'Morning Star', 'code': 'CDLMORNINGSTAR', 'strength': 'Strong'},
                {'name': 'Piercing Pattern', 'code': 'CDLPIERCING', 'strength': 'Moderate'},
                {'name': 'Hammer', 'code': 'CDLHAMMER', 'strength': 'Moderate'},
                {'name': 'Inverted Hammer', 'code': 'CDLINVERTEDHAMMER', 'strength': 'Moderate'},
                {'name': 'Bullish Engulfing', 'code': 'CDLENGULFING_bullish', 'strength': 'Strong'},
                {'name': 'Bullish Harami', 'code': 'CDLHARAMI_bullish', 'strength': 'Moderate'},
                {'name': 'Morning Doji Star', 'code': 'CDLMORNINGDOJISTAR', 'strength': 'Strong'},
            ],
            'bearish': [
                {'name': 'Three Black Crows', 'code': 'CDL3BLACKCROWS', 'strength': 'Strong'},
                {'name': 'Evening Star', 'code': 'CDLEVENINGSTAR', 'strength': 'Strong'},
                {'name': 'Shooting Star', 'code': 'CDLSHOOTINGSTAR', 'strength': 'Moderate'},
                {'name': 'Hanging Man', 'code': 'CDLHANGINGMAN', 'strength': 'Moderate'},
                {'name': 'Dark Cloud Cover', 'code': 'CDLDARKCLOUDCOVER', 'strength': 'Moderate'},
                {'name': 'Bearish Engulfing', 'code': 'CDLENGULFING_bearish', 'strength': 'Strong'},
                {'name': 'Bearish Harami', 'code': 'CDLHARAMI_bearish', 'strength': 'Moderate'},
                {'name': 'Evening Doji Star', 'code': 'CDLEVENINGDOJISTAR', 'strength': 'Strong'},
            ],
            'neutral': [
                {'name': 'Doji', 'code': 'CDLDOJI', 'strength': 'Weak'},
                {'name': 'Spinning Top', 'code': 'CDLSPINNINGTOP', 'strength': 'Weak'},
                {'name': 'Marubozu', 'code': 'CDLMARUBOZU', 'strength': 'Context-dependent'},
            ]
        },
        'usage': '''
When using candlestick patterns:
1. Look for patterns that occur at key support or resistance levels
2. Consider the overall trend - patterns are more reliable when they align with the trend
3. Use in conjunction with other technical indicators for confirmation
4. Higher confidence signals occur when multiple patterns appear simultaneously
        '''
    }
    
    return explanations