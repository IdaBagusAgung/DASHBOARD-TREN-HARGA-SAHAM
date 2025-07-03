import pandas as pd
import pymysql
import json

def get_backtesting_data():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='harga_saham'
    )
    cursor = connection.cursor()

    processed_data = []  # Initialize an empty list to store rows
    
    # SQL query to fetch the data
    sql = "SELECT TIMESTAMP, title, ticker, analysis, signal_data FROM data_signal_dashboard2"
    cursor.execute(sql)
    results = cursor.fetchall()  # Fetch all rows

    # Iterate through the rows
    for row in results:
        timestamp = row[0]  # TIMESTAMP
        title = row[1]      # Title
        ticker = row[2]     # Ticker
        analysis = row[3]   # Analysis (string)
        signal_data_json = row[4]  # JSON data (signal_data)
        
        if signal_data_json:
            # Parse JSON data
            signal_data = json.loads(signal_data_json)
            
            # Extract "signal_details" and flatten it
            signal_details = signal_data.get("signal_details", {})
            
            for signal_type, signals in signal_details.items():
                # Iterate over both "Buy" and "Sell" signals
                for signal_entry in signals.get("Buy", []):  # Extract "Buy" signals
                    processed_data.append({
                        'Timestamp': timestamp,
                        'Title': title,
                        'Ticker': ticker,
                        'Analysis': analysis,  # Include the analysis column
                        'Signal Type': "Buy",  # Explicitly mark as "Buy"
                        'Datetime': signal_entry.get('datetime'),
                        'Price': signal_entry.get('price'),
                        'Signal': signal_entry.get('signal'),
                        'Volume': signal_entry.get('volume'),
                        'Analysis_info': signal_entry.get('analysis')  
                    })
                
                for signal_entry in signals.get("Sell", []):  # Extract "Sell" signals
                    processed_data.append({
                        'Timestamp': timestamp,
                        'Title': title,
                        'Ticker': ticker,
                        'Analysis': analysis,  # Include the analysis column
                        'Signal Type': "Sell",  # Explicitly mark as "Sell"
                        'Datetime': signal_entry.get('datetime'),
                        'Price': signal_entry.get('price'),
                        'Signal': signal_entry.get('signal'),
                        'Volume': signal_entry.get('volume'),
                        'Analysis_info': signal_entry.get('analysis')
                    })

                for signal_entry in signals.get("Hold", []):  # Extract "Sell" signals
                    processed_data.append({
                        'Timestamp': timestamp,
                        'Title': title,
                        'Ticker': ticker,
                        'Analysis': analysis,  # Include the analysis column
                        'Signal Type': "Sell",  # Explicitly mark as "Sell"
                        'Datetime': signal_entry.get('datetime'),
                        'Price': signal_entry.get('price'),
                        'Signal': signal_entry.get('signal'),
                        'Volume': signal_entry.get('volume'),
                        'Analysis_info': signal_entry.get('analysis')
                    })

    cursor.close()
    connection.close()
    
    # Convert the list of processed data into a DataFrame
    backtesting_data = pd.DataFrame(processed_data)
    
    return backtesting_data