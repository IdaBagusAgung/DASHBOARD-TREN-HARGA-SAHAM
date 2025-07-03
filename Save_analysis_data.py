import pymysql
import json
import pandas as pd
from datetime import datetime
import traceback
import numpy as np

# Kelas untuk mengubah objek yang tidak bisa di-serialize JSON
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)

class AnalysisDataSaver:
    def __init__(self, host='localhost', database='harga_saham', user='root', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host, 
                user=self.user, 
                password=self.password,
                database=self.database, 
                cursorclass=pymysql.cursors.DictCursor,
                charset='utf8mb4'
            )
            print("✅ Database connection established successfully")
            return True
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed")

    def create_table(self):
        """
        Create a single, denormalized table for storing all analysis results.
        REVISI: Membersihkan kolom yang duplikat/tidak perlu.
        """
        try:
            cursor = self.connection.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS hasil_analisis_lengkap (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ticker VARCHAR(15) NOT NULL,
                timestamp_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rentang_data_mulai DATE,
                rentang_data_selesai DATE,
                
                analysis_parameters LONGTEXT,
                raw_data_with_signals LONGTEXT,
                individual_backtests LONGTEXT,
                combination_backtests LONGTEXT,
                signal_pairs_profit LONGTEXT,
                signal_counting LONGTEXT,
                overall_summary LONGTEXT,
                
                all_individual_performance LONGTEXT,
                top_10_combination_performance LONGTEXT,
                top_10_diff_ind_vs_ind LONGTEXT,
                top_10_diff_mixed_combo LONGTEXT,
                
                -- <<< PERUBAHAN 1: TAMBAHKAN KOLOM BARU DI SINI >>>
                backtesting_insights LONGTEXT, -- Untuk menyimpan ringkasan insight dalam format JSON

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ticker_ts (ticker, timestamp_analisis)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_table_query)
            # Hapus kolom lama yang mungkin ada untuk kebersihan
            try:
                cursor.execute("ALTER TABLE hasil_analisis_lengkap DROP COLUMN different_combination_backtests;")
            except Exception: pass
            try:
                cursor.execute("ALTER TABLE hasil_analisis_lengkap DROP COLUMN top_3_different_combination;")
            except Exception: pass

            self.connection.commit()
            print("✅ Table 'hasil_analisis_lengkap' (revised) created/verified successfully.")
            return True
        except Exception as e:
            print(f"❌ Error creating table 'hasil_analisis_lengkap': {e}")
            traceback.print_exc()
            return False
            
    # ... fungsi _clean_and_validate_json_data dan _reduce_dict_size tetap sama ...
    def _clean_and_validate_json_data(self, data, max_size_mb=16):
        try:
            if data is None: return "{}"
            json_str = json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)
            size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            if size_mb > max_size_mb:
                print(f"⚠️ Warning: JSON data size ({size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)")
                if isinstance(data, list) and len(data) > 1000:
                    step = len(data) // 1000
                    data = data[::step][:1000]
                    json_str = json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)
                elif isinstance(data, dict):
                    cleaned_data = self._reduce_dict_size(data)
                    json_str = json.dumps(cleaned_data, cls=CustomJSONEncoder, ensure_ascii=False)
            json.loads(json_str)
            return json_str
        except Exception as e:
            print(f"❌ Error cleaning JSON data: {e}")
            return "{}"

    def _reduce_dict_size(self, data):
        if not isinstance(data, dict): return data
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 500:
                cleaned_data[key] = value[:100] + ["... truncated ..."] + value[-100:]
            elif isinstance(value, dict):
                cleaned_data[key] = self._reduce_dict_size(value)
            else:
                cleaned_data[key] = value
        return cleaned_data
    
    def save_analysis(self, ticker, analysis_params, raw_data_ticker,
                      individual_results_ticker, combination_results_ticker,
                      signal_pairs_data, signal_counts_data,
                      all_individual_data, top_10_combination_data, # Nama parameter diubah
                      top_10_diff_ind_vs_ind_data, top_10_diff_mixed_combo_data, # Nama parameter diubah
                      overall_summary_ticker, backtesting_insights_data):
        """
        Saves a complete analysis for a single ticker into one row.
        REVISI: Definisi fungsi disesuaikan dengan skema tabel yang baru (all/top_10).
        """
        try:
            cursor = self.connection.cursor()
            start_date, end_date = self._get_data_range(raw_data_ticker)
            
            print(f"🧹 Cleaning and validating JSON data for {ticker}...")
            
            values = (
                ticker, start_date, end_date,
                self._clean_and_validate_json_data(analysis_params),
                self._clean_and_validate_json_data(raw_data_ticker),
                self._clean_and_validate_json_data(individual_results_ticker),
                self._clean_and_validate_json_data(combination_results_ticker),
                self._clean_and_validate_json_data(signal_pairs_data),
                self._clean_and_validate_json_data(signal_counts_data),
                self._clean_and_validate_json_data(overall_summary_ticker),
                # Menggunakan parameter baru
                self._clean_and_validate_json_data(all_individual_data),
                self._clean_and_validate_json_data(top_10_combination_data),
                self._clean_and_validate_json_data(top_10_diff_ind_vs_ind_data),
                self._clean_and_validate_json_data(top_10_diff_mixed_combo_data),
                self._clean_and_validate_json_data(backtesting_insights_data) 
            )

            # Query INSERT disesuaikan dengan nama kolom baru
            insert_query = """
            INSERT INTO hasil_analisis_lengkap (
                ticker, rentang_data_mulai, rentang_data_selesai,
                analysis_parameters, raw_data_with_signals, 
                individual_backtests, combination_backtests,
                signal_pairs_profit, signal_counting, overall_summary, 
                all_individual_performance, top_10_combination_performance,
                top_10_diff_ind_vs_ind, top_10_diff_mixed_combo,
                backtesting_insights
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            cursor.execute(insert_query, values)
            analisis_id = cursor.lastrowid
            self.connection.commit()
            
            print(f"✅ Successfully saved analysis for {ticker} with ID: {analisis_id}")
            return analisis_id
        except Exception as e:
            print(f"❌ Error saving analysis for {ticker}: {e}")
            traceback.print_exc()
            if self.connection: self.connection.rollback()
            return None

    # ... _get_data_range dan get_saved_analysis tetap sama ...
    def _get_data_range(self, ticker_data):
        if not ticker_data: return None, None
        try:
            if isinstance(ticker_data, list):
                dates = [pd.to_datetime(d['Datetime']) for d in ticker_data if 'Datetime' in d and pd.notna(d.get('Datetime'))]
            else:
                dates = [pd.to_datetime(str(ticker_data))]
            if dates: return min(dates).date(), max(dates).date()
            return datetime.now().date(), datetime.now().date()
        except Exception as e:
            print(f"⚠️ Warning: Could not extract date range: {e}")
            return datetime.now().date(), datetime.now().date()

    def get_saved_analysis(self, ticker=None, limit=10):
        # ... kode tetap sama
        pass

# <<<--- PERBAIKAN: Definisi kelas DifferentCombinationSaver diletakkan di sini ---<<<
# class DifferentCombinationSaver:
#     def __init__(self, connection):
#         self.connection = connection

#     def create_tables(self):
#         """Membuat kedua tabel untuk kombinasi sinyal berbeda."""
#         try:
#             cursor = self.connection.cursor()
#             table1_query = """
#             CREATE TABLE IF NOT EXISTS diff_combo_ind_performance (
#                 id INT AUTO_INCREMENT PRIMARY KEY,
#                 analysis_id INT NOT NULL,
#                 ticker VARCHAR(15) NOT NULL,
#                 buy_indicator VARCHAR(255) NOT NULL,
#                 sell_indicator VARCHAR(255) NOT NULL,
#                 return_percentage DECIMAL(10, 4),
#                 total_profit DECIMAL(15, 4),
#                 win_rate DECIMAL(7, 4),
#                 num_trades INT,
#                 trades_detail LONGTEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (analysis_id) REFERENCES hasil_analisis_lengkap(id) ON DELETE CASCADE
#             ) ENGINE=InnoDB;
#             """
#             table2_query = """
#             CREATE TABLE IF NOT EXISTS diff_combo_mix_performance (
#                 id INT AUTO_INCREMENT PRIMARY KEY,
#                 analysis_id INT NOT NULL,
#                 ticker VARCHAR(15) NOT NULL,
#                 buy_indicator VARCHAR(255) NOT NULL,
#                 sell_indicator VARCHAR(255) NOT NULL,
#                 return_percentage DECIMAL(10, 4),
#                 total_profit DECIMAL(15, 4),
#                 win_rate DECIMAL(7, 4),
#                 num_trades INT,
#                 trades_detail LONGTEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (analysis_id) REFERENCES hasil_analisis_lengkap(id) ON DELETE CASCADE
#             ) ENGINE=InnoDB;
#             """
#             cursor.execute(table1_query)
#             cursor.execute(table2_query)
#             self.connection.commit()
#             print("✅ Tables 'diff_combo_ind_performance' and 'diff_combo_mix_performance' created/verified.")
#             return True
#         except Exception as e:
#             print(f"❌ Error creating different combination tables: {e}")
#             return False
    





    def save_results(self, analysis_id, results):
        """Menyimpan hasil ke tabel yang sesuai berdasarkan nama indikator."""
        if not results: return
        cursor = self.connection.cursor()
        try:
            for result in results:
                buy_indicator = result.get('buy_indicator', 'N/A')
                sell_indicator = result.get('sell_indicator', 'N/A')
                
                is_buy_combo = '_Combined_Signal' in buy_indicator
                is_sell_combo = '_Combined_Signal' in sell_indicator
                
                table_name = 'diff_combo_mix_performance' if is_buy_combo or is_sell_combo else 'diff_combo_ind_performance'
                
                trades_json = json.dumps(result.get('trades', []), cls=CustomJSONEncoder)
                
                query = f"""
                INSERT INTO {table_name} (
                    analysis_id, ticker, buy_indicator, sell_indicator,
                    return_percentage, total_profit, win_rate, num_trades, trades_detail
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    analysis_id, result.get('ticker'), buy_indicator, sell_indicator,
                    result.get('return_percentage'), result.get('total_profit'),
                    result.get('win_rate'), result.get('total_trades'), trades_json
                )
                cursor.execute(query, values)
            
            self.connection.commit()
            print(f"✅ Saved {len(results)} different combination results to their respective tables for analysis_id {analysis_id}.")
        except Exception as e:
            print(f"❌ Error saving different combination results: {e}")
            self.connection.rollback()



def _calculate_category_insight(results_list, category_name):
    """Menghitung statistik rata-rata untuk sebuah kategori hasil."""
    total_strategies = len(results_list)
    
    if total_strategies == 0:
        return {
            'category': category_name,
            'total_strategies': 0,
            'avg_profit': 0.0,
            'avg_win_rate': 0.0,
            'avg_profit_percentage': 0.0
        }
    
    # Ekstrak metrik dengan aman, tangani None atau key yang hilang
    total_profits = [r.get('total_profit', 0) or 0 for r in results_list]
    win_rates = [r.get('win_rate', 0) or 0 for r in results_list]
    profit_percentages = [r.get('profit_percentage', r.get('return_percentage', 0)) or 0 for r in results_list]
    
    avg_profit = sum(total_profits) / total_strategies
    avg_win_rate = sum(win_rates) / total_strategies
    avg_profit_percentage = sum(profit_percentages) / total_strategies
    
    return {
        'category': category_name,
        'total_strategies': total_strategies,
        'avg_profit': avg_profit,
        'avg_win_rate': avg_win_rate,
        'avg_profit_percentage': avg_profit_percentage
    }

# --- FUNGSI WRAPPER YANG AKAN DIPANGGIL DARI CALLBACK ---
def save_bulk_analysis_to_database(
    tickers, analysis_params, raw_data_with_signals, 
    individual_results, combination_results, different_combination_results, all_results_ui,
    all_signal_pairs_profit, all_signal_counts
):
    """
    REVISI: Menyimpan SEMUA hasil individual dan TOP 10 untuk kategori lainnya.
    """
    db_saver = AnalysisDataSaver()
    # diff_combo_saver = None
    
    try:
        if not db_saver.connect(): return False
        
        # diff_combo_saver = DifferentCombinationSaver(db_saver.connection)
        
        # if not db_saver.create_table() or not diff_combo_saver.create_tables(): return False

        if not db_saver.create_table(): 
            return False
        
        saved_analyses = []
        
        # Helper function tetap sama
        def get_safe_profit(result_dict):
            profit = result_dict.get('profit_percentage', result_dict.get('return_percentage'))
            return float(profit) if profit is not None and isinstance(profit, (int, float)) else -999999.0

        for ticker in tickers:
            print(f"\n💾 Processing, sorting, and saving for ticker: {ticker}")
            
            raw_data_ticker = raw_data_with_signals.get(ticker, [])
            if not raw_data_ticker: 
                print(f"   ⚠️ No raw data for {ticker}, skipping save...")
                continue

            # <<< PERUBAHAN 4: Logika untuk Kategori 1 (Individual) >>>
            # Simpan SEMUA hasil, bukan hanya top 3
            individual_results_ticker = [r for r in individual_results if r.get('ticker') == ticker]
            individual_results_ticker.sort(key=get_safe_profit, reverse=True)
            # Tidak ada pemotongan [:..], simpan semua
            all_individual_performance_data = individual_results_ticker
            print(f"   - Storing all {len(all_individual_performance_data)} individual results.")

            # <<< PERUBAHAN 5: Logika untuk Kategori 2 (Kombinasi - Voting) >>>
            # Simpan TOP 10
            combination_results_ticker = [r for r in combination_results if r.get('ticker') == ticker]
            combination_results_ticker.sort(key=get_safe_profit, reverse=True)
            top_10_combination = combination_results_ticker[:10] # Diubah dari 3 ke 10
            # print(f"   - Top 10 Combination (Voting): {[f'{p.get("indicator")}: {get_safe_profit(p):.2f}%' for p in top_10_combination]}")

            # <<< PERUBAHAN 6: Logika untuk Kategori 3 (Kombinasi Berbeda) >>>
            # Simpan TOP 10 untuk setiap sub-kategori
            diff_combo_results_ticker = [r for r in different_combination_results if r.get('ticker') == ticker]
            


            diff_ind_vs_ind = []
            diff_mixed = []
            print(f"   - Analyzing {len(diff_combo_results_ticker)} 'different combination' results...")
            for r in diff_combo_results_ticker:
                buy_ind = r.get('buy_indicator', '')
                sell_ind = r.get('sell_indicator', '')
                is_buy_signal_combo_type = '_Combined_Signal' in buy_ind
                is_sell_signal_combo_type = '_Combined_Signal' in sell_ind

                if is_buy_signal_combo_type or is_sell_signal_combo_type:
                    diff_mixed.append(r)
                else:
                    diff_ind_vs_ind.append(r)
            
            # print(f"     - Found {len(diff_ind_vs_ind)} 'Ind vs Ind' strategies.")
            # print(f"     - Found {len(diff_mixed)} 'Mixed Combo' strategies.")

            # Urutkan dan ambil Top 3 untuk masing-masing sub-kategori
            diff_ind_vs_ind.sort(key=get_safe_profit, reverse=True)
            top_10_diff_ind_vs_ind = diff_ind_vs_ind[:10]
            # <<<--- PERBAIKAN SINTAKS DI SINI ---<<<
            # print(f"   - Top 3 Diff (Ind vs Ind): {[f'{p.get("buy_indicator")}->{p.get("sell_indicator")}: {get_safe_profit(p):.2f}%' for p in top_3_diff_ind_vs_ind]}")

            diff_mixed.sort(key=get_safe_profit, reverse=True)
            top_10_diff_mixed_combo = diff_mixed[:10]
            # <<<--- PERBAIKAN SINTAKS DI SINI ---<<<
            # print(f"   - Top 3 Diff (Mixed Combo): {[f'{p.get("buy_indicator")}->{p.get("sell_indicator")}: {get_safe_profit(p):.2f}%' for p in top_3_diff_mixed_combo]}")


            print(f"   🧮 Calculating backtesting insights for {ticker}...")
            
            individual_insight = _calculate_category_insight(individual_results_ticker, 'Individual')
            combination_insight = _calculate_category_insight(combination_results_ticker, 'Combination')
            different_combo_insight = _calculate_category_insight(diff_combo_results_ticker, 'Different_Combination')
            overall_results = individual_results_ticker + combination_results_ticker + diff_combo_results_ticker
            overall_insight = _calculate_category_insight(overall_results, 'Overall')

            # <<<--- PERUBAHAN 3: Kumpulkan insight ke dalam satu list ---<<<
            # List ini yang akan di-serialize ke JSON dan disimpan
            backtesting_insights_data = [
                individual_insight, 
                combination_insight, 
                different_combo_insight, 
                overall_insight
            ]
            
            # print(f"     -> Overall Insight: {overall_insight['total_strategies']} strategies, Avg Profit: {overall_insight['avg_profit']:.2f}, Avg Winrate: {overall_insight['avg_win_rate']:.2f}%")

            
            # Buat overall summary
            all_top_performers = (all_individual_performance_data + top_10_combination + top_10_diff_ind_vs_ind + top_10_diff_mixed_combo)
            leaderboard_summary = sorted(
                [{'indicator': p.get('strategy', p.get('indicator', 'N/A')), 'profit_percentage': get_safe_profit(p)} for p in all_top_performers],
                key=lambda x: x.get('profit_percentage', -999999), 
                reverse=True
            )
            
            overall_summary_ticker = {
                'num_individual_strategies': len(individual_results_ticker),
                'num_combination_strategies': len(combination_results_ticker),
                'num_different_combination_strategies': len(diff_combo_results_ticker),
                'leaderboard': leaderboard_summary[:10] # Tampilkan top 10 di summary juga
            }

            # <<< PERUBAHAN 7: Panggil `save_analysis` dengan parameter yang sudah diubah >>>
            analysis_id = db_saver.save_analysis(
                ticker=ticker,
                analysis_params=analysis_params,
                raw_data_ticker=raw_data_ticker,
                individual_results_ticker=individual_results_ticker,
                combination_results_ticker=combination_results_ticker,
                signal_pairs_data=all_signal_pairs_profit.get(ticker, {}),
                signal_counts_data=all_signal_counts.get(ticker, {}),
                # Menggunakan variabel baru
                all_individual_data=all_individual_performance_data,
                top_10_combination_data=top_10_combination,
                top_10_diff_ind_vs_ind_data=top_10_diff_ind_vs_ind,
                top_10_diff_mixed_combo_data=top_10_diff_mixed_combo,
                overall_summary_ticker=overall_summary_ticker,
                backtesting_insights_data=backtesting_insights_data
            )
            
            if analysis_id:
                # if diff_combo_results_ticker:
                #     diff_combo_saver.save_results(analysis_id, diff_combo_results_ticker)
                saved_analyses.append({'ticker': ticker, 'analysis_id': analysis_id})
                print(f"✅ Successfully processed and saved all data for {ticker}")
            else:
                print(f"❌ Failed to save main analysis for {ticker}.")

        print(f"\n🎉 Bulk analysis save process finished. Saved {len(saved_analyses)} of {len(tickers)} tickers.")
        return saved_analyses

    except Exception as e:
        print(f"❌ Critical error in database save process: {e}")
        traceback.print_exc()
        return False
    finally:
        if db_saver: db_saver.disconnect()


# ... sisa fungsi Anda (get_saved_analysis_history, test_database_connection) tetap sama ...
def get_saved_analysis_history(ticker=None, limit=10):
    db_saver = AnalysisDataSaver()
    try:
        if not db_saver.connect(): return []
        return db_saver.get_saved_analysis(ticker, limit)
    except Exception as e:
        print(f"❌ Error retrieving analysis history: {e}")
        return []
    finally:
        if db_saver: db_saver.disconnect()

# def test_database_connection():
#     db_saver = AnalysisDataSaver()
#     if db_saver.connect():
#         if db_saver.create_table():
#             diff_saver = DifferentCombinationSaver(db_saver.connection)
#             if diff_saver.create_tables():
#                 print("✅ All database tables setup completed successfully!")
#         else:
#             print("❌ Failed to create tables")
#         db_saver.disconnect()
#     else:
#         print("❌ Database connection failed!")

# if __name__ == "__main__"