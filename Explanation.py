# from dash import html

# # Create dictionary of explanations for each technical indicator in Bahasa Indonesia
# indicator_explanations = {
#     'Bollinger_Signal': html.Div([
#         html.H4("Cara Membaca Bollinger Bands", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Pita Bollinger terdiri dari garis tengah (SMA) dengan dua pita di atas dan di bawahnya yang mewakili volatilitas.", className="mb-2"),
#             html.Li("Ketika harga mendekati pita atas, saham mungkin mengalami kondisi 'overbought' (jenuh beli).", className="mb-2"),
#             html.Li("Ketika harga mendekati pita bawah, saham mungkin mengalami kondisi 'oversold' (jenuh jual).", className="mb-2"),
#             html.Li("Penyempitan pita menunjukkan volatilitas rendah, sering mendahului pergerakan harga yang signifikan.", className="mb-2"),
#             html.Li("Pelebaran pita menunjukkan volatilitas tinggi.", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), " Ketika harga menyentuh atau menembus pita bawah dan kemudian bergerak kembali ke atas.", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), " Ketika harga menyentuh atau menembus pita atas dan kemudian bergerak kembali ke bawah.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'MA_Signal': html.Div([
#         html.H4("Cara Membaca Moving Average (MA)", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Moving Average menunjukkan tren harga dengan meratakan fluktuasi harian.", className="mb-2"),
#             html.Li("MA jangka pendek (garis biru) lebih responsif terhadap perubahan harga terbaru.", className="mb-2"),
#             html.Li("MA jangka panjang (garis merah) menunjukkan tren jangka panjang yang lebih stabil.", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), " Ketika MA jangka pendek memotong ke atas MA jangka panjang (golden cross).", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), " Ketika MA jangka pendek memotong ke bawah MA jangka panjang (death cross).", className="mb-2"),
#             html.Li("Harga di atas kedua MA menunjukkan tren naik yang kuat.", className="mb-2"),
#             html.Li("Harga di bawah kedua MA menunjukkan tren turun yang kuat.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'RSI_Signal': html.Div([
#         html.H4("Cara Membaca Relative Strength Index (RSI)", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("RSI mengukur kecepatan dan perubahan pergerakan harga, berkisar dari 0 hingga 100.", className="mb-2"),
#             html.Li(f"Nilai di atas {70} menunjukkan kondisi 'overbought' (jenuh beli), mengindikasikan potensi pembalikan turun.", className="mb-2"),
#             html.Li(f"Nilai di bawah {30} menunjukkan kondisi 'oversold' (jenuh jual), mengindikasikan potensi pembalikan naik.", className="mb-2"),
#             html.Li("Nilai 50 adalah garis tengah yang menunjukkan keseimbangan antara kekuatan pembeli dan penjual.", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), f" Ketika RSI turun di bawah {30} dan kemudian naik kembali di atas {30}.", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), f" Ketika RSI naik di atas {70} dan kemudian turun kembali di bawah {70}.", className="mb-2"),
#             html.Li("Divergensi: Ketika harga membentuk puncak/lembah baru tetapi RSI tidak, bisa menjadi indikasi pembalikan arah.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'MACD_Signal': html.Div([
#         html.H4("Cara Membaca Moving Average Convergence Divergence (MACD)", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("MACD terdiri dari garis MACD (biru), garis sinyal (merah), dan histogram yang menunjukkan perbedaan antara keduanya.", className="mb-2"),
#             html.Li("Garis MACD adalah selisih antara dua exponential moving average (EMA) dengan periode berbeda.", className="mb-2"),
#             html.Li("Histogram positif (hijau) menunjukkan MACD di atas garis sinyal, mengindikasikan momentum naik.", className="mb-2"),
#             html.Li("Histogram negatif (merah) menunjukkan MACD di bawah garis sinyal, mengindikasikan momentum turun.", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), " Ketika garis MACD memotong ke atas garis sinyal.", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), " Ketika garis MACD memotong ke bawah garis sinyal.", className="mb-2"),
#             html.Li("MACD di atas nol menunjukkan tren naik, sementara di bawah nol menunjukkan tren turun.", className="mb-2"),
#             html.Li("Divergensi: Ketika harga dan MACD bergerak ke arah berlawanan, bisa mengindikasikan pembalikan tren.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'ADX_Signal': html.Div([
#         html.H4("Cara Membaca Average Directional Index (ADX)", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("ADX mengukur kekuatan tren terlepas dari arahnya, dengan nilai berkisar 0-100.", className="mb-2"),
#             html.Li("Garis ADX (oranye) menunjukkan kekuatan tren secara keseluruhan.", className="mb-2"),
#             html.Li("Garis +DI (hijau) menunjukkan kekuatan tren naik.", className="mb-2"),
#             html.Li("Garis -DI (merah) menunjukkan kekuatan tren turun.", className="mb-2"),
#             html.Li(f"ADX di atas {25} menunjukkan tren yang kuat, semakin tinggi nilai semakin kuat trennya.", className="mb-2"),
#             html.Li(f"ADX di bawah {20} menunjukkan tidak ada tren yang jelas (sideways/ranging).", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), " Ketika +DI memotong ke atas -DI dan ADX di atas 25.", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), " Ketika -DI memotong ke atas +DI dan ADX di atas 25.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'Volume_Signal': html.Div([
#         html.H4("Cara Membaca Analisis Volume", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Volume menunjukkan jumlah saham yang diperdagangkan dalam periode tertentu.", className="mb-2"),
#             html.Li("Volume tinggi (di atas rata-rata) menunjukkan minat pasar yang kuat terhadap saham tersebut.", className="mb-2"),
#             html.Li("Volume rendah (di bawah rata-rata) menunjukkan kurangnya minat pasar.", className="mb-2"),
#             html.Li("Volume tinggi saat harga naik mengkonfirmasi kekuatan tren naik.", className="mb-2"),
#             html.Li("Volume tinggi saat harga turun mengkonfirmasi kekuatan tren turun.", className="mb-2"),
#             html.Li("Volume rendah selama pergerakan harga bisa mengindikasikan lemahnya tren dan potensi pembalikan.", className="mb-2"),
#             html.Li("Lonjakan volume yang tiba-tiba sering mendahului pergerakan harga yang signifikan.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'Fibonacci_Signal': html.Div([
#         html.H4("Cara Membaca Fibonacci Retracement", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Fibonacci Retracement mengidentifikasi level dukungan dan resistensi potensial berdasarkan rasio Fibonacci.", className="mb-2"),
#             html.Li("Level umum yang digunakan: 23.6%, 38.2%, 50%, 61.8%, dan 78.6%.", className="mb-2"),
#             html.Li("Level ini dihitung dari titik tertinggi (swing high) dan terendah (swing low) dalam periode tertentu.", className="mb-2"),
#             html.Li("Level 61.8% dan 38.2% adalah level Fibonacci yang paling signifikan untuk pembalikan.", className="mb-2"),
#             html.Li("Dalam tren naik, level-level ini sering bertindak sebagai dukungan ketika harga mengalami koreksi.", className="mb-2"),
#             html.Li("Dalam tren turun, level-level ini sering bertindak sebagai resistensi ketika harga mengalami rally.", className="mb-2"),
#             html.Li(html.Strong("Sinyal beli:"), " Ketika harga mencapai level Fibonacci (khususnya 61.8% atau 78.6%) dan menunjukkan tanda-tanda pembalikan naik.", className="mb-2"),
#             html.Li(html.Strong("Sinyal jual:"), " Ketika harga mencapai level Fibonacci dalam rally dan menunjukkan tanda-tanda pembalikan turun.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),


#     'Candlestick_Signal': html.Div([
#         html.H3("Candlestick Pattern Analysis", className="text-lg font-bold mb-2"),
#         html.P([
#             "Candlestick patterns are formations on price charts that can indicate potential market reversals, continuations, or indecision. ",
#             "This indicator detects multiple candlestick patterns and generates signals based on the prevalence of bullish vs bearish patterns."
#         ], className="mb-4"),
#         html.Div([
#             html.H4("Key Concepts:", className="font-bold mb-1"),
#             html.Ul([
#                 html.Li("Bullish Patterns: Formations that typically indicate potential upward price movement (e.g., Hammer, Morning Star, Bullish Engulfing)"),
#                 html.Li("Bearish Patterns: Formations that typically indicate potential downward price movement (e.g., Shooting Star, Evening Star, Bearish Engulfing)"),
#                 html.Li("Neutral Patterns: Formations that indicate indecision (e.g., Doji, Spinning Top)"),
#                 html.Li("Pattern Confidence: Higher when multiple patterns confirm the same direction"),
#             ], className="list-disc ml-6 mb-3")
#         ]),
#         html.Div([
#             html.H4("How to Use:", className="font-bold mb-1"),
#             html.Ul([
#                 html.Li("Look for clusters of patterns in the same direction"),
#                 html.Li("Pay attention to patterns that form at key support/resistance levels"),
#                 html.Li("Use the confidence threshold to filter out weaker signals"),
#                 html.Li("Combine with other technical indicators for confirmation"),
#             ], className="list-disc ml-6 mb-3")
#         ]),
#         html.Div([
#             html.H4("Interpretation:", className="font-bold mb-1"),
#             html.Ul([
#                 html.Li("Buy Signal: Generated when bullish patterns outnumber bearish patterns"),
#                 html.Li("Sell Signal: Generated when bearish patterns outnumber bullish patterns"),
#                 html.Li("Hold Signal: Generated when the pattern count is equal or below the confidence threshold"),
#             ], className="list-disc ml-6 mb-3")
#         ]),
#     ], className="p-4 bg-blue-50 border border-blue-200 rounded-lg mt-2"),
    
    
#     'combination': html.Div([
#         html.H4("Cara Membaca Grafik Kombinasi", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Grafik kombinasi menunjukkan pergerakan harga bersama dengan berbagai indikator teknikal.", className="mb-2"),
#             html.Li("Perhatikan sinyal yang muncul secara bersamaan dari beberapa indikator untuk konfirmasi yang lebih kuat.", className="mb-2"),
#             html.Li("Sinyal beli yang kuat terjadi ketika beberapa indikator memberikan sinyal beli pada waktu yang sama.", className="mb-2"),
#             html.Li("Sinyal jual yang kuat terjadi ketika beberapa indikator memberikan sinyal jual pada waktu yang sama.", className="mb-2"),
#             html.Li("Perhatikan volume untuk mengkonfirmasi kekuatan dari pergerakan harga dan sinyal indikator.", className="mb-2"),
#             html.Li("Divergensi antara harga dan indikator (terutama RSI dan MACD) dapat mengindikasikan pembalikan trend.", className="mb-2"),
#             html.Li("Tren naik diidentifikasi ketika harga membentuk puncak dan lembah yang lebih tinggi.", className="mb-2"),
#             html.Li("Tren turun diidentifikasi ketika harga membentuk puncak dan lembah yang lebih rendah.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
#     'all_graph': html.Div([
#         html.H4("Cara Membaca Semua Grafik", className="text-lg font-semibold mt-4 mb-2"),
#         html.Ul([
#             html.Li("Halaman ini menampilkan seluruh grafik indikator teknikal secara terpisah untuk analisis yang lebih mendalam.", className="mb-2"),
#             html.Li("Gunakan perbandingan antar grafik untuk mencari konfirmasi dari beberapa indikator.", className="mb-2"),
#             html.Li("Perhatikan tren harga pada grafik utama dan bagaimana indikator-indikator merespons pergerakan tersebut.", className="mb-2"),
#             html.Li("Cari momen di mana beberapa indikator memberikan sinyal pada waktu yang sama untuk sinyal perdagangan yang lebih kuat.", className="mb-2"),
#             html.Li("Untuk analisis yang lebih lengkap, perhatikan volume perdagangan bersama dengan sinyal dari indikator lain.", className="mb-2"),
#             html.Li("RSI dan MACD sering digunakan untuk mengidentifikasi divergensi dengan harga, yang bisa mengindikasikan pembalikan tren.", className="mb-2"),
#             html.Li("ADX berguna untuk mengkonfirmasi kekuatan tren yang terdeteksi oleh indikator lain seperti Moving Average.", className="mb-2"),
#         ], className="list-disc pl-5"),
#     ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
# }




from dash import html

# Kamus penjelasan untuk setiap indikator teknikal dalam Bahasa Indonesia
indicator_explanations = {
    'Bollinger_Signal': html.Div([
        html.H4("Cara Membaca Bollinger Bands", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Bollinger Bands adalah indikator volatilitas yang membantu mengidentifikasi kondisi jenuh beli (overbought) dan jenuh jual (oversold) serta potensi pembalikan harga.", className="mb-3"),
        html.Ul([
            html.Li("Terdiri dari garis tengah (SMA) dengan dua pita di atas dan di bawahnya yang mewakili volatilitas.", className="mb-2"),
            html.Li("Ketika harga mendekati pita atas, saham mungkin dalam kondisi 'overbought' (jenuh beli).", className="mb-2"),
            html.Li("Ketika harga mendekati pita bawah, saham mungkin dalam kondisi 'oversold' (jenuh jual).", className="mb-2"),
            html.Li("Penyempitan pita menunjukkan volatilitas rendah, sering mendahului pergerakan harga yang signifikan.", className="mb-2"),
            html.Li("Pelebaran pita menunjukkan volatilitas tinggi.", className="mb-2"),
            html.Li([html.Strong("Sinyal beli: "), "Ketika harga menyentuh atau menembus pita bawah dan kemudian bergerak kembali ke atas."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual: "), "Ketika harga menyentuh atau menembus pita atas dan kemudian bergerak kembali ke bawah."], className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'MA_Signal': html.Div([
        html.H4("Cara Membaca Moving Average (MA)", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Moving Average (MA) adalah indikator tren yang menghaluskan data harga untuk menunjukkan arah tren jangka pendek dan jangka panjang.", className="mb-3"),
        html.Ul([
            html.Li("MA meratakan fluktuasi harga untuk menunjukkan tren yang lebih jelas.", className="mb-2"),
            html.Li("MA jangka pendek (garis biru) lebih responsif terhadap perubahan harga terbaru.", className="mb-2"),
            html.Li("MA jangka panjang (garis merah) menunjukkan tren jangka panjang yang lebih stabil.", className="mb-2"),
            html.Li([html.Strong("Sinyal beli (Golden Cross): "), "Ketika MA jangka pendek memotong ke atas MA jangka panjang."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual (Death Cross): "), "Ketika MA jangka pendek memotong ke bawah MA jangka panjang."], className="mb-2"),
            html.Li("Harga di atas kedua MA menunjukkan tren naik yang kuat.", className="mb-2"),
            html.Li("Harga di bawah kedua MA menunjukkan tren turun yang kuat.", className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'RSI_Signal': html.Div([
        html.H4("Cara Membaca Relative Strength Index (RSI)", className="text-lg font-semibold mt-4 mb-2"),
        html.P("RSI adalah indikator momentum yang mengukur kecepatan dan perubahan pergerakan harga untuk mengidentifikasi kondisi overbought atau oversold.", className="mb-3"),
        html.Ul([
            html.Li("RSI bergerak dalam skala 0 hingga 100.", className="mb-2"),
            html.Li(f"Nilai di atas 70 menunjukkan kondisi 'overbought' (jenuh beli), mengindikasikan potensi pembalikan turun.", className="mb-2"),
            html.Li(f"Nilai di bawah 30 menunjukkan kondisi 'oversold' (jenuh jual), mengindikasikan potensi pembalikan naik.", className="mb-2"),
            html.Li("Nilai 50 adalah garis tengah yang menunjukkan keseimbangan antara kekuatan pembeli dan penjual.", className="mb-2"),
            html.Li([html.Strong("Sinyal beli: "), f"Ketika RSI turun di bawah 30 dan kemudian naik kembali di atas 30."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual: "), f"Ketika RSI naik di atas 70 dan kemudian turun kembali di bawah 70."], className="mb-2"),
            html.Li("Divergensi: Ketika harga membentuk puncak/lembah baru tetapi RSI tidak, bisa menjadi indikasi pembalikan arah.", className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'MACD_Signal': html.Div([
        html.H4("Cara Membaca Moving Average Convergence Divergence (MACD)", className="text-lg font-semibold mt-4 mb-2"),
        html.P("MACD adalah indikator momentum tren yang menunjukkan hubungan antara dua moving average harga aset.", className="mb-3"),
        html.Ul([
            html.Li("Terdiri dari garis MACD (biru), garis sinyal (merah), dan histogram.", className="mb-2"),
            html.Li("Histogram positif (hijau) menunjukkan momentum naik (garis MACD di atas garis sinyal).", className="mb-2"),
            html.Li("Histogram negatif (merah) menunjukkan momentum turun (garis MACD di bawah garis sinyal).", className="mb-2"),
            html.Li([html.Strong("Sinyal beli: "), "Ketika garis MACD (biru) memotong ke atas garis sinyal (merah)."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual: "), "Ketika garis MACD (biru) memotong ke bawah garis sinyal (merah)."], className="mb-2"),
            html.Li("MACD di atas garis nol menunjukkan tren umum sedang naik, di bawah nol menunjukkan tren turun.", className="mb-2"),
            html.Li("Divergensi: Ketika harga dan MACD bergerak ke arah berlawanan, bisa mengindikasikan pembalikan tren.", className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'ADX_Signal': html.Div([
        html.H4("Cara Membaca Average Directional Index (ADX)", className="text-lg font-semibold mt-4 mb-2"),
        html.P("ADX adalah indikator yang digunakan untuk mengukur kekuatan sebuah tren, baik itu tren naik maupun turun, bukan arahnya.", className="mb-3"),
        html.Ul([
            html.Li("ADX terdiri dari 3 garis: ADX (oranye), +DI (hijau), dan -DI (merah).", className="mb-2"),
            html.Li("Garis ADX menunjukkan kekuatan tren secara keseluruhan.", className="mb-2"),
            html.Li("Garis +DI menunjukkan kekuatan tren naik, sementara -DI menunjukkan kekuatan tren turun.", className="mb-2"),
            html.Li(f"ADX di atas 25 menunjukkan tren yang kuat. Semakin tinggi nilainya, semakin kuat trennya.", className="mb-2"),
            html.Li(f"ADX di bawah 20 menunjukkan pasar sedang sideways atau tren yang lemah.", className="mb-2"),
            html.Li([html.Strong("Sinyal beli: "), "Ketika garis +DI (hijau) memotong ke atas garis -DI (merah) dan ADX di atas 25."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual: "), "Ketika garis -DI (merah) memotong ke atas garis +DI (hijau) dan ADX di atas 25."], className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'Volume_Signal': html.Div([
        html.H4("Cara Membaca Analisis Volume", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Analisis volume digunakan untuk mengukur minat pasar dan mengkonfirmasi kekuatan sebuah tren harga.", className="mb-3"),
        html.Ul([
            html.Li("Volume menunjukkan jumlah saham yang diperdagangkan dalam periode tertentu.", className="mb-2"),
            html.Li("Volume tinggi (di atas rata-rata) mengkonfirmasi kekuatan tren saat ini.", className="mb-2"),
            html.Li("Volume rendah (di bawah rata-rata) menunjukkan kurangnya minat dan bisa menandakan tren yang lemah.", className="mb-2"),
            html.Li("Lonjakan volume yang tiba-tiba sering mendahului pergerakan harga yang signifikan atau pembalikan arah.", className="mb-2"),
            html.Li([html.Strong("Konfirmasi Tren Naik: "), "Harga naik disertai volume yang juga meningkat."], className="mb-2"),
            html.Li([html.Strong("Konfirmasi Tren Turun: "), "Harga turun disertai volume yang meningkat."], className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'Fibonacci_Signal': html.Div([
        html.H4("Cara Membaca Fibonacci Retracement", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Fibonacci Retracement adalah alat untuk mengidentifikasi level support dan resistance potensial di mana harga kemungkinan akan berbalik arah.", className="mb-3"),
        html.Ul([
            html.Li("Level ini digambar antara titik terendah (swing low) dan tertinggi (swing high) dari sebuah tren.", className="mb-2"),
            html.Li("Level kunci yang digunakan: 23.6%, 38.2%, 50%, 61.8%, dan 78.6%.", className="mb-2"),
            html.Li("Dalam tren naik, level-level ini bertindak sebagai 'support' (lantai) potensial saat harga koreksi.", className="mb-2"),
            html.Li("Dalam tren turun, level-level ini bertindak sebagai 'resistance' (atap) potensial saat harga naik sementara.", className="mb-2"),
            html.Li([html.Strong("Sinyal beli: "), "Ketika harga turun ke salah satu level Fibonacci (misal 61.8%) dalam tren naik dan menunjukkan tanda-tanda akan berbalik naik."], className="mb-2"),
            html.Li([html.Strong("Sinyal jual: "), "Ketika harga naik ke salah satu level Fibonacci dalam tren turun dan menunjukkan tanda-tanda akan berbalik turun."], className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),

    'Candlestick_Signal': html.Div([
        html.H4("Cara Membaca Analisis Pola Candlestick", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Pola candlestick adalah formasi pada grafik harga yang dapat mengindikasikan potensi pembalikan arah pasar, kelanjutan tren, atau keraguan pasar.", className="mb-3"),
        html.Div([
            html.H5("Konsep Utama:", className="font-semibold mb-1"),
            html.Ul([
                html.Li("Pola Bullish: Formasi yang biasanya mengindikasikan potensi kenaikan harga (contoh: Hammer, Morning Star, Bullish Engulfing).", className="mb-2"),
                html.Li("Pola Bearish: Formasi yang biasanya mengindikasikan potensi penurunan harga (contoh: Shooting Star, Evening Star, Bearish Engulfing).", className="mb-2"),
                html.Li("Pola Netral: Formasi yang menunjukkan keraguan di pasar (contoh: Doji, Spinning Top).", className="mb-2"),
            ], className="list-disc pl-5 mb-3")
        ]),
        html.Div([
            html.H5("Interpretasi Sinyal:", className="font-semibold mb-1"),
            html.Ul([
                html.Li([html.Strong("Sinyal beli: "), "Dihasilkan ketika jumlah pola bullish lebih dominan daripada pola bearish."], className="mb-2"),
                html.Li([html.Strong("Sinyal jual: "), "Dihasilkan ketika jumlah pola bearish lebih dominan daripada pola bullish."], className="mb-2"),
                html.Li([html.Strong("Sinyal tahan (Hold): "), "Dihasilkan ketika jumlah pola seimbang atau tidak ada pola yang signifikan."], className="mb-2"),
            ], className="list-disc pl-5 mb-3")
        ]),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'combination': html.Div([
        html.H4("Cara Membaca Grafik Kombinasi", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Grafik ini menggabungkan pergerakan harga dengan beberapa indikator teknikal sekaligus untuk memberikan gambaran yang komprehensif.", className="mb-3"),
        html.Ul([
            html.Li("Cari konfirmasi sinyal: sebuah sinyal dianggap lebih kuat jika didukung oleh beberapa indikator secara bersamaan.", className="mb-2"),
            html.Li("Contoh sinyal beli kuat: Harga memantul dari pita bawah Bollinger, RSI keluar dari area oversold, dan MACD melakukan golden cross.", className="mb-2"),
            html.Li("Contoh sinyal jual kuat: Harga ditolak oleh pita atas Bollinger, RSI masuk ke area overbought, dan MACD melakukan death cross.", className="mb-2"),
            html.Li("Perhatikan volume untuk mengkonfirmasi kekuatan dari setiap sinyal yang muncul.", className="mb-2"),
            html.Li("Identifikasi divergensi antara harga dan indikator (RSI/MACD) sebagai sinyal awal potensi pembalikan tren.", className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
    
    'all_graph': html.Div([
        html.H4("Cara Menganalisis Semua Grafik", className="text-lg font-semibold mt-4 mb-2"),
        html.P("Halaman ini menampilkan setiap grafik indikator secara terpisah untuk analisis yang lebih mendalam dan perbandingan.", className="mb-3"),
        html.Ul([
            html.Li("Analisis Tren Utama: Gunakan grafik harga dengan Moving Average untuk menentukan tren utama (naik, turun, atau sideways).", className="mb-2"),
            html.Li("Konfirmasi Momentum: Gunakan RSI dan MACD untuk mengukur kekuatan atau momentum di balik tren tersebut.", className="mb-2"),
            html.Li("Ukur Volatilitas: Gunakan Bollinger Bands untuk melihat apakah pasar sedang tenang atau bergejolak.", className="mb-2"),
            html.Li("Validasi Kekuatan Tren: Gunakan ADX untuk memastikan apakah sebuah tren cukup kuat untuk diikuti.", className="mb-2"),
            html.Li("Konfirmasi dengan Volume: Selalu periksa grafik volume. Sinyal yang disertai volume tinggi lebih dapat diandalkan.", className="mb-2"),
            html.Li("Cari Sinyal Gabungan: Keputusan trading terbaik seringkali datang dari konfirmasi beberapa indikator yang memberikan sinyal searah.", className="mb-2"),
        ], className="list-disc pl-5"),
    ], className="bg-gray-50 p-4 rounded-lg mt-2 mb-4"),
}