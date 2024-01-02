"""
Created on Thu Dec 21 15:01:21 2023

@author: thyou
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import os 

# 解決中文問題
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 或其他支持中文的字體



# Load the CSV file to examine its structure

file_path = 'E:/EMBA/20231221/seq_orign/5876上海商銀.csv'
 
Stock_name = file_path.split(".")[-2].split("/")[-1]
# 載入數據
df_original = pd.read_csv(file_path)

# 數據處理
# 使用前一天的價格填充NaN值
df_original['price'].fillna(method='ffill', inplace=True)
df_original['date'] = pd.to_datetime(df_original['date'], format='%Y%m%d')
#df_original.set_index('date', inplace=True)
df_original['pre_m_price'] = df_original['price'].shift(20)
df_original['profit'] = (df_original['price'] - df_original['pre_m_price']) / df_original['pre_m_price']
#df_original.drop(columns=['pre_m_price'], inplace=True)

# 創建一個新的DataFrame來處理LSTM模型的數據
df_lstm = df_original.copy()
#df_lstm.drop(columns=['price'], inplace=True)
df_lstm.drop(columns=['pre_m_price'], inplace=True)
#df_lstm.dropna(inplace=True)

# 現在 df_original 保留了原始數據，df_lstm 用於LSTM模型
df_original.head(), df_lstm.head()

# 選擇特徵和標籤
X = df_lstm[['volume','price']]
y = df_lstm['profit']



period = 20
data = []
non = 0 # 用來避免20天內重複交易

for i in range(period, len(X) - period):
    # 計算前20天的平均成交量
    volume_mean = X['volume'][i-period : i].mean()
    
    # 獲取當天的成交量
    cur_volume = X['volume'][i]

    # 如果當天成交量大於前20天平均的5倍
    if cur_volume > 5 * volume_mean:
        # 記錄輸入數據（前20天的成交量）
        input_volumes = X['volume'][i-period : i].tolist()
        input_prices = X['price'][i-period : i].tolist()
        
        # 計算買入價格（第21天的收盤價）
        buy_price = X['price'][i+1]

        # 判斷未來20天內是否漲幅超過2%
        # future_prices = X['price'][i + period + 1:i + period * 2 + 1]
        # max_future_price = future_prices.max()
        
        future_price = X['price'][i + period + 1]
        if i < non:
            continue
        if future_price >= buy_price * 1.02:
            non = i + 20
            output = 1
        else:
            output = 0
            
        # 將日期字符串轉換為 datetime 對象
        date_obj = df_original['date'][i + period]    
        
        #date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
        
        # 將 datetime 對象格式化為 'YYYY-MM-DD' 格式的字符串
        formatted_date = date_obj.strftime('%Y-%m-%d')
        
        # 添加到數據列表
        data.append( [str(formatted_date)] + input_volumes + input_prices  + [output])


columns = ["買入"] + ['Volume_Day_' + str(i) for i in range(1, period + 1)] + ['Outcome']
# 將數據轉換為 DataFrame 並保存到 CSV 檔案
columns = ["買入"] + \
          ['Volume_Day_' + str(i) for i in range(1, period + 1)] + \
          ['Price_Day_' + str(i) for i in range(1, period + 1)] + \
          ['Outcome']
          
df = pd.DataFrame(data, columns=columns)
df.to_csv(f"./new_data/{Stock_name}.csv", index=False, encoding = 'cp950')
