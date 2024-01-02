# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:01:21 2023
@author: thyou
"""

import os
import pandas as pd
import matplotlib
from datetime import datetime

# Set matplotlib configuration for Chinese characters
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

def load_csv(file_path):
    """ Load a CSV file and return a DataFrame. """
    df = pd.read_csv(file_path, encoding="cp950")
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    #df['pre_m_price'] = df['price'].shift(20)
    #df['profit'] = (df['price'] - df['pre_m_price']) / df['pre_m_price']
    return df

def preprocess_data(df):
    """ Preprocess the DataFrame for LSTM model. """
    df_lstm = df.copy()
    df_lstm.fillna(method='ffill', inplace=True)
    #df_lstm.drop(columns=['pre_m_price'], inplace=True)
    return df_lstm

def create_lstm_dataset(df, period=20):
    """ Create dataset for LSTM model. """
    data = []
    non_trade_period = 60  # To avoid repeated trades within 20 days

    for i in range(period, len(df) - period - 1):
        volume_mean = df['筆數'][i - period: i].mean()
        if volume_mean < 20 or df['筆數'][i - period: i].isna().any() or df['最後'][i - period: i].isna().any():
            continue

        current_volume = df['筆數'][i]
        if current_volume > 5 * volume_mean:
            input_volumes = df['成交量'][i - period: i].tolist()
            input_prices = df['最後'][i - period: i].tolist()
            buy_price = df['最後'][i + 1]
            future_price = df['最後'][i + period + 1] # 計算方式要修改為平均
            input_highest = df['最高'][i - period: i].tolist()
            input_lowest = df['最低'][i - period: i].tolist()

            if i < non_trade_period:
                continue

            output = 1 if future_price >= buy_price * 1.02 else 0
            non_trade_period = i + period
            date_obj = df['date'][i + period]
            formatted_date = date_obj.strftime('%Y-%m-%d')
            #data.append([formatted_date] + input_volumes + input_prices + input_highest + input_lowest  + [output])
            data.append([formatted_date] + input_volumes + input_prices + input_highest + input_lowest  + [output])

    columns = ["買入"] + ['Volume_Day_' + str(i) for i in range(1, period + 1)] + \
              ['Price_Day_' + str(i) for i in range(1, period + 1)] + \
              ['Highest_Day_' + str(i) for i in range(1, period + 1)] + \
              ['Lowest_Day_' + str(i) for i in range(1, period + 1)] + \
              ['Outcome']
    return pd.DataFrame(data, columns=columns)

def save_to_csv(df, file_name):
    """ Save DataFrame to a CSV file. """
    if not df.empty:
        df.to_csv(file_name, index=False, encoding='cp950')

# Main execution
folder_path = "E:/EMBA/20231221/seq_orign/"
files = os.listdir(folder_path)

for file in files:
    file_path = os.path.join(folder_path, file)
    stock_name = os.path.splitext(file)[0]
    
    df_original = load_csv(file_path)
    df_lstm = preprocess_data(df_original)
    df_for_lstm = create_lstm_dataset(df_lstm)

    output_file = f"./new_data/{stock_name}.csv"
    save_to_csv(df_for_lstm, output_file)
