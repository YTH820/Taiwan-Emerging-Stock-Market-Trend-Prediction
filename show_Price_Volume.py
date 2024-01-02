# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:42:39 2023
@author: thyou
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Setting matplotlib configuration for Chinese characters
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

def load_and_preprocess_data(file_path):
    """
    Load data from a CSV file and preprocess for analysis.
    """
    df = pd.read_csv(file_path, encoding="cp950")
    df['price'].fillna(method='ffill', inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def plot_stock_data(df, stock_name):
    """
    Plot stock price and volume data.
    """
    fig, axs = plt.subplots(2, figsize=(24, 12), height_ratios=[3, 1])

    axs[0].plot(df['date'], df['price'], label='Actual Price')
    axs[0].set_ylabel('Price')
    axs[0].set_title(f"{stock_name} - Stock Profit Prediction - Train and Test")
    axs[0].legend()

    axs[1].bar (df['date'], df['成交量'], label='Actual Volume')
    axs[1].set_ylabel('Volume')
    axs[1].legend()

    plt.xlabel('Dates')
    plt.show()

# Main Execution
file_path = 'E:/EMBA/20231221/seq_orign/4923力士.csv'
stock_name = file_path.split(".")[-2].split("/")[-1]

df_original = load_and_preprocess_data(file_path)
plot_stock_data(df_original, stock_name)
