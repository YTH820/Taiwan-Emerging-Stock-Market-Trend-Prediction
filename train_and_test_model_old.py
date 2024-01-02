# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 20:20:33 2023

@author: thyou
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加載數據
file_path = 'E:/EMBA/20231221/new_data/1594日高.csv'  # 請替換成您的文件路徑
df = pd.read_csv(file_path, encoding = 'cp950')

# 選擇特徵和標籤
features = [col for col in df.columns if 'Price_Day_' in col or 'Volume_Day_' in col]
X = df[features].values
y = df['Outcome'].values

# 正規化特徵
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)


# 重塑 X 以符合 LSTM 的輸入要求
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))



# 分割數據為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 定義 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# 查看模型摘要
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
