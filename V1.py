import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_and_preprocess_data(folder_path):
    files = os.listdir(folder_path)
    X_all, y_all = [], []

    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file), encoding='cp950')
        X, y = preprocess_data(df)
        X_all.append(X)
        y_all.append(y)

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def preprocess_data(df):
    price_features = [col for col in df.columns if 'Price_Day_' in col]
    volume_features = [col for col in df.columns if 'Volume_Day_' in col]
    highest_features = [col for col in df.columns if 'Highest_Day_' in col]
    lowest_features = [col for col in df.columns if 'Lowest_Day_' in col]
    
    X_prices = df[price_features].values
    X_volumes = df[volume_features].values
    X_highest = df[highest_features].values
    X_lowest = df[lowest_features].values
    X_diff = X_highest - X_lowest  # Calculate the difference
    
    y = df['Outcome'].values
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(0, 1))
    X_prices_scaled = scaler.fit_transform(X_prices)
    X_volumes_scaled = scaler.fit_transform(X_volumes)
    X_highest_scaled = scaler.fit_transform(X_highest)
    X_lowest_scaled = scaler.fit_transform(X_lowest)
    X_diff_scaled = scaler.fit_transform(X_diff)
    

    #X_combined = np.stack((X_prices_scaled, X_volumes_scaled, X_highest_scaled, X_lowest_scaled), axis=-1)
    #X_combined = np.stack((X_prices_scaled, X_volumes_scaled), axis=-1)
    X_combined = np.stack((X_volumes_scaled), axis=0)
    return X_combined, y

def build_model(input_shape):


      model = Sequential()
      model.add(LSTM(units=256, input_shape=input_shape))
      model.add(Dense(100))
      model.add(Dropout(0.5))
      model.add(Dense(1, activation='sigmoid'))  # Binary classification
      return model
# 
# =============================================================================
# =============================================================================
#     model = Sequential()
# 
#     # 第一個雙向LSTM層，帶有正則化
# =============================================================================
#      model.add(Bidirectional(LSTM(units=256, return_sequences=True,
#                                   kernel_regularizer=l1_l2(l1=0.01, l2=0.01)), 
#                                   input_shape=input_shape))
#      model.add(Dropout(0.5))
#      model.add(BatchNormalization())
# # 
# #     # 第二個LSTM層
#      model.add(LSTM(units=128, return_sequences=True))
#      model.add(Dropout(0.5))
#      model.add(BatchNormalization())
# # 
# #     # 第三個LSTM層
#      model.add(LSTM(units=64))
#      model.add(Dropout(0.5))
#      model.add(BatchNormalization())
# # 
# #     # 密集連接層
#      model.add(Dense(64, activation='relu'))
#      model.add(Dropout(0.5))
#  
# #     # 輸出層，用於二元分類
#      model.add(Dense(1, activation='sigmoid'))
# =============================================================================
#     return model
    

def train_model(model, X_train, y_train, X_test, y_test):
    callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr if epoch < 10 else lr * tf.math.exp(-0.1)
    )
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for binary classification
    return model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[callback])

def plot_metrics(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_test, predictions):
    predicted_classes = (predictions > 0.5).astype(int)  # Threshold for binary classification
    cm = confusion_matrix(y_test, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Main execution
folder_path = "E:/EMBA/20231221/new_data/"
X_all, y_all = load_and_preprocess_data(folder_path)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

model = build_model(input_shape=(X_train.shape[1], 1))
model.summary()
history = train_model(model, X_train, y_train, X_test, y_test)



plot_metrics(history)
predictions = model.predict(X_test)
plot_confusion_matrix(y_test, predictions)
