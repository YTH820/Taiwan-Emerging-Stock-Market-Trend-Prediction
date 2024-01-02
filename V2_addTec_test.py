import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
    technical_indicators = ["SMA_5" , "SMA_20", "EMA_5", "EMA_20"] 
    X_tech = df[technical_indicators] .values
    X_volumes = df[volume_features].values
    y = df['Outcome'].values
    
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(0, 1))
    X_tech_scaled = scaler.fit_transform(X_tech)
    X_volumes_scaled = scaler.fit_transform(X_volumes)

    X_combined = np.stack((X_tech_scaled, X_volumes_scaled), axis=-1)
    return X_combined, y

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, input_shape=input_shape))
    model.add(Dense(20))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

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
# =============================================================================
# 
# model = build_model(input_shape=(X_train.shape[1], 1))
# history = train_model(model, X_train, y_train, X_test, y_test)
# 
# model.summary()
# 
# plot_metrics(history)
# predictions = model.predict(X_test)
# plot_confusion_matrix(y_test, predictions)
# =============================================================================
