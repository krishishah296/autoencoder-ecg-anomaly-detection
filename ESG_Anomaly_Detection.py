# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:00:36 2026

@author: jenil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import Model, layers

# 1. Data Preparation
URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
df = pd.read_csv(URL, header=None)  # 140 signal columns + 1 label

data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Min-Max Scaling to [0, 1]
min_val = np.min(train_data)
max_val = np.max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

# Isolate normal data for training
normal_train_data = train_data[train_labels == 1]
normal_test_data = test_data[test_labels == 1]
anomalous_test_data = test_data[test_labels == 0]


# 2. Autoencoder Training and Anomaly Scoring
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

# Train the autoencoder exclusively on normal heartbeats
print("Training Autoencoder...")
history = autoencoder.fit(
    normal_train_data, normal_train_data,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_data),
    shuffle=True,
    verbose=1
)

# Score anomalies using Mean Absolute Error (MAE) reconstruction loss
train_reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(train_reconstructions, normal_train_data)

# Set threshold (mean + 1 standard deviation of normal training loss)
threshold = np.mean(train_loss) + np.std(train_loss)
print(f"Anomaly threshold set to: {threshold:.4f}")


# 3. Visualization Part A: Original vs Reconstructed ECGs
def plot_comparison(original, reconstructed, title):
    plt.figure(figsize=(10, 4))
    plt.plot(original, 'b.', label="Original ECG")
    plt.plot(reconstructed, 'r-', label="Reconstructed ECG")
    plt.fill_between(np.arange(140), reconstructed, original, color='lightcoral', alpha=0.3, label="Error")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

# Visualize Normal ECG Reconstruction
plot_comparison(
    normal_test_data[0], 
    autoencoder.predict(normal_test_data[0:1])[0], 
    "Normal ECG: Original vs Reconstructed (Low Error)"
)

# Visualize Anomalous ECG Reconstruction
plot_comparison(
    anomalous_test_data[0], 
    autoencoder.predict(anomalous_test_data[0:1])[0], 
    "Anomalous ECG: Original vs Reconstructed (High Error)"
)


# 4. Visualization Part B: t-SNE Latent Space Analysis
print("Generating t-SNE visualization...")
subset_size = 1000
X_subset = test_data[:subset_size]
y_subset = test_labels[:subset_size]

# Get the encoded bottleneck features (8-dimensional vectors)
encoded_features = autoencoder.encoder(X_subset).numpy()

# Apply t-SNE to reduce down to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(encoded_features)

# Plot the t-SNE results
plt.figure(figsize=(8, 6))
normal_mask = (y_subset == 1)
anomaly_mask = (y_subset == 0)

plt.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
            c='blue', label='Normal ECG', alpha=0.6, s=20)
plt.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1], 
            c='red', label='Anomalous ECG', alpha=0.8, s=30)

plt.title("t-SNE Visualization of ECG Latent Space")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.show()