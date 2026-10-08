# autoencoder-ecg-anomaly-detection

A deep learning project in **TensorFlow/Keras** for real-time time-series anomaly detection on ECG signals using a Bottleneck Autoencoder architecture and **t-SNE** latent space visualization.

## 📌 Overview

Traditional supervised models require extensive labeled anomaly data. This project trains an Autoencoder exclusively on *normal* heartbeat patterns. When an anomalous ECG is passed through the network, the reconstruction loss (Mean Absolute Error) spikes significantly, providing a reliable threshold for anomaly detection.

## 🛠️ Tech Stack & Libraries

- **Frameworks:** TensorFlow, Keras
- **Data Processing:** Python, NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, t-SNE (t-Distributed Stochastic Neighbor Embedding)

## 🧠 Model Architecture

- **Encoder:** Dense(32, ReLU) ➔ Dense(16, ReLU) ➔ Dense(8, ReLU)
- **Decoder:** Dense(16, ReLU) ➔ Dense(32, ReLU) ➔ Dense(140, Sigmoid)
- **Anomaly Threshold:** Calculated as **Mean Loss + 1 Std Dev** ($\mu + 1\sigma$) on normal training reconstructions.

## 📊 Key Results

- **MAE Reconstruction Loss:** Low error on normal ECG inputs vs. high error spikes on anomalous signals.
- **t-SNE Latent Space Analysis:** Reduced the 8-dimensional bottleneck vectors to 2D space, demonstrating clear cluster separation between normal and anomalous heartbeats.

## 📸 Visual Output

<img width="761" alt="Original vs. Reconstructed ECG signals" src="https://github.com/user-attachments/assets/0913fd9e-bea2-4412-b452-c8bd81c88720" />
<p><em>Original vs. Reconstructed ECG signals</em></p>

<img width="677" alt="2D t-SNE scatter plot" src="https://github.com/user-attachments/assets/027c5a79-4c0f-4d38-8657-452bc04e69f8" />
<p><em>2D t-SNE scatter plot of normal vs. anomalous clusters</em></p>

