# autoencoder-ecg-anomaly-detection

```markdown
# Autoencoder Anomaly Detection & Latent Space Mapping

A deep learning project in **TensorFlow/Keras** for real-time time-series anomaly detection on ECG signals using a Bottleneck Autoencoder architecture and **t-SNE** latent space visualization.

## 📌 Overview

Traditional supervised models require extensive labeled anomaly data. This project trains an Autoencoder exclusively on *normal* heartbeat patterns. When an anomalous ECG is passed through the network, the reconstruction loss (Mean Absolute Error) spikes significantly, providing a reliable threshold for anomaly detection.

## 🛠️ Tech Stack & Libraries

* **Frameworks:** TensorFlow, Keras
* **Data Processing:** Python, NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, t-SNE (t-Distributed Stochastic Neighbor Embedding)

## 🧠 Model Architecture

* **Encoder:** Dense(32, ReLU) ➔ Dense(16, ReLU) ➔ Dense(8, ReLU)
* **Decoder:** Dense(16, ReLU) ➔ Dense(32, ReLU) ➔ Dense(140, Sigmoid)
* **Anomaly Threshold:** Calculated as $\mu_{\text{loss}} + 1\sigma_{\text{loss}}$ on normal training reconstructions.

## 📊 Key Results

* **MAE Reconstruction Loss:** Low error on normal ECG inputs vs. high error spikes on anomalous signals.
* **t-SNE Latent Space Analysis:** Reduced the 8-dimensional bottleneck vectors to 2D space, demonstrating clear cluster separation between normal and anomalous heartbeats.

## 📸 Visual Output

*(Tip: Drag and drop your generated matplotlib plots here)*

* <img width="761" height="362" alt="image" src="https://github.com/user-attachments/assets/0913fd9e-bea2-4412-b452-c8bd81c88720" />
 — Original vs. Reconstructed ECG signals.
* <img width="677" height="527" alt="image" src="https://github.com/user-attachments/assets/027c5a79-4c0f-4d38-8657-452bc04e69f8" />
 — 2D t-SNE scatter plot of normal vs. anomalous clusters.

## 🏃 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/krishishah296/autoencoder-ecg-anomaly-detection.git](https://github.com/krishishah296/autoencoder-ecg-anomaly-detection.git)
   cd autoencoder-ecg-ano
