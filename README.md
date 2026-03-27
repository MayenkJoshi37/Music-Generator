# 🎵 AI Music Generator (LSTM)

## 📌 Overview

This project generates MIDI music using an LSTM neural network.
It also provides basic analytics like mood detection and pitch metrics.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend (Flask)

```bash
python backend_api.py
```

### 3. Start Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 📂 Project Structure

### 🔹 backend_api.py

* Flask server
* Loads trained model
* Generates music
* Returns MIDI + analytics

### 🔹 create.py

* Core music generation logic
* Handles sequence prediction
* Converts output to notes

### 🔹 model.py

* Defines and trains LSTM model
* Saves trained model (`.h5`)

### 🔹 app.py

* Streamlit frontend
* UI for generating music
* Displays mood + metrics + playback

---

## 📊 Features

* 🎼 MIDI music generation
* 😊 Mood detection (happy/sad/calm/energetic)
* 📈 Basic analytics (pitch, repetition, etc.)
* ▶️ Play and download generated music

---

## ⚠️ Notes

* Model file (`.h5`) is required to run backend
* First run may take time (model loading)

---

## 🛠 Tech Stack

* TensorFlow / Keras
* Music21
* Flask
* Streamlit
