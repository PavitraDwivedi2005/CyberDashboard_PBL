# 🛡️ CyberShield — AI-Powered Threat Detection Dashboard

CyberShield is a professional-grade network security intelligence platform that leverages Machine Learning to identify and classify malicious network traffic in real-time. Built with **Streamlit** and **Scikit-Learn**, it provides deep insights into network flows based on the **CICIDS-2017** dataset.

![Dashboard Preview](https://img.shields.io/badge/Status-Operational-success?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)

## 🚀 Features

- **Real-Time Classification**: Classify network flows as Benign or Malicious with high precision.
- **Threat Intelligence**: Interactive distribution charts and sequential threat scoring.
- **Model Explainability**: Top feature importance analysis to understand why threats were flagged.
- **Performance Metrics**: Detailed evaluation including Confusion Matrix, ROC Curves, and F1 Scores.
- **Secure by Design**: Automatic data cleaning, infinity handling, and feature alignment.

## 🛠️ Technical Stack

- **Model**: Random Forest Classifier
- **Frontend**: Streamlit (with custom Glassmorphism CSS)
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📦 Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PavitraDwivedi2005/CyberDashboard_PBL.git
   cd CyberDashboard_PBL
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset Requirements

The model is trained on the **CICIDS-2017** schema. To use the dashboard:
1. Upload a CSV file containing network traffic features.
2. Ensure columns match the standard flow feature names (e.g., `Destination Port`, `Flow Duration`, etc.).
3. The app will automatically handle missing columns by filling them with `0`.

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard with custom dark-theme styling.
- `model.pkl`: Pre-trained Random Forest model.
- `scaler.pkl`: StandardScaler instance for feature normalization.
- `features.pkl`: List of expected features for alignment.
- `data_small.csv`: Sample dataset for testing.

---

*Developed for PBL — Cyber Intelligence & Threat Detection.*
