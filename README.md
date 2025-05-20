# 🛌 AI Sleep Apnea Detector

An intelligent, web-based platform to detect obstructive sleep apnea using biomedical signal processing and ensemble machine learning.

---

## 📁 Project Overview

Developed by graduate students at **University of Maryland, Baltimore County (UMBC)**, this project classifies 30-second biomedical signal windows to detect sleep apnea events using ECG, EEG, and BP signals from the **MIT-BIH Polysomnographic Database (PhysioNet)**.

> 🚀 Live Goal: Real-time sleep apnea detection from uploaded WFDB files with interpretable results.

---

## 👨‍🔬 Team Members

- **Afshaan Khan**
- **Saroj Kothakota**
- **Siri Harshitha Karimilla**

---

## 🧠 Technologies Used

| Backend       | ML/DS Libraries            | Frontend        | Utilities       |
|---------------|----------------------------|------------------|-----------------|
| Flask         | XGBoost, Random Forest     | Tailwind CSS     | WFDB, SciPy     |
| Python 3.11   | StackingClassifier, Optuna | Chart.js         | SHAP            |
| Jupyter       | SMOTE, SMOTEENN            | AOS.js (Animation) | pickle         |

---

## 📊 Dataset Details

- **Name**: MIT-BIH Polysomnographic Sleep Apnea Database
- **Source**: [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/)
- **Instances**: ~70 recordings
- **Sampling Rate**: 100 Hz
- **Channels**: ECG, EEG, BP
- **Labels**: Apnea / Non-Apnea

---

## 🧪 EDA Highlights

- Class Imbalance: ~7% apnea, 93% normal → Resolved using SMOTE & SMOTEENN
- Most apnea events occur in clusters during early-mid sleep cycles
- ECG entropy and RMS showed highest separability

---

## 🔍 Model Pipeline

```WFDB File Upload ➜ Signal Processing ➜ 30-sec Windows ➜ Feature Extraction ➜ SMOTE ➜ Stacking Ensemble (XGBoost + RF + LR) ➜ Isotonic Calibration ➜ Output Diagnosis```

---

## 📈 Evaluation Metrics

| Metric              | Value    |
|---------------------|----------|
| ✅ Accuracy          | **82.84%** |
| 🎯 Macro F1 Score    | **58.92%** |
| 📌 Precision (Apnea) | **0.62**   |
| 📈 Recall (Apnea)    | **0.59**   |
| 📊 AUC-ROC           | **0.81**   |

## 🔑 Key Takeaways

- Combined biomedical signal processing with advanced ensemble models (XGBoost + Random Forest).
- 🩻 Used 30-second windows of ECG, EEG, and BP signals for detecting apneatic patterns.
- ⚖️ Handled class imbalance using SMOTE and SMOTEENN to improve recall on apnea cases.
- 📊 Achieved **~82.84% accuracy** and **~0.81 AUC** using calibrated StackingClassifier.
- 💡 Integrated SHAP for explainability and model interpretability.
- 🌐 Deployed a fully functional web app using **Flask + Tailwind CSS + Chart.js**.
- 📈 Visualizes apnea prediction window-by-window and shows severity using AHI metrics.
- Planned integration of LLM-powered AI Assistant in future updates.

## 🔮 Future Work
- Integrate LLM-powered chatbot for guidance
- PDF reports with AHI and SHAP insights
- Real-time mobile/IoT device integration
- Compare with CNN + Transformer-based architectures

## 📜 References
	1.	Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 2000.
	2.	XGBoost: A Scalable Tree Boosting System. Chen & Guestrin, KDD ’16
	3.	SMOTE: Synthetic Minority Oversampling Technique. Chawla et al.

## Sleep smarter. Live stronger. 🛌✨
