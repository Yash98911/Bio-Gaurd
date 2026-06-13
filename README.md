# 🛩️ BioGuard MRO Dashboard
### AI-Powered Aircraft Engine Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-89.81%25_CV_Accuracy-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Dataset](https://img.shields.io/badge/Dataset-NASA_CMAPSS-orange)

## Overview
BioGuard is an industry-grade **Predictive Maintenance** system that monitors aircraft engine health in real-time and predicts failures before they occur — enabling proactive maintenance decisions.

Built on NASA's CMAPSS dataset, the system classifies each engine's condition as **SAFE 🟢**, **WARNING 🟡**, or **CRITICAL 🔴** based on live sensor readings.

---

## Problem Statement
Traditional maintenance is either:
- **Reactive** — fix after failure (dangerous & expensive)
- **Scheduled** — fixed intervals regardless of condition (wasteful)

**BioGuard enables Condition-Based Maintenance** — predict failure before it happens, act only when needed.

---

## Methodology

### 1. Data Preprocessing
- NASA CMAPSS FD001 — 100 engines, 20,631 readings, 21 sensors
- Cleaned extra whitespace columns
- Calculated RUL (Remaining Useful Life) per engine cycle

### 2. Condition Labeling
| Label | Condition | RUL |
|---|---|---|
| 0 | 🟢 SAFE | > 50 cycles |
| 1 | 🟡 WARNING | 16–50 cycles |
| 2 | 🔴 CRITICAL | ≤ 15 cycles |

### 3. Sensor Selection
- Std deviation analysis on all 21 sensors
- Visual distribution shift analysis — SAFE vs CRITICAL
- Selected 8 key sensors: `s2, s3, s4, s7, s9, s12, s14, s17`

### 4. Time Series Feature Engineering
Applied rolling window (10 cycles) for each sensor:
- Rolling Mean — gradual degradation trend
- Rolling Std — sensor instability detection
- Rolling Min — minimum threshold breach
- Rolling Max — peak value detection

**Result: 8 sensors × 4 features = 32 time series features → 40 total features**

### 5. Model Comparison
| Model | Test Accuracy |
|---|---|
| Logistic Regression | 95% |
| Random Forest | 95% |
| **XGBoost** | **95% 🏆** |
| SVM | 90% |

### 6. Final Evaluation
| Metric | Score |
|---|---|
| Test Accuracy | 98% |
| Cross-Validated Accuracy | 89.81% ± 0.76% |
| Critical → Safe Misclassification | 0 ✅ |

---

## Live Demo
🔗 [BioGuard MRO Dashboard](https://bio-gaurd-tz7ravyqose768h2sxn26m.streamlit.app/)

---

## Tech Stack
| Category | Tools |
|---|---|
| Language | Python |
| ML | XGBoost, Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Deployment | Streamlit Cloud |

---

## Project Structure
```
BioGuard/
├── app.py                 ← Streamlit dashboard
├── bioguard_model.pkl     ← Trained XGBoost pipeline
├── feature_cols.pkl       ← Feature column names
├── df_clean.csv           ← Processed dataset
├── requirements.txt       ← Dependencies
└── README.md
```

---

## Real World Applications
- ✈️ Commercial Aviation MRO
- 🪖 Defence & DRDO Research
- 🏭 Industrial IoT & Manufacturing
- ⚡ Power Plant Maintenance

---

## Author
**Yash Dhollakhandi**
B.Tech — Automation & Robotics | Delhi
