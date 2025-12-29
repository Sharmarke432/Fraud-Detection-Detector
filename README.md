# Credit Card Fraud Detection – Unsupervised Anomaly Detection

This repository contains an end-to-end **credit card fraud detection** project using **unsupervised learning** on the well-known Kaggle dataset. [web:260][web:264]

The goal is to detect **rare fraudulent transactions (~0.17%)** without relying on labeled data at training time, and to compare two anomaly detection algorithms:

- **Isolation Forest** – tree-based model that isolates anomalies quickly in feature space  
- **Local Outlier Factor (LOF)** – density-based model that detects points in low-density regions  

The project includes:

- 🔍 **Exploratory Data Analysis (EDA)** on class imbalance and feature distributions  
- 🧹 **Data preprocessing** (scaling, train/test split with stratification)  
- 🤖 **Unsupervised models**: Isolation Forest and LOF, tuned for extreme imbalance  
- 📊 **Evaluation** using AUPRC, ROC-AUC, precision, recall, and confusion matrices  
- 🖼️ **Visualizations**: score distributions, PR/ROC curves, PCA anomaly plots  
- 💰 **Business impact analysis**: estimate fraud savings vs investigation cost  
- 🌐 **(Optional)** Streamlit dashboard for interactive model exploration and demo  

This repo is designed as a **portfolio-ready machine learning project** showing how to tackle **highly imbalanced fraud detection** with unsupervised methods, clean code, and clear storytelling. [web:253][web:263][web:267]
