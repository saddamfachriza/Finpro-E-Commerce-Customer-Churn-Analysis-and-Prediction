# Finpro-E-Commerce-Customer-Churn-Analysis-and-Prediction
Cartify E-Commerce Customer Churn Analysis and Prediction with Machine Learning

## Deskripsi Project
Cartify adalah perusahaan e-commerce yang mengalami peningkatan jumlah customer yang **churn** (berhenti bertransaksi). Project ini bertujuan membangun model machine learning yang mampu **memprediksi customer yang berpotensi churn secara proaktif**, sehingga tim retensi dapat mengambil tindakan sebelum customer benar-benar pergi.

## Business Problem
- Cartify belum memiliki sistem untuk mengidentifikasi customer yang akan churn
- Upaya retensi bersifat reaktif baru bertindak setelah customer pergi
- Biaya akuisisi customer baru 5x lebih mahal dibanding mempertahankan customer lama
- **False Negative (gagal mendeteksi churner) lebih merugikan** dibanding False Positive

## Metric Utama: F2-Score
F2-Score dipilih sebagai primary metric karena **Recall dua kali lebih penting dari Precision** dalam konteks churn prediction. Gagal mendeteksi customer yang akan churn (False Negative) jauh lebih merugikan bisnis dibanding salah prediksi customer yang tidak akan churn (False Positive).

## Dataset
| Info | Detail |
|---|---|
| Jumlah Data Awal | 5.630 baris, 20 kolom |
| Jumlah Data Setelah Cleaning | 5.074 baris, 19 kolom |
| Target Variable | `Churn` (0 = Not Churn, 1 = Churn) |
| Class Distribution | Not Churn: 83.4% \| Churn: 16.6% |
| Missing Values | 1.856 data di 7 kolom |
| Duplikat | 556 data (setelah drop CustomerID) |

## Tahapan Project
1. Business Problem Understanding
2. Exploratory Data Analysis (EDA)
3. Data Cleaning & Preprocessing
4. Data Analyst & Uji Statistik (Mann-Whitney & Chi-Square)
5. Modeling & Benchmark
6. Cross-Validation (dengan & tanpa Resampling)
7. Hyperparameter Tuning (RandomizedSearchCV)
8. Threshold Optimization
9. Final Model Evaluation
10. Feature Importance & SHAP Analysis
11. Conclusion & Business Recommendation

## Modeling
**Model yang Diuji**
- XGBoost
- CatBoost  
- LightGBM

**Strategi yang Dieksperimen**
| Strategi | Keterangan |
|---|---|
| No Resampling | Baseline |
| SMOTE | Oversampling kelas minoritas |
| SMOTETomek | SMOTE + pembersihan batas kelas |
| Class Weight (`scale_pos_weight`) | Pendekatan final |

**Model Terpilih**
> **XGBoost + No Resampling + Class Weight (scale_pos_weight=5)**


## Hasil Model Final
| Metric | Score |
|---|---|
| **F2-Score** | **0.9192** ← Primary Metric |
| **Recall** | **0.9345** |
| Precision | 0.8065 |
| F1-Score | 0.8657 |
| ROC-AUC | 0.9800 |
| PR-AUC | 0.9231 |
| Accuracy | 0.9527 |
| Threshold Optimal | 0.38 |

> Dari **168 customer churn aktual**, model berhasil mendeteksi **157 customer (93.5%)** dengan hanya 11 yang terlewat.
