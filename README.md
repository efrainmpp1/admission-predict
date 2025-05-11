# 🧠 Project 01 – Binary Classification | PPGEEC2318

This repository contains the implementation of **Project 01 - Binary Classification**, developed for the course **PPGEEC2318 - Machine Learning**, instructed by Professor **Ivanovitch Medeiros**.

---

## 📌 Project Overview

The goal of this project is to build a simple **binary classification model** using **PyTorch**. The classification task is based on the "Admission Prediction" problem, where we use a dataset containing applicants’ profiles and attempt to predict whether each applicant was admitted or not.

Throughout this project, we will:

- Explore and preprocess the dataset;
- Build and train a neural network for binary classification;
- Evaluate the model’s performance;
- Document our findings and results clearly.

This project aims to provide a hands-on introduction to binary classification using PyTorch and to reinforce fundamental machine learning concepts.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Pandas
- Matplotlib / Seaborn

---

# 📄 Model Card – Admission Prediction

Model cards are concise documents that provide context and transparency about a machine learning model. This model card presents the key aspects of the binary classification model developed in this project, following documentation best practices.

---

## 🧠 Model Details

This model was developed by Efrain Marcelo Pulgar Pantaleon as part of the PPGEEC2318 course. It uses a neural network built with **PyTorch**, structured around a reusable [`Architecture`](https://github.com/ivanovitchm/PPGEEC2318/blob/main/lessons/week05/week05c.ipynb) class to manage training, validation, prediction, and logging.

The repository structure:

- `preprocess/`: data cleaning, transformation and encoding.
- `train/`: model architecture, loss function, optimizer, and training loop.
- `evaluate/`: classification metrics and confusion matrix.
- `check_model/`: overfitting analysis and SMOTE balancing strategy.

To handle class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data only.

---

## 🎯 Intended Use

This model is for **educational purposes only**. It demonstrates:

- The full ML pipeline with deep learning,
- The impact of preprocessing, and
- The handling of imbalanced binary classification using SMOTE.

Not intended for real-world decision-making in academic admissions.

---

## 🧪 Training and Evaluation Data

The dataset contains features such as:

- GPA, GMAT, Work Experience (numeric)
- Gender, Race, International status, Major, Work Industry (categorical)

Target:

- `1`: Admit
- `0`: NotAdmit or Waitlist

The data was split using `train_test_split` (80/20). Only the training data was oversampled using **SMOTE** to address class imbalance.

### 📊 Exploratory Data Analysis (EDA)

An [EDA](./pipeline/eda/exploratory_analisis.ipynb) was conducted prior to modeling, and the following observations were made:

- The dataset was **highly imbalanced**, with only ~15% of samples labeled as `Admit`.
- **GPA and GMAT** had a positive correlation with admission outcomes, with admitted students showing higher average scores.
- **Work Experience** showed little correlation with admission status.
- Categorical features such as **Major** and **Work Industry** showed uneven distributions across classes.
- A correlation matrix revealed that **GPA** and **GMAT** were moderately correlated (`r ≈ 0.58`), while `Work Experience` was uncorrelated with both.

Visualizations included:

- Histograms and boxplots for numeric features grouped by admission status,
- Countplots for categorical variables segmented by class,
- A heatmap of the numerical correlation matrix,
- A confusion matrix after model evaluation.

These analyses were essential to understand feature importance, inform preprocessing strategies, and guide model design decisions.

---

## 📈 Metrics

Evaluation was done using the original (unbalanced) test data.

| Metric    | Value (example) |
| --------- | --------------- |
| Accuracy  | 0.75            |
| Precision | 0.37            |
| Recall    | 0.85            |
| F1 Score  | 0.52            |

A confusion matrix is also generated for visual performance analysis.

---

## ⚖️ Ethical Considerations

Caution should be exercised when interpreting features like race, gender, and profession as indicators of merit. These variables are socially constructed and can reflect historical bias.

The model is not intended for use in any real admission processes.

---

## ⚠️ Caveats and Recommendations

- SMOTE improved recall but reduced precision (trade-off).
- Dataset remains imbalanced despite mitigation.
- Future improvements could include:
  - ROC/AUC analysis
  - Threshold tuning
  - Class weight penalties
  - Explainability tools (e.g., SHAP, LIME)
