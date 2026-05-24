# 🏦 Loan Default Risk Predictor

Predicts the probability of a borrower experiencing **serious financial distress** within two years using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

---

## 📂 Project Structure

```
ML_PROJECT/
├── .github/workflows/        ← CI/CD (GitHub Actions)
├── artifacts/                ← Saved model & preprocessor (auto-generated)
├── notebook/
│   ├── data/                 ← Place cs-training.csv here
│   └── EDA.ipynb             ← Exploratory Data Analysis
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       ← Load & split data
│   │   ├── data_transformation.py  ← Preprocessing + SMOTE
│   │   └── model_trainer.py        ← Train & compare 4 models
│   ├── pipeline/
│   │   ├── train_pipeline.py       ← End-to-end training
│   │   └── predict_pipeline.py     ← Inference pipeline
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py                    ← Streamlit application
├── requirements.txt
└── setup.py
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ML_PROJECT.git
cd ML_PROJECT

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📥 Dataset

1. Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)
2. Place it in `notebook/data/cs-training.csv`

---

## 🚀 Train the Model

```bash
python src/pipeline/train_pipeline.py
```

This will:
- Ingest & split data → `artifacts/`
- Apply median imputation + StandardScaler + SMOTE
- Train **Logistic Regression, Decision Tree, Random Forest, XGBoost**
- Save the best model by ROC-AUC → `artifacts/model.pkl`

---

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

**App Features:**
| Page | Description |
|------|-------------|
| 🔍 Single Prediction | Predict default risk for one borrower |
| 📂 Batch Prediction  | Upload CSV and predict for multiple borrowers |
| 📊 Model Performance | Compare all models, feature importances |

---

## 🧠 ML Concepts Used

| Concept | Implementation |
|---------|---------------|
| Classification | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| Imbalanced Data | SMOTE oversampling |
| Feature Engineering | Median imputation, StandardScaler |
| Model Evaluation | ROC-AUC, Accuracy, F1, Precision, Recall |
| Hyperparameter Tuning | Custom params per model |
| Explainability | Feature importances |
| Pipelines | sklearn Pipeline + ColumnTransformer |

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `RevolvingUtilizationOfUnsecuredLines` | Credit card usage ratio |
| `age` | Borrower age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days late |
| `DebtRatio` | Monthly debt / income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Total open credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days late |
| `NumberOfDependents` | Number of dependents |

---

## 🛠 Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`
