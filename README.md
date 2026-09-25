# Machine Learning Playground

A structured laboratory and implementation repository showcasing hands-on Machine Learning experiments, exploratory data analysis (EDA), model training, evaluation, and inference.

---

## Lab Modules Overview

| No. | Module / Experiment | Problem Type | Algorithm / Framework | Key Metrics / Evaluation | Status |
|---|---|---|---|---|---|
| 01 | [California Housing Price Prediction](./01-california-housing-xgboost/) | Tabular Regression | XGBoost Regressor | R² Score, MAE, Feature Importance | Completed |
| 02 | Diabetes Risk Classification | Binary Classification | Support Vector Machine / Logistic Regression | Accuracy, Precision, Recall | Planned |
| 03 | Underwater Sonar Detection (Mines vs Rocks) | Signal Classification | Logistic Regression / Neural Net | Accuracy Score, Confusion Matrix | Planned |

---

## Module 01: California Housing Price Prediction

- **Directory:** [`01-california-housing-xgboost/`](./01-california-housing-xgboost/)
- **Objective:** Predict the median value of owner-occupied homes (`MedHouseVal`) across California districts based on demographic, economic, and geographical variables.
- **Dataset:** California Housing Dataset (20,640 records, 8 numeric features).
- **Core Pipeline:**
  1. Data ingestion & missing-value diagnostics.
  2. Exploratory Data Analysis (EDA) with a feature correlation heatmap.
  3. 80:20 Train-Test split.
  4. Model training using `XGBRegressor`.
  5. Performance evaluation via R² Score and Mean Absolute Error (MAE).
  6. Feature Importance ranking to identify dominant price drivers (e.g., `MedInc` - Median Income).
  7. Actual vs. Predicted scatter plot with ideal diagonal fit reference.
  8. Single-sample inference simulation with currency conversion (USD to IDR).

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ardanrizky/machine-learning-playground.git
cd machine-learning-playground
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv env
# On Windows:
.\env\Scripts\activate

pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook
```
Navigate to `01-california-housing-xgboost/california_housing_prediction.ipynb` to view and run the notebook.

---

## Author
- **May Rizky Ardanata**  
  Applied Data Science Student at Electronic Engineering Polytechnic Institute of Surabaya (PENS)  
  GitHub: [@ardanrizky](https://github.com/ardanrizky) | LinkedIn: [May Rizky Ardanata](https://www.linkedin.com/in/may-rizky-ardanata-7001942b8/)
