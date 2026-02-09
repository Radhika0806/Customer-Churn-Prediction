# 📊 Customer Churn Prediction in Banking | Machine Learning Project

## 📌 Overview

This project builds an **end-to-end machine learning pipeline** to predict **customer churn in the banking domain**. The focus is on **data quality, exploratory data analysis (EDA), feature engineering, class imbalance handling, and model evaluation**, with the goal of delivering **actionable business insights** rather than just model accuracy.

---

## 🎯 Business Problem

Customer churn directly impacts revenue and long-term growth in banking.
This project aims to:

* Predict customers likely to churn
* Identify key behavioral and engagement drivers
* Support **data-driven customer retention strategies**

---

## 🧾 Dataset Description

The dataset contains structured banking customer data, including:

* Customer demographics (Age, Geography, Gender)
* Financial attributes (Credit Score, Balance, Salary)
* Behavioral indicators (Active Member, Credit Card Usage, Product Count)
* Target variable: **Churn (Binary Classification)**

**Key Challenge:** Severe class imbalance (~80% non-churn vs ~20% churn)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was used to guide modeling decisions and business interpretation.

Key findings:

* Presence of **outliers and unrealistic age values**
* Skewed salary distribution affecting summary statistics
* Strong churn association with **low engagement indicators**
* Majority of customers hold **only 1–2 products**
* Imbalanced churn distribution requiring metric-aware modeling

---

## 🛠️ Data Preprocessing & Feature Engineering

* Data cleaning and validation
* Handling missing and inconsistent values
* Feature scaling and transformation
* Encoding categorical variables
* Addressing class imbalance
* Prevention of data leakage

---

## 🤖 Machine Learning Models

* Multiple supervised learning models explored
* Binary classification framework
* Emphasis on **interpretability and business usability**

---

## 📈 Model Evaluation

Due to class imbalance, evaluation prioritized:

* Confusion Matrix
* Precision, Recall, and F1-Score
* Class-specific performance analysis

> **Accuracy was intentionally deprioritized** to avoid misleading conclusions.

---

## 💡 Key Insights

* Churn is more strongly driven by **customer engagement** than by income alone
* Inactive customers with fewer products show higher churn risk
* Data quality issues significantly impact churn prediction reliability
* Predictive models must be paired with **business logic** to be actionable

---

## 🚀 Business Impact

This solution can be applied to:

* Customer retention and loyalty programs
* Risk-based customer segmentation
* Early churn warning systems
* Strategic decision-making in retail banking

---

## 🧰 Tech Stack

**Programming:** Python
**Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
**Environment:** Jupyter Notebook
**Concepts:**

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Class Imbalance Handling
* Supervised Machine Learning
* Model Evaluation Metrics

---
## How to Run

**1. Clone the repository**
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

**2. (Optional) Create and activate virtual environment**
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

**3. Install required dependencies**
pip install -r requirements.txt

**4. Launch the Jupyter Notebook**
jupyter notebook

**5. Open and run the notebook**
Open: customer_churn_prediction.ipynb
Run all cells sequentially to reproduce EDA, preprocessing, and model results

---

## 🔮 Future Improvements

* SMOTE / cost-sensitive learning
* Model explainability (SHAP, feature importance)
* Model deployment using Flask or Streamlit
* Integration with real-time banking data

---

## 🏁 Conclusion

This project demonstrates a **complete data science workflow**, combining **statistical analysis, machine learning, and business reasoning**. The emphasis is on building models that are **interpretable, reliable, and actionable**, making the project suitable for real-world banking applications.

---

### 🔑 ATS Keywords Embedded

Data Analysis · Machine Learning · Customer Churn · Banking Analytics · EDA · Feature Engineering · Python · Scikit-learn · Model Evaluation · Business Insights · Classification · Imbalanced Data · Predictive Analytics
