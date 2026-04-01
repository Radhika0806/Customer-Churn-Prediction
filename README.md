# Bank Customer Churn Prediction

A machine learning application that predicts the probability of bank customer churn. This Streamlit-based app helps identify customers at risk of leaving the bank.

## Project Overview

This project uses a Random Forest classifier to predict customer churn in the banking sector. The model analyzes customer demographics, account information, and banking behavior to estimate the likelihood of churn.

## Dataset

- **Source:** Bank Customer Churn Prediction dataset
- **Size:** 10,000 customer records
- **Features:** 11 features including:
  - Customer demographics (age, gender, country)
  - Account information (credit score, balance, tenure)
  - Engagement metrics (products number, credit card, active member status)
  - Financial information (estimated salary)

## Features & Preprocessing

The model includes sophisticated preprocessing:
- **Categorical Encoding:** OneHotEncoder for country and gender
- **Numerical Scaling:** StandardScaler for all numerical features
- **Age Normalization:** Ages over 70 are capped at 70
- **Tenure Grouping:** Customer tenure is grouped into 3 categories
- **Class Imbalance Handling:** SMOTENC (Synthetic Minority Oversampling) for balanced training

## Model Performance

- **Algorithm:** Random Forest Classifier
- **ROC-AUC Score:** 0.85
- **Accuracy:** ~82%
- **Data Split:** 80% training, 20% testing

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Customer Churn Banking"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the data file is in the same directory:**
   ```
   Bank Customer Churn Prediction.csv
   ```

## Running the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

The app will:
1. Open in your default browser at `http://localhost:8501`
2. Load or train the model on first run
3. Display an interactive interface for making predictions

## How to Use

1. **Enter Customer Details** in the sidebar:
   - Credit Score (300-850)
   - Country (France, Spain, Germany)
   - Gender (Male, Female)
   - Age (18-92)
   - Tenure (0-10 years)
   - Account Balance
   - Number of Products
   - Credit Card Status
   - Active Member Status
   - Estimated Salary

2. **View Prediction Results:**
   - Risk Assessment (High/Low Churn Risk)
   - Churn Probability (%)
   - Probability Distribution Chart

## Model Details

The Random Forest classifier was selected based on comparative analysis of multiple algorithms:
- KNN: ROC-AUC 0.80
- SVM: Not suitable for probability output
- Decision Tree: ROC-AUC 0.70
- Logistic Regression: ROC-AUC 0.76
- **Random Forest: ROC-AUC 0.85** ← Best Performer

## File Structure

```
Customer Churn Banking/
├── app.py                                  # Streamlit application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── Bank Customer Churn Prediction.csv      # Training data
├── Customer_Churn_Prediction.ipynb        # Analysis notebook
└── churn_model.pkl                        # Trained model (auto-generated)
```

## Screenshots

| Input Form | Prediction Result |
|-----------|-------------------|
| ![Input](screenshots/input.png) | ![Prediction](screenshots/prediction.png) |

> **To add screenshots:** Run `streamlit run app.py`, fill in customer details, and save screenshots of the form and prediction in the `screenshots/` folder.

## Key Features of the App

- **st.cache_resource:** Efficient model caching to avoid retraining
- **Automated Training:** Creates fresh model from CSV if pickle files don't exist
- **Interactive UI:** User-friendly sidebar for parameter input
- **Visual Feedback:** Color-coded predictions and probability charts
- **Responsive Design:** Works on desktop and mobile browsers

## Data Preprocessing Pipeline

1. Load CSV data
2. Cap ages > 70 to 70
3. Group tenure into categories (0: ≤3 years, 1: 4-6 years, 2: ≥7 years)
4. Encode categorical features (country, gender)
5. Scale numerical features
6. Apply SMOTENC to handle class imbalance
7. Train Random Forest on balanced data

## Troubleshooting

**"Data file not found" error:**
- Ensure `Bank Customer Churn Prediction.csv` is in the same directory as `app.py`

**Slow first run:**
- The app trains a fresh model on first run if no pickle files exist
- Subsequent runs will load the cached model (very fast)

**Port already in use:**
- Run with a different port: `streamlit run app.py --server.port 8502`

## Future Improvements

- Add model performance metrics dashboard
- Implement hyperparameter tuning interface
- Add feature importance visualization
- Export predictions to CSV
- Add batch prediction capability
- Include model retraining feature

## License

This project is provided as-is for educational and business use.

## Contact & Support

For issues or questions, please refer to the main project documentation or contact the development team.
