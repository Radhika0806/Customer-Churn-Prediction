import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC


@st.cache_resource
def train_model():
    df = pd.read_csv("Bank Customer Churn Prediction.csv")

    # Same preprocessing as notebook: cap age at 70
    df["age"] = df["age"].apply(lambda x: 70 if x > 70 else x)

    X = df.drop(["churn", "customer_id"], axis=1)
    y = df["churn"]

    # Label-encode categoricals for SMOTENC
    cat_cols = ["country", "gender"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    cat_indices = [X.columns.get_loc(col) for col in cat_cols]

    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    pipe.fit(X_resampled, y_resampled)

    return pipe, label_encoders


def main():
    st.title("Bank Customer Churn Prediction")
    st.write("Predict whether a bank customer will churn using a Random Forest model.")

    pipe, label_encoders = train_model()

    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, format="%.2f")
        products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        credit_card = st.selectbox("Has Credit Card", [0, 1], index=1)
        active_member = st.selectbox("Active Member", [0, 1], index=1)
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=100000.0, format="%.2f"
        )

    if st.button("Predict Churn"):
        # Cap age at 70 like notebook
        age_capped = 70 if age > 70 else age

        # Encode categoricals the same way as training
        country_encoded = label_encoders["country"].transform([country])[0]
        gender_encoded = label_encoders["gender"].transform([gender])[0]

        input_data = pd.DataFrame(
            {
                "credit_score": [credit_score],
                "country": [country_encoded],
                "gender": [gender_encoded],
                "age": [age_capped],
                "tenure": [tenure],
                "balance": [balance],
                "products_number": [products_number],
                "credit_card": [credit_card],
                "active_member": [active_member],
                "estimated_salary": [estimated_salary],
            }
        )

        prediction = pipe.predict(input_data)[0]
        probability = pipe.predict_proba(input_data)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**Churn** - Probability: {probability[1]:.2%}")
        else:
            st.success(f"**No Churn** - Probability of staying: {probability[0]:.2%}")

        st.write(f"Churn probability: {probability[1]:.2%}")
        st.write(f"No-churn probability: {probability[0]:.2%}")


if __name__ == "__main__":
    main()
