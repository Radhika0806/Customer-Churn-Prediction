import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
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

        # --- Explainability Section ---
        st.header("Explainability")

        # Human-readable feature name mapping
        feature_descriptions = {
            "credit_score": "Credit Score",
            "country": "Country",
            "gender": "Gender",
            "age": "Customer's Age",
            "tenure": "Tenure (years with bank)",
            "balance": "Account Balance",
            "products_number": "Number of Products",
            "credit_card": "Has Credit Card",
            "active_member": "Active Member Status",
            "estimated_salary": "Estimated Salary",
        }

        # Get the preprocessed input and the Random Forest classifier
        preprocessor = pipe.named_steps["preprocessor"]
        clf = pipe.named_steps["classifier"]
        X_transformed = preprocessor.transform(input_data)

        # Retrieve transformed feature names
        cat_feature_names = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(["country", "gender"])
            .tolist()
        )
        num_feature_names = [
            c for c in input_data.columns if c not in ["country", "gender"]
        ]
        transformed_feature_names = cat_feature_names + num_feature_names

        # SHAP TreeExplainer on the Random Forest
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)

        # shap_values is [array_class0, array_class1] for binary classification
        # Use class-1 (churn) SHAP values
        sv_churn = shap_values[1][0]  # single sample, churn class

        # --- Waterfall plot for individual prediction ---
        st.subheader("SHAP Waterfall Plot")
        explanation = shap.Explanation(
            values=sv_churn,
            base_values=explainer.expected_value[1],
            data=X_transformed[0]
            if hasattr(X_transformed, "__getitem__")
            else X_transformed.toarray()[0],
            feature_names=transformed_feature_names,
        )
        fig_wf, ax_wf = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        fig_wf = plt.gcf()
        st.pyplot(fig_wf)
        plt.close("all")

        # --- Top 5 contributing features with plain English ---
        with st.expander("Why this prediction?", expanded=True):
            abs_shap = np.abs(sv_churn)
            top5_idx = np.argsort(abs_shap)[::-1][:5]

            st.markdown("**Top 5 features driving this prediction (toward churn):**")
            for rank, idx in enumerate(top5_idx, start=1):
                fname = transformed_feature_names[idx]
                # Map one-hot encoded names back to readable descriptions
                readable = fname
                for raw, desc in feature_descriptions.items():
                    if fname.startswith(raw):
                        readable = fname.replace(raw, desc)
                        break
                direction = "increases" if sv_churn[idx] > 0 else "decreases"
                st.write(
                    f"{rank}. **{readable}** — SHAP value "
                    f"{sv_churn[idx]:+.4f} ({direction} churn probability)"
                )

        # --- Global feature importance bar chart ---
        st.subheader("Global Feature Importance (SHAP)")
        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        fig_gi, ax_gi = plt.subplots(figsize=(8, 5))
        ax_gi.barh(
            [transformed_feature_names[i] for i in sorted_idx],
            importances[sorted_idx],
        )
        ax_gi.set_xlabel("Feature Importance")
        ax_gi.set_title("Random Forest — Global Feature Importance")
        ax_gi.invert_yaxis()
        fig_gi.tight_layout()
        st.pyplot(fig_gi)
        plt.close("all")


if __name__ == "__main__":
    main()
