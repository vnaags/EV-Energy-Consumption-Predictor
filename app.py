import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EV_Energy_Consumption_Dataset.csv")
    return df

# -------------------------
# Train Model
# -------------------------
def train_model(df, target_col):
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' is not numeric. Please choose a numeric column.")

    X = numeric_df.drop(target_col, axis=1)
    y = numeric_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X.columns.tolist()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="EV Energy Consumption Predictor", layout="centered")

st.title("⚡ EV Energy Consumption Predictor")

# Load Data
df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.write("All columns in dataset:", df.columns.tolist())

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_columns:
    st.error("No numeric columns found in dataset! Please check your CSV.")
else:
    target_col = st.selectbox("The target column:", numeric_columns)

    if target_col:
        model, mse, r2, features = train_model(df, target_col)

        st.write(f"✅ Model trained with R² = {r2:.2f}, MSE = {mse:.2f}")

        # -------------------------
        # User Input for Prediction
        # -------------------------
        st.subheader("🔮 Predict")
        user_inputs = {}
        for feature in features:
            val = st.number_input(
                f"Enter {feature}",
                value=float(df[feature].mean()) if pd.api.types.is_numeric_dtype(df[feature]) else 0.0
            )
            user_inputs[feature] = val

        input_df = pd.DataFrame([user_inputs])

        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            st.success(f"⚡ Predicted {target_col}: {prediction:.2f}")
