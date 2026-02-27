import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
    accuracy_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML App with EDA", layout="centered")
st.title("Classification & Regression App with Descriptive Statistics")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)

# ---------------- DATA PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ================= DESCRIPTIVE STATISTICS =================
st.header("Descriptive Statistics & Exploratory Data Analysis")

numeric_cols = df.select_dtypes(include="number")
categorical_cols = df.select_dtypes(exclude="number")

# ---- NUMERIC STATS ----
if not numeric_cols.empty:
    st.subheader("Numeric Columns Summary")
    st.dataframe(numeric_cols.describe())

    col_num = st.selectbox("Select Numeric Column for Graphs", numeric_cols.columns)

    # Histogram
    st.markdown("**Histogram**")
    fig, ax = plt.subplots()
    ax.hist(df[col_num], bins=20)
    st.pyplot(fig)

    # Box plot
    st.markdown("**Box Plot**")
    fig, ax = plt.subplots()
    ax.boxplot(df[col_num], vert=False)
    st.pyplot(fig)

    # Correlation heatmap
    if numeric_cols.shape[1] > 1:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---- CATEGORICAL STATS ----
if not categorical_cols.empty:
    st.subheader("Categorical Columns Summary")
    st.dataframe(categorical_cols.describe())

    col_cat = st.selectbox("Select Categorical Column for Bar Chart", categorical_cols.columns)

    st.markdown("**Category Distribution**")
    fig, ax = plt.subplots()
    df[col_cat].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ================= MODEL SELECTION =================
st.header("Model Training")

model_type = st.radio(
    "Choose Model Type",
    ["Classification", "Regression"]
)

# ---------------- FILTER TARGET COLUMNS ----------------
if model_type == "Classification":
    target_options = [
        col for col in df.columns
        if df[col].dtype == object or df[col].nunique() <= 10
    ]
else:
    target_options = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

target_col = st.selectbox("Select Target Column", target_options)

# ---------------- FEATURE SELECTION ----------------
feature_options = [
    col for col in df.columns
    if col != target_col and pd.api.types.is_numeric_dtype(df[col])
]

feature_cols = st.multiselect("Select Feature Columns", feature_options)

train_percent = st.slider("Training Data Percentage", 50, 90, 70)

# ---------------- CLASSIFIER SELECTION ----------------
if model_type == "Classification":
    classifier_name = st.selectbox(
        "Choose Classifier",
        [
            "Naive Bayes",
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Decision Tree"
        ]
    )

# ================= TRAIN MODEL =================
if st.button("Train Model"):

    X = df[feature_cols]
    y_raw = df[target_col]

    if model_type == "Classification":

        if y_raw.dtype != object and y_raw.nunique() > 10:
            y = pd.qcut(y_raw, q=3, labels=["Low", "Medium", "High"])
        else:
            y = y_raw.astype(str)

        if classifier_name == "Naive Bayes":
            model = GaussianNB()
        elif classifier_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif classifier_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        else:
            model = DecisionTreeClassifier(random_state=42)

    else:
        y = y_raw
        model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(100 - train_percent) / 100, random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ---------------- OUTPUT ----------------
    if model_type == "Classification":
        st.subheader("Accuracy")
        st.write("Train:", accuracy_score(y_train, y_train_pred))
        st.write("Test:", accuracy_score(y_test, y_test_pred))

        st.subheader("Confusion Matrix (Test)")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax)
        st.pyplot(fig)

    else:
        st.subheader("Regression Metrics")
        mse = mean_squared_error(y_test, y_test_pred)
        st.write("MSE:", mse)
        st.write("RMSE:", mse ** 0.5)
        st.write("R²:", r2_score(y_test, y_test_pred))

    st.success("Model trained successfully.")