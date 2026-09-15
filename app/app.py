import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =========================================================
# CUSTOM UI STYLING
# =========================================================

st.markdown("""
<style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Section headings */
    h2, h3 {
        font-weight: 600;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 18px;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Healthcare AI Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "healthcare_disease_risk_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "healthcare_scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    return model, scaler, feature_names


try:

    model, scaler, feature_names = load_model()
    model_loaded = True

except Exception as e:

    model_loaded = False
    st.error(f"Model loading error: {e}")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🏥 Healthcare AI")

    st.caption("Disease Risk Prediction Platform")

    st.divider()

    page = st.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Risk Prediction",
        "📊 Data Analysis",
        "📈 Model Performance",
        "🤖 Model Information",
        "ℹ️ About"
    ]
)

    st.divider()

    if model_loaded:
        st.success("🟢 ML Model Online")
    else:
        st.error("🔴 ML Model Offline")

    st.caption("Python • Scikit-learn • Streamlit")


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🏥 Healthcare AI Risk Prediction")

    st.subheader(
        "Intelligent disease-risk prediction powered by machine learning."
    )
    st.write(
        "Analyze patient health parameters and generate an "
        "estimated risk probability using Tuned XGBoost."
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Patient Features",
            len(feature_names)
        )

    with col2:
        st.metric(
            "🤖 ML Algorithm",
            "Tuned XGBoost"
        )

    with col3:
        st.metric(
            "⚡ Prediction",
            "Real-Time"
        )

    with col4:
        st.metric(
            "🟢 System",
            "Online" if model_loaded else "Offline"
        )

    st.divider()

    st.subheader("🚀 System Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🔍 Risk Prediction

            Enter patient health information and generate
            a machine-learning based risk prediction.

            **Input categories**

            - Age
            - BMI
            - Blood Pressure
            - Cholesterol
            - Glucose
            - Heart Rate
            - Smoking
            - Physical Activity
            - Family History
            """
        )

    with col2:

        st.markdown(
            """
            ### 🤖 Machine Learning

            The application uses a trained **Tuned XGBoost**
            classification model.

            **ML pipeline**

            1. Patient Input
            2. Data Encoding
            3. Feature Alignment
            4. Feature Scaling
            5. ML Prediction
            6. Risk Probability
            """
        )

    st.divider()

    st.info(
        "💡 Select **Risk Prediction** from the sidebar "
        "to test the trained machine learning model."
    )


# =========================================================
# RISK PREDICTION
# =========================================================

elif page == "🔍 Risk Prediction":

    st.title("🔍 Disease Risk Prediction")

    st.write(
        "Enter the patient's information below to generate "
        "an estimated disease-risk prediction."
    )

    st.divider()

    with st.form("prediction_form"):

        st.subheader("👤 Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=27.5,
                step=0.1
            )

        with col2:

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=50,
                max_value=250,
                value=130
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=500,
                value=210
            )

            glucose = st.number_input(
                "Glucose",
                min_value=40,
                max_value=500,
                value=115
            )

        with col3:

            heart_rate = st.number_input(
                "Heart Rate",
                min_value=30,
                max_value=220,
                value=78
            )

            smoking = st.selectbox(
                "Smoking",
                ["No", "Yes"]
            )

            physical_activity = st.selectbox(
                "Physical Activity",
                ["Low", "Medium", "High"]
            )

        family_history = st.selectbox(
            "Family History",
            ["No", "Yes"]
        )

        st.write("")

        predict_button = st.form_submit_button(
            "🔍 Predict Disease Risk",
            use_container_width=True
        )

        # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict_button:

        patient = {
            "Age": age,
            "BMI": bmi,
            "BloodPressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Glucose": glucose,
            "HeartRate": heart_rate,
            "Gender": gender,
            "Smoking": smoking,
            "PhysicalActivity": physical_activity,
            "FamilyHistory": family_history
        }

        patient_df = pd.DataFrame([patient])

        patient_encoded = pd.get_dummies(
            patient_df,
            columns=[
                "Gender",
                "Smoking",
                "PhysicalActivity",
                "FamilyHistory"
            ],
            drop_first=True
        )

        patient_encoded = patient_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        patient_scaled = scaler.transform(
            patient_encoded
        )

        prediction = model.predict(
            patient_scaled
        )[0]

        probability = model.predict_proba(
            patient_scaled
        )[0][1]

        risk_percentage = probability * 100

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")

        result1, result2, result3 = st.columns(3)

        with result1:

            if prediction == 1:
                st.error("⚠️ HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        with result2:

            st.metric(
                "Risk Probability",
                f"{risk_percentage:.2f}%"
            )

        with result3:

            st.metric(
                "Model",
                "Tuned XGBoost"
            )

        st.divider()

        st.subheader("📈 Risk Probability")

        st.progress(float(probability))

        st.write(
            f"**Estimated Probability: {risk_percentage:.2f}%**"
        )

        if prediction == 1:

            st.warning(
                "⚠️ The model estimates a higher disease-risk "
                "probability for the provided inputs."
            )

        else:

            st.info(
                "✅ The model estimates a lower disease-risk "
                "probability for the provided inputs."
            )

        st.divider()

        st.caption(
            "⚠️ Educational and demonstration purposes only. "
            "This prediction is not a medical diagnosis and should "
            "not replace advice from a qualified healthcare professional."
        )
# =========================================================
# DATA ANALYSIS
# =========================================================



elif page == "📊 Data Analysis":

    st.title("📊 Healthcare Data Analysis")

    st.write(
        "Explore the healthcare dataset through "
        "interactive visualizations and statistics."
    )

    dataset_path = (
        BASE_DIR
        / "data"
        / "raw"
        / "healthcare_disease_risk.csv"
    )

    if dataset_path.exists():

        data = pd.read_csv(dataset_path)
        st.success("✅ Healthcare dataset loaded successfully.")

        # Dataset summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Patients", f"{len(data):,}")

        with col2:
            st.metric("Total Features", data.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                f"{data.isnull().sum().sum():,}"
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                f"{data.duplicated().sum():,}"
            )

        st.divider()

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )
        st.divider()

        st.subheader("📊 Risk Distribution")

        if "Disease_Risk" in data.columns:

            risk_counts = data["Disease_Risk"].value_counts().reset_index()
            risk_counts.columns = ["Disease_Risk", "Count"]

            fig = px.bar(
                risk_counts,
                x="Disease_Risk",
                y="Count",
                title="Disease Risk Distribution",
                text="Count"
            )

            fig.update_layout(
                xaxis_title="Disease Risk",
                yaxis_title="Number of Patients"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            st.divider()

        st.subheader("🔎 Age vs BMI Analysis")

        if "Age" in data.columns and "BMI" in data.columns:

            fig_age_bmi = px.scatter(
                data,
                x="Age",
                y="BMI",
                color="Disease_Risk",
                title="Age vs BMI by Disease Risk",
                hover_data=[
                    "BloodPressure",
                    "Cholesterol",
                    "Glucose"
                ]
            )

            fig_age_bmi.update_layout(
                xaxis_title="Age",
                yaxis_title="BMI"
            )

            st.plotly_chart(
                fig_age_bmi,
                use_container_width=True
            )
            st.divider()

        st.subheader("🧪 Glucose vs Cholesterol Analysis")

        if "Glucose" in data.columns and "Cholesterol" in data.columns:

            fig_health = px.scatter(
                data,
                x="Glucose",
                y="Cholesterol",
                color="Disease_Risk",
                title="Glucose vs Cholesterol by Disease Risk",
                hover_data=[
                    "Age",
                    "BMI",
                    "BloodPressure",
                    "HeartRate"
                ]
            )

            fig_health.update_layout(
                xaxis_title="Glucose",
                yaxis_title="Cholesterol"
            )

            st.plotly_chart(
                fig_health,
                use_container_width=True
            )

       
        st.divider()

        # -------------------------------------------------
        # DATASET PREVIEW
        # -------------------------------------------------

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            data.head(20),
            use_container_width=True
        )

        st.divider()

        # -------------------------------------------------
        # DISEASE RISK DISTRIBUTION
        # -------------------------------------------------

        if "Disease_Risk" in data.columns:

            st.subheader("⚠️ Disease Risk Distribution")

            risk_counts = (
                data["Disease_Risk"]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = [
                "Disease_Risk",
                "Count"
            ]

            fig = px.bar(
                risk_counts,
                x="Disease_Risk",
                y="Count",
                title="Low Risk vs High Risk Patients",
                text="Count"
            )

            fig.update_layout(
                xaxis_title="Disease Risk",
                yaxis_title="Number of Patients"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # -------------------------------------------------
        # AGE DISTRIBUTION
        # -------------------------------------------------

        if "Age" in data.columns:

            st.subheader("👤 Age Distribution")

            fig_age = px.histogram(
                data,
                x="Age",
                nbins=20,
                title="Patient Age Distribution"
            )

            fig_age.update_layout(
                xaxis_title="Age",
                yaxis_title="Number of Patients"
            )

            st.plotly_chart(
                fig_age,
                use_container_width=True
            )

        # -------------------------------------------------
        # BMI DISTRIBUTION
        # -------------------------------------------------

        if "BMI" in data.columns:

            st.subheader("⚖️ BMI Distribution")

            fig_bmi = px.histogram(
                data,
                x="BMI",
                nbins=20,
                title="BMI Distribution"
            )

            fig_bmi.update_layout(
                xaxis_title="BMI",
                yaxis_title="Number of Patients"
            )

            st.plotly_chart(
                fig_bmi,
                use_container_width=True
            )

        # -------------------------------------------------
        # NUMERICAL FEATURES
        # -------------------------------------------------

        st.subheader("📈 Numerical Feature Analysis")

        numerical_columns = data.select_dtypes(
            include="number"
        ).columns.tolist()

        if "Disease_Risk" in numerical_columns:

            numerical_columns.remove("Disease_Risk")

        selected_feature = st.selectbox(
            "Select a feature",
            numerical_columns
        )

        fig_feature = px.histogram(
            data,
            x=selected_feature,
            color="Disease_Risk"
            if "Disease_Risk" in data.columns
            else None,
            title=f"{selected_feature} Distribution"
        )

        st.plotly_chart(
            fig_feature,
            use_container_width=True
        )

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        st.subheader("📊 Statistical Summary")

        st.dataframe(
            data.describe(),
            use_container_width=True
        )

    else:

        st.error(
            "❌ Dataset not found at:"
        )

        st.code(
            str(dataset_path)
        )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Comparison of machine learning models evaluated "
        "during the training process."
    )

    st.divider()

    # -----------------------------------------------------
    # MODEL RESULTS
    # -----------------------------------------------------

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Tuned Random Forest",
            "Tuned XGBoost"
        ],
        "Accuracy": [
            0.8975,
            0.8725,
            0.9525,
            0.9875,
            0.9900,
            0.9450,
            0.9950
        ],
        "Precision": [
            0.925676,
            0.885350,
            0.947712,
            0.986486,
            0.989831,
            0.935691,
            0.996587
        ],
        "Recall": [
            0.935154,
            0.948805,
            0.989761,
            0.996587,
            0.996587,
            0.993174,
            0.996587
        ],
        "F1 Score": [
            0.930390,
            0.915980,
            0.968280,
            0.991511,
            0.993197,
            0.963576,
            0.996587
        ],
        "ROC-AUC": [
            0.959300,
            0.900609,
            0.994546,
            0.999745,
            0.999777,
            0.995790,
            0.999968
        ]
    })

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    best_model = results.loc[
        results["Accuracy"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🏆 Best Model",
            best_model["Model"]
        )

    with col2:

        st.metric(
            "Accuracy",
            f"{best_model['Accuracy'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "F1 Score",
            f"{best_model['F1 Score'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "ROC-AUC",
            f"{best_model['ROC-AUC'] * 100:.3f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON TABLE
    # -----------------------------------------------------

        # -----------------------------------------------------
    # ACCURACY CHART
    # -----------------------------------------------------

    st.subheader("🎯 Accuracy Comparison")

    accuracy_chart = results.copy()

    accuracy_chart["Accuracy"] = (
        accuracy_chart["Accuracy"] * 100
    )

    fig_accuracy = px.bar(
        accuracy_chart,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Machine Learning Model Accuracy"
    )

    fig_accuracy.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_accuracy.update_layout(
        xaxis_title="Machine Learning Model",
        yaxis_title="Accuracy (%)",
        yaxis_range=[80, 102],
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_accuracy,
        use_container_width=True
    )

    st.caption(
        "Accuracy values are shown as percentages."
    )
    st.divider()

    # -----------------------------------------------------
    # F1 SCORE CHART
    # -----------------------------------------------------

    st.subheader("📈 F1 Score Comparison")

    f1_chart = results.copy()

    f1_chart["F1 Score"] = (
        f1_chart["F1 Score"] * 100
    )

    st.bar_chart(
        f1_chart.set_index("Model")["F1 Score"]
    )

    st.divider()

    # -----------------------------------------------------
    # BEST MODEL SUMMARY
    # -----------------------------------------------------

    st.subheader("🏆 Selected Model")

    st.success(
        f"**{best_model['Model']}** achieved the highest "
        f"accuracy of **{best_model['Accuracy'] * 100:.2f}%** "
        f"among the evaluated models."
    )

    st.info(
        "The model used by the current prediction system "
        "should match the model saved in the models folder."
    )
elif page == "🤖 Model Information":

    st.title("🤖 Machine Learning Model")

    st.write(
        "Information about the trained disease-risk "
        "classification model."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Model")

        st.markdown(
            """
            **Algorithm:** Tuned XGBoost Classifier

            **Task:** Binary Classification

            **Output:**

            - Low Risk
            - High Risk

            **Prediction Type:** Classification
            """
        )

    with col2:

        st.subheader("Prediction Pipeline")

        st.markdown(
            """
            ```text
            Patient Input
                  ↓
            Data Preprocessing
                  ↓
            Categorical Encoding
                  ↓
            Feature Alignment
                  ↓
            Feature Scaling
                  ↓
            Tuned XGBoost
                  ↓
            Risk Prediction
            ```
            """
        )

    st.divider()

    st.subheader("📌 Model Features")

    feature_df = pd.DataFrame(
        {"Feature Name": feature_names}
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## Healthcare Disease Risk Prediction

        This project demonstrates an end-to-end machine
        learning workflow for healthcare risk prediction.

        ### 🛠 Technology Stack

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit

        ### 🔄 Workflow

        Dataset → Data Analysis → Feature Engineering →
        Model Training → Model Evaluation →
        Prediction System → Streamlit Application

        ### 🎯 Project Goal

        The goal is to demonstrate how machine learning
        can be integrated into an interactive application
        for educational healthcare analytics.

        **Important:** This application is not intended
        to replace professional medical advice or diagnosis.
        """
    )

    st.divider()

    st.success(
        "Healthcare AI Risk Prediction System is running successfully."
    )