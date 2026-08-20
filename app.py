import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "bmi"
]

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical Activity",
    "bmi": "BMI",
}

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 2.5rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at top right,
        rgba(255,70,90,.20), transparent 35%),
        linear-gradient(135deg,
        rgba(170,25,45,.18),
        rgba(20,20,30,.05));
    border: 1px solid rgba(220,80,100,.25);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: .4rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    opacity: .80;
    max-width: 850px;
}

.card {
    padding: 1.4rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    margin-bottom: 1rem;
}

.status-box {
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,.22);
}

div.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}

.small {
    font-size: .88rem;
    opacity: .70;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("cardio_xgboost_model.pkl")


try:
    model = load_model()
except Exception as exc:
    st.error("Unable to load cardio_xgboost_model.pkl")
    st.exception(exc)
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_patient(
    age, gender, height_cm, weight,
    systolic, diastolic, cholesterol,
    glucose, smoke, alcohol, active
):

    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    return pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height_m,
        "weight": weight,
        "ap_hi": systolic,
        "ap_lo": diastolic,
        "cholesterol": cholesterol,
        "gluc": glucose,
        "smoke": smoke,
        "alco": alcohol,
        "active": active,
        "bmi": bmi,
    }])[FEATURES]


def validate_inputs(patient):

    warnings = []

    systolic = patient["ap_hi"].iloc[0]
    diastolic = patient["ap_lo"].iloc[0]
    bmi = patient["bmi"].iloc[0]

    if systolic <= diastolic:
        warnings.append(
            "Systolic blood pressure is not greater than "
            "diastolic blood pressure."
        )

    if bmi < 10 or bmi > 70:
        warnings.append(
            "The calculated BMI is unusually extreme. "
            "Please verify height and weight."
        )

    return warnings


def probability_gauge(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 46}},
            title={
                "text": "Model Probability",
                "font": {"size": 20}
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.30},
                "steps": [
                    {"range": [0, 50]},
                    {"range": [50, 100]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": .8,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=70, b=20),
    )

    return fig


def get_feature_contributions(patient):

    try:
        booster = model.get_booster()

        dmatrix = __import__("xgboost").DMatrix(
            patient,
            feature_names=FEATURES
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True
        )[0]

        values = contributions[:-1]

        result = pd.DataFrame({
            "Feature": [
                FEATURE_LABELS[x] for x in FEATURES
            ],
            "Contribution": values,
        })

        result["Absolute Impact"] = (
            result["Contribution"].abs()
        )

        return result.sort_values(
            "Absolute Impact",
            ascending=False
        )

    except Exception:
        return None


def create_report(patient, probability):

    predicted_class = (
        "Cardiovascular disease"
        if probability >= 0.50
        else "No cardiovascular disease"
    )

    report = f"""
CARDIORISK AI
MODEL ASSESSMENT SUMMARY
============================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M")}

MODEL
--------------------------------------------
Algorithm: XGBoost
Test ROC-AUC: 0.8022
Test Accuracy: 73.45%
Classification threshold: 0.50

MODEL OUTPUT
--------------------------------------------
Estimated probability:
{probability * 100:.2f}%

Predicted class:
{predicted_class}

PATIENT INPUT
--------------------------------------------
Age: {patient['age'].iloc[0]:.0f} years
Gender code: {patient['gender'].iloc[0]}
Height: {patient['height'].iloc[0]:.2f} m
Weight: {patient['weight'].iloc[0]:.1f} kg
BMI: {patient['bmi'].iloc[0]:.2f}
Systolic BP: {patient['ap_hi'].iloc[0]:.0f}
Diastolic BP: {patient['ap_lo'].iloc[0]:.0f}
Cholesterol code: {patient['cholesterol'].iloc[0]}
Glucose code: {patient['gluc'].iloc[0]}
Smoking: {patient['smoke'].iloc[0]}
Alcohol: {patient['alco'].iloc[0]}
Physical activity: {patient['active'].iloc[0]}

IMPORTANT
--------------------------------------------
This output is generated by an educational
machine-learning project.

It is not a medical diagnosis, clinical risk
score, or recommendation for treatment.

The model has not undergone prospective
clinical validation.
"""

    return report

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("❤️ CardioRisk AI")

    st.caption("Machine Learning Portfolio Project")

    st.divider()

    st.subheader("Model Performance")

    st.metric("ROC-AUC", "0.8022")
    st.metric("Accuracy", "73.45%")
    st.metric("Precision", "75.51%")
    st.metric("Recall", "68.60%")

    st.divider()

    st.write("**Model:** XGBoost")
    st.write("**Features:** 12")
    st.write("**Threshold:** 0.50")

    st.divider()

    st.warning(
        "Educational demonstration only. "
        "Not a clinical diagnostic system."
    )

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        ❤️ CardioRisk AI
    </div>

    <div class="hero-subtitle">
        Interactive cardiovascular disease prediction
        powered by XGBoost. Explore model predictions,
        patient-level feature contributions, model
        performance and explainability.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "🩺 Risk Assessment",
    "🧠 Model Insights",
    "ℹ️ About",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader("Machine Learning Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Accuracy", "73.45%")
    c2.metric("ROC-AUC", "0.8022")
    c3.metric("Training Records", "55,003")
    c4.metric("Test Records", "13,751")

    st.divider()

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("### What this project does")

        st.write("""
        CardioRisk AI demonstrates an end-to-end
        machine-learning workflow for cardiovascular
        disease classification.

        The final XGBoost model uses demographic,
        clinical and lifestyle information to generate
        a modeled probability for cardiovascular
        disease.
        """)

        st.markdown("### Project workflow")

        st.write("""
        **1. Data preparation**  
        Reproducible cleaning and feature engineering.

        **2. Model development**  
        Logistic Regression, Random Forest and XGBoost.

        **3. Validation**  
        Train/test evaluation and five-fold
        cross-validation.

        **4. Model selection**  
        XGBoost selected based on overall discrimination.

        **5. Explainability**  
        Global and patient-level model interpretation.

        **6. Deployment**  
        Saved model integrated into this Streamlit
        application.
        """)

    with right:

        st.markdown("### Final Model")

        st.info("""
        **Algorithm:** XGBoost

        **Accuracy:** 73.45%

        **Precision:** 75.51%

        **Recall:** 68.60%

        **F1 Score:** 71.89%

        **ROC-AUC:** 0.8022
        """)

        st.markdown("### Most Influential Features")

        st.write("""
        Global SHAP analysis identified:

        1. Systolic blood pressure
        2. Age
        3. Cholesterol
        4. Diastolic blood pressure
        5. BMI
        """)

# ============================================================
# TAB 2 — RISK ASSESSMENT
# ============================================================

with tab2:

    st.subheader("Patient Risk Assessment")

    st.caption(
        "Enter patient characteristics below. "
        "BMI is calculated automatically."
    )

    with st.form("patient_form"):

        st.markdown("### 👤 Demographics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            age = st.number_input(
                "Age (years)",
                18, 100, 50
            )

        with c2:
            gender = st.selectbox(
                "Gender",
                [1, 2],
                format_func=lambda x:
                "Female" if x == 1 else "Male"
            )

        with c3:
            height_cm = st.number_input(
                "Height (cm)",
                100.0, 250.0, 170.0
            )

        with c4:
            weight = st.number_input(
                "Weight (kg)",
                20.0, 200.0, 70.0
            )

        st.divider()

        st.markdown("### 🩺 Clinical Measurements")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            systolic = st.number_input(
                "Systolic BP",
                30, 300, 120
            )

        with c2:
            diastolic = st.number_input(
                "Diastolic BP",
                30, 300, 80
            )

        with c3:
            cholesterol = st.selectbox(
                "Cholesterol",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal",
                }[x]
            )

        with c4:
            glucose = st.selectbox(
                "Glucose",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal",
                }[x]
            )

        st.divider()

        st.markdown("### 🏃 Lifestyle")

        c1, c2, c3 = st.columns(3)

        with c1:
            smoke = st.selectbox(
                "Smoking",
                [0, 1],
                format_func=lambda x:
                "No" if x == 0 else "Yes"
            )

        with c2:
            alcohol = st.selectbox(
                "Alcohol Consumption",
                [0, 1],
                format_func=lambda x:
                "No" if x == 0 else "Yes"
            )

        with c3:
            active = st.selectbox(
                "Physically Active",
                [0, 1],
                format_func=lambda x:
                "No" if x == 0 else "Yes"
            )

        st.write("")

        submitted = st.form_submit_button(
            "🔍 Run Risk Analysis",
            use_container_width=True
        )

    if submitted:

        patient = build_patient(
            age,
            gender,
            height_cm,
            weight,
            systolic,
            diastolic,
            cholesterol,
            glucose,
            smoke,
            alcohol,
            active,
        )

        probability = float(
            model.predict_proba(patient)[0, 1]
        )

        prediction = int(probability >= 0.50)

        bmi = patient["bmi"].iloc[0]

        warnings = validate_inputs(patient)

        st.divider()

        if warnings:
            for warning in warnings:
                st.warning(warning)

        st.markdown("## Assessment Results")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Model Probability",
            f"{probability * 100:.1f}%"
        )

        m2.metric(
            "Classification",
            "Positive" if prediction else "Negative"
        )

        m3.metric(
            "Calculated BMI",
            f"{bmi:.1f}"
        )

        m4.metric(
            "Decision Threshold",
            "50%"
        )

        left, right = st.columns([1, 1])

        with left:

            st.plotly_chart(
                probability_gauge(probability),
                use_container_width=True
            )

        with right:

            st.markdown("### Model Classification")

            if prediction:

                st.error("""
                **Positive model classification**

                The estimated probability is above
                the model's 0.50 classification
                threshold.
                """)

            else:

                st.success("""
                **Negative model classification**

                The estimated probability is below
                the model's 0.50 classification
                threshold.
                """)

            st.write(
                f"Model probability: "
                f"**{probability * 100:.2f}%**"
            )

            st.write(
                "Classification threshold: **50%**"
            )

        # ----------------------------------------------------
        # LOCAL EXPLANATION
        # ----------------------------------------------------

        st.divider()

        st.markdown("## 🧠 Patient-Level Model Explanation")

        contributions = get_feature_contributions(patient)

        if contributions is not None:

            top = contributions.head(8).copy()

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=top["Contribution"],
                    y=top["Feature"],
                    orientation="h",
                    text=[
                        f"{x:+.3f}"
                        for x in top["Contribution"]
                    ],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Largest Feature Contributions",
                xaxis_title="Contribution to model output",
                yaxis_title="",
                height=450,
                yaxis={
                    "categoryorder": "total ascending"
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Positive values push the model toward "
                "the cardiovascular-disease class. "
                "Negative values push it away from that "
                "class. These values explain model "
                "behavior and are not causal effects."
            )

            display_table = top[
                ["Feature", "Contribution"]
            ].copy()

            display_table["Direction"] = np.where(
                display_table["Contribution"] > 0,
                "Toward positive class",
                "Away from positive class"
            )

            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True
            )

        else:
            st.info(
                "Patient-level feature contributions "
                "could not be generated."
            )

        # ----------------------------------------------------
        # INPUT DETAILS
        # ----------------------------------------------------

        with st.expander("📋 View exact model inputs"):

            display_patient = patient.rename(
                columns=FEATURE_LABELS
            )

            st.dataframe(
                display_patient,
                hide_index=True,
                use_container_width=True
            )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        report = create_report(
            patient,
            probability
        )

        st.download_button(
            label="⬇️ Download Assessment Summary",
            data=report,
            file_name="cardiorisk_model_assessment.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.warning("""
        This output is a machine-learning model
        prediction, not a clinically validated
        cardiovascular risk score.

        It must not be used for diagnosis, treatment,
        or medical decision-making.
        """)

# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================

with tab3:

    st.subheader("Model Performance & Explainability")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", "73.45%")
    c2.metric("Precision", "75.51%")
    c3.metric("Recall", "68.60%")
    c4.metric("F1", "71.89%")
    c5.metric("ROC-AUC", "0.8022")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### Model Selection")

        st.write("""
        Three primary algorithms were evaluated:

        - Logistic Regression
        - Random Forest
        - XGBoost

        XGBoost produced the strongest overall
        discrimination and was selected as the final
        classifier.
        """)

        st.markdown("### Cross-Validation")

        st.write("""
        Five-fold cross-validation was used to assess
        performance beyond a single train/test split.

        Randomized hyperparameter optimization was also
        evaluated. The tuned configuration did not
        materially outperform the baseline XGBoost
        model, so the simpler configuration was
        retained.
        """)

    with right:

        st.markdown("### Global Explainability")

        importance = pd.DataFrame({
            "Feature": [
                "Systolic BP",
                "Age",
                "Cholesterol",
                "Diastolic BP",
                "BMI",
            ],
            "Rank": [5, 4, 3, 2, 1]
        })

        fig = go.Figure(
            go.Bar(
                x=importance["Rank"],
                y=importance["Feature"],
                orientation="h"
            )
        )

        fig.update_layout(
            title="Top Features from SHAP Analysis",
            xaxis_title="Relative importance rank",
            yaxis_title="",
            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "This visualization communicates the "
            "ranking observed in the notebook's SHAP "
            "analysis rather than inventing numerical "
            "SHAP magnitudes."
        )

    st.divider()

    st.markdown("### Threshold Interpretation")

    st.write("""
    The application uses the model's default
    classification threshold of **0.50**.

    Lower thresholds generally increase recall but
    generate more false positives. Higher thresholds
    generally increase precision while producing more
    false negatives.

    No claim is made that 0.50 is an optimal clinical
    threshold.
    """)

# ============================================================
# TAB 4 — ABOUT
# ============================================================

with tab4:

    st.subheader("About CardioRisk AI")

    st.write("""
    CardioRisk AI is an educational machine-learning
    portfolio project demonstrating the complete
    development lifecycle of a binary classification
    model.
    """)

    st.markdown("### Technology Stack")

    st.code("""
Python
pandas
NumPy
scikit-learn
XGBoost
SHAP
Plotly
Streamlit
Joblib
Git / GitHub
""")

    st.markdown("### Dataset")

    st.write("""
    The source cardiovascular dataset contains
    approximately 70,000 observations.

    After reproducible data-quality filtering, 68,754
    observations were retained.

    The final split contained:

    - 55,003 training observations
    - 13,751 test observations
    """)

    st.markdown("### Limitations")

    st.write("""
    - Single observational dataset
    - No external validation cohort
    - No prospective clinical validation
    - Rule-based data-quality filtering
    - Potential demographic and measurement bias
    - Model associations do not establish causality
    - Threshold not optimized for clinical use
    - Not approved as a medical device
    """)

    st.markdown("### Responsible Use")

    st.error("""
    This application is an educational demonstration.

    The probability shown by the application should
    not be interpreted as an individual's clinically
    validated probability of cardiovascular disease.

    Medical concerns should be evaluated by qualified
    healthcare professionals.
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CardioRisk AI • XGBoost Cardiovascular Risk "
    "Prediction • Machine Learning Portfolio Project"
)