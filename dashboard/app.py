import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "ai_job_market_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "salary_prediction_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "salary_preprocessor.pkl"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:
    df = load_data()
    model = load_model()
    preprocessor = load_preprocessor()

except Exception as e:
    st.error("❌ Unable to load project files.")
    st.error(str(e))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("💼 AI Job Market Intelligence & Salary Prediction System")

st.write(
    "An interactive dashboard for analyzing AI job demand, "
    "salary trends, skills, countries and salary predictions."
)

st.success(
    "Dataset, model and preprocessing pipeline loaded successfully!"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Dashboard Filters")


# ============================================================
# ABOUT PROJECT
# ============================================================

with st.sidebar.expander("ℹ️ About This Project"):

    st.markdown(
        """
        ### AI Job Market Intelligence

        This project analyzes job-market data to identify:

        - 📈 Job demand and market trends
        - 💰 Salary patterns
        - 🧠 In-demand technical skills
        - 🌍 Geographic job distribution
        - 🤖 AI-related job demand
        - 🔮 Salary predictions using Machine Learning

        **Dataset**

        - 1,696 job postings
        - 3 countries
        - 23 processed features
        - 1,513 jobs with salary information

        **Machine Learning**

        The salary prediction system uses a
        **Log-target Random Forest Regressor**.

        **Technologies**

        Python • Pandas • NumPy • Scikit-learn •
        Plotly • Streamlit • Joblib

        **Model R²:** 0.6926
        """
    )


# ============================================================
# COUNTRY FILTER
# ============================================================

countries = sorted(
    df["country_name"]
    .dropna()
    .unique()
    .tolist()
)

selected_country = st.sidebar.multiselect(
    "Select Country",
    countries,
    default=countries
)


# ============================================================
# JOB TYPE FILTER
# ============================================================

selected_ai = st.sidebar.selectbox(
    "Job Type",
    [
        "All",
        "AI Related",
        "Non-AI Related"
    ]
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_country:

    filtered_df = filtered_df[
        filtered_df["country_name"].isin(selected_country)
    ]


if selected_ai == "AI Related":

    filtered_df = filtered_df[
        filtered_df["is_ai_related"] == True
    ]


elif selected_ai == "Non-AI Related":

    filtered_df = filtered_df[
        filtered_df["is_ai_related"] == False
    ]


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.subheader("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Jobs",
        f"{len(filtered_df):,}"
    )


with col2:

    ai_jobs = int(
        filtered_df["is_ai_related"].sum()
    )

    st.metric(
        "AI Related Jobs",
        f"{ai_jobs:,}"
    )


with col3:

    salary_count = int(
        filtered_df["salary_usd"].notna().sum()
    )

    st.metric(
        "Jobs With Salary",
        f"{salary_count:,}"
    )


with col4:

    median_salary = filtered_df["salary_usd"].median()

    if pd.notna(median_salary):

        st.metric(
            "Median Salary",
            f"${median_salary:,.0f}"
        )

    else:

        st.metric(
            "Median Salary",
            "N/A"
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📋 Dataset Information")

info1, info2, info3 = st.columns(3)


with info1:

    st.write(
        "**Rows:**",
        f"{len(filtered_df):,}"
    )


with info2:

    st.write(
        "**Columns:**",
        len(filtered_df.columns)
    )


with info3:

    st.write(
        "**Countries:**",
        filtered_df["country_name"].nunique()
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Model Performance")

st.caption(
    "Evaluation results from the held-out test set used during model development."
)

model_col1, model_col2, model_col3, model_col4 = st.columns(4)


with model_col1:

    st.metric(
        "Model",
        "Log-target Random Forest"
    )


with model_col2:

    st.metric(
        "MAE",
        "$16,446.95"
    )


with model_col3:

    st.metric(
        "RMSE",
        "$29,365.88"
    )


with model_col4:

    st.metric(
        "R²",
        "0.6926"
    )


st.info(
    "MAE represents the average absolute prediction error. "
    "RMSE gives greater weight to larger errors. "
    "R² indicates the proportion of variance explained by the model "
    "on the test set."
)


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

st.download_button(
    label="⬇️ Download Filtered Job Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="ai_job_market_filtered.csv",
    mime="text/csv"
)


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Job Market",
        "💰 Salary Analysis",
        "🧠 Skill Demand",
        "🌍 Country Analysis",
        "🤖 Salary Prediction"
    ]
)


# ============================================================
# TAB 1 — JOB MARKET
# ============================================================

with tab1:

    st.header("📈 Job Market Analysis")


    # --------------------------------------------------------
    # TOP JOB CATEGORIES
    # --------------------------------------------------------

    category_counts = (
        filtered_df["category"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Job_Count"
    ]


    fig_category = px.bar(
        category_counts,
        x="Job_Count",
        y="Category",
        orientation="h",
        title="Top 10 Job Categories"
    )


    fig_category.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )


    st.plotly_chart(
        fig_category,
        width="stretch"
    )


    # --------------------------------------------------------
    # JOB POSTINGS BY YEAR
    # --------------------------------------------------------

    year_data = (
        pd.to_datetime(
            filtered_df["created_date"],
            errors="coerce"
        )
        .dt.year
    )


    jobs_by_year = (
        year_data
        .value_counts()
        .sort_index()
        .reset_index()
    )


    jobs_by_year.columns = [
        "Year",
        "Job_Count"
    ]


    fig_year = px.line(
        jobs_by_year,
        x="Year",
        y="Job_Count",
        markers=True,
        title="Job Postings by Year"
    )


    st.plotly_chart(
        fig_year,
        width="stretch"
    )


    st.info(
        "Note: this dataset is heavily concentrated in 2026, "
        "so the yearly chart should not be interpreted as a "
        "complete long-term market-growth trend."
    )


# ============================================================
# TAB 2 — SALARY ANALYSIS
# ============================================================

with tab2:

    st.header("💰 Salary Analysis")


    salary_df = filtered_df.dropna(
        subset=["salary_usd"]
    ).copy()


    if len(salary_df) > 0:


        # ----------------------------------------------------
        # SALARY DISTRIBUTION
        # ----------------------------------------------------

        fig_salary = px.histogram(
            salary_df,
            x="salary_usd",
            nbins=50,
            title="Salary Distribution"
        )


        fig_salary.update_xaxes(
            title="Salary (USD)"
        )


        fig_salary.update_yaxes(
            title="Number of Jobs"
        )


        st.plotly_chart(
            fig_salary,
            width="stretch"
        )


        # ----------------------------------------------------
        # AI VS NON-AI SALARY
        # ----------------------------------------------------

        ai_salary = (
            salary_df
            .groupby("is_ai_related")["salary_usd"]
            .median()
            .reset_index()
        )


        ai_salary["Job_Type"] = (
            ai_salary["is_ai_related"]
            .map({
                True: "AI Related",
                False: "Non-AI Related"
            })
        )


        fig_ai_salary = px.bar(
            ai_salary,
            x="Job_Type",
            y="salary_usd",
            title="Median Salary: AI vs Non-AI Jobs",
            labels={
                "salary_usd": "Median Salary (USD)",
                "Job_Type": "Job Type"
            }
        )


        st.plotly_chart(
            fig_ai_salary,
            width="stretch"
        )


        # ----------------------------------------------------
        # SALARY BY COUNTRY
        # ----------------------------------------------------

        country_salary = (
            salary_df
            .groupby("country_name")["salary_usd"]
            .median()
            .reset_index()
            .sort_values(
                "salary_usd",
                ascending=False
            )
        )


        fig_country_salary = px.bar(
            country_salary,
            x="country_name",
            y="salary_usd",
            title="Median Salary by Country",
            labels={
                "country_name": "Country",
                "salary_usd": "Median Salary (USD)"
            }
        )


        st.plotly_chart(
            fig_country_salary,
            width="stretch"
        )


        # ----------------------------------------------------
        # SALARY STATISTICS
        # ----------------------------------------------------

        salary_summary = (
            salary_df["salary_usd"]
            .describe()
            .round(2)
            .to_frame("Value")
        )


        st.subheader("Salary Statistics")


        st.dataframe(
            salary_summary,
            width="stretch"
        )


    else:

        st.warning(
            "No salary data available for the selected filters."
        )


# ============================================================
# TAB 3 — SKILL DEMAND
# ============================================================

with tab3:

    st.header("🧠 Skill Demand Analysis")


    skills = [
        "Python",
        "SQL",
        "AWS",
        "Azure",
        "GCP",
        "TensorFlow",
        "PyTorch",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "Artificial Intelligence",
        "Data Science",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Docker"
    ]


    skill_counts = {}


    for skill in skills:

        skill_counts[skill] = (
            filtered_df["description"]
            .str.contains(
                skill,
                case=False,
                na=False
            )
            .sum()
        )


    skill_df = pd.DataFrame(
        list(skill_counts.items()),
        columns=[
            "Skill",
            "Job_Count"
        ]
    )


    skill_df = skill_df.sort_values(
        "Job_Count",
        ascending=False
    )


    fig_skills = px.bar(
        skill_df,
        x="Job_Count",
        y="Skill",
        orientation="h",
        title="AI Job Skill Demand"
    )


    fig_skills.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )


    st.plotly_chart(
        fig_skills,
        width="stretch"
    )


    st.subheader("Skill Demand Table")


    st.dataframe(
        skill_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TAB 4 — COUNTRY ANALYSIS
# ============================================================

with tab4:

    st.header("🌍 Country Analysis")


    # --------------------------------------------------------
    # JOB POSTINGS BY COUNTRY
    # --------------------------------------------------------

    country_counts = (
        filtered_df["country_name"]
        .value_counts()
        .reset_index()
    )


    country_counts.columns = [
        "Country",
        "Job_Count"
    ]


    fig_countries = px.bar(
        country_counts,
        x="Country",
        y="Job_Count",
        title="Job Postings by Country"
    )


    st.plotly_chart(
        fig_countries,
        width="stretch"
    )


    # --------------------------------------------------------
    # AI JOBS BY COUNTRY
    # --------------------------------------------------------

    country_ai = (
        filtered_df
        .groupby("country_name")["is_ai_related"]
        .sum()
        .reset_index()
    )


    country_ai.columns = [
        "Country",
        "AI_Jobs"
    ]


    fig_country_ai = px.bar(
        country_ai,
        x="Country",
        y="AI_Jobs",
        title="AI-Related Jobs by Country"
    )


    st.plotly_chart(
        fig_country_ai,
        width="stretch"
    )


    # --------------------------------------------------------
    # COUNTRY TABLE
    # --------------------------------------------------------

    st.subheader("Country Job Summary")


    country_summary = (
        filtered_df
        .groupby("country_name")
        .agg(
            Total_Jobs=("id", "count"),
            AI_Jobs=("is_ai_related", "sum"),
            Jobs_With_Salary=("salary_usd", "count")
        )
        .reset_index()
    )


    st.dataframe(
        country_summary,
        width="stretch",
        hide_index=True
    )


# ============================================================
# TAB 5 — SALARY PREDICTION
# ============================================================

with tab5:

    st.header("🤖 AI Salary Prediction")

    st.write(
        "Enter job details below to estimate the salary in USD."
    )

    pred_col1, pred_col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with pred_col1:

        # ----------------------------------------------------
        # JOB TITLE
        # ----------------------------------------------------

        title_options = sorted(
            df["title"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_title = st.selectbox(
            "Job Title",
            title_options
        )

        # ----------------------------------------------------
        # AUTOMATIC JOB CATEGORY
        # ----------------------------------------------------

        matching_categories = (
            df.loc[
                df["title"] == selected_title,
                "category"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if matching_categories:

            selected_category = matching_categories[0]

        else:

            selected_category = (
                df["category"].mode()[0]
            )

        st.text_input(
            "Job Category",
            value=selected_category,
            disabled=True
        )

        # ----------------------------------------------------
        # CONTRACT TYPE
        # ----------------------------------------------------

        contract_type_options = sorted(
            df["contract_type"]
            .fillna("Unknown")
            .unique()
            .tolist()
        )

        selected_contract_type = st.selectbox(
            "Contract Type",
            contract_type_options
        )

        # ----------------------------------------------------
        # CONTRACT TIME
        # ----------------------------------------------------

        contract_time_options = sorted(
            df["contract_time"]
            .fillna("Unknown")
            .unique()
            .tolist()
        )

        selected_contract_time = st.selectbox(
            "Contract Time",
            contract_time_options
        )

    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with pred_col2:

        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        country_options = sorted(
            df["country_name"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_prediction_country = st.selectbox(
            "Country",
            country_options
        )

        # ----------------------------------------------------
        # AI RELATED
        # ----------------------------------------------------

        selected_ai_related = st.checkbox(
            "AI Related Job",
            value=True
        )

        # ----------------------------------------------------
        # SALARY PREDICTED
        # ----------------------------------------------------

        selected_salary_predicted = st.checkbox(
            "Salary is Predicted",
            value=False
        )

    # --------------------------------------------------------
    # SEPARATOR
    # --------------------------------------------------------

    st.divider()

    # --------------------------------------------------------
    # PREDICT SALARY BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Salary",
        width="stretch"
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        new_job = pd.DataFrame({

            "title": [
                selected_title
            ],

            "category": [
                selected_category
            ],

            "contract_type": [
                selected_contract_type
            ],

            "contract_time": [
                selected_contract_time
            ],

            "country_name": [
                selected_prediction_country
            ],

            "is_ai_related": [
                selected_ai_related
            ],

            "salary_is_predicted": [
                int(selected_salary_predicted)
            ]
        })

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            # Apply the saved preprocessing pipeline
            new_job_processed = (
                preprocessor.transform(new_job)
            )

            # Predict log salary
            predicted_log_salary = (
                model.predict(new_job_processed)
            )

            # Convert log salary back to USD
            predicted_salary = (
                np.expm1(predicted_log_salary)[0]
            )

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.success(
                "Salary prediction generated successfully!"
            )

            st.metric(
                "Estimated Salary",
                f"${predicted_salary:,.2f}"
            )

            st.caption(
                "The prediction is an estimate based on the "
                "trained log-target Random Forest model and "
                "the selected job attributes."
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.error(str(e))