# 💼 AI Job Market Intelligence & Salary Prediction System

An end-to-end Data Science and Machine Learning project that analyzes job-market data to identify AI job demand, salary patterns, in-demand skills, geographic trends, and estimated salaries for different job profiles.

The project combines **Data Analysis, Data Visualization, Machine Learning, and an interactive Streamlit dashboard** into a single application.

---

## 📌 Project Overview

The AI job market is changing rapidly, with increasing demand for skills such as Machine Learning, Artificial Intelligence, Python, Cloud Computing, and Data Science.

This project analyzes job postings to answer questions such as:

- Which job categories have the highest demand?
- How many jobs are AI-related?
- Which countries have the most job postings?
- What technical skills appear most frequently?
- How do salaries differ between AI and non-AI jobs?
- How do salaries vary across countries?
- Can Machine Learning be used to estimate a job's salary?

The project also provides an interactive dashboard where users can explore the data and generate salary predictions.

---

# 🎯 Objectives

The main objectives of this project are:

1. Analyze AI and non-AI job demand.
2. Identify the most common job categories.
3. Analyze salary distributions.
4. Compare AI-related and non-AI salaries.
5. Analyze job postings by country.
6. Identify frequently requested technical skills.
7. Build a Machine Learning model for salary prediction.
8. Deploy the analysis through an interactive Streamlit dashboard.
9. Allow users to download filtered job-market data.

---

# 📊 Dataset

The project uses an AI Job Market dataset containing job postings collected across multiple countries.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total Job Postings | 1,696 |
| Countries | 3 |
| Processed Features | 23 |
| Jobs With Salary Information | 1,513 |
| AI-Related Jobs | 1,021 |
| Non-AI Jobs | 675 |

### Countries

The dataset contains job postings from:

- 🇺🇸 United States
- 🇬🇧 United Kingdom
- 🇮🇳 India

### Important Dataset Note

Salary information is not available for every job posting.

Only **1,513 of the 1,696 job postings** contain salary information.

India contains 200 job postings, but only 17 contain salary information. Therefore, salary comparisons involving India should be interpreted with caution.

---

# 🧹 Data Cleaning & Preprocessing

The dataset was processed before analysis and Machine Learning.

The preprocessing workflow included:

- Loading the raw CSV dataset
- Inspecting data types
- Checking missing values
- Checking duplicate records
- Validating salary ranges
- Handling missing company values
- Handling missing contract information
- Creating a salary-analysis dataset
- Converting currencies into USD
- Identifying unusual salary values
- Creating salary quality flags
- Converting date columns
- Creating additional analytical features

### Salary Standardization

The dataset contains salaries in:

- USD
- GBP
- INR

For cross-country salary analysis, salary values were converted into USD.

The currency conversion used fixed reference exchange rates documented during the analysis.

These converted values are intended for comparative analysis rather than historical exchange-rate reconstruction.

---

# 🔎 Exploratory Data Analysis

The project performs several types of exploratory analysis.

## Job Market Analysis

The dashboard analyzes:

- Job categories
- Job postings by year
- AI-related job demand
- Country distribution

The dataset is heavily concentrated in 2026, so the yearly distribution should not be interpreted as a complete long-term job-market growth trend.

---

## 💰 Salary Analysis

The project analyzes:

- Salary distribution
- Median salary
- Average salary
- Salary by country
- Salary by job category
- AI vs non-AI salary differences

Salary values are analyzed in USD after currency standardization.

---

## 🧠 Skill Demand Analysis

Technical skills were searched within job descriptions.

The analysis includes skills such as:

- Machine Learning
- Artificial Intelligence
- Python
- Data Science
- AWS
- Azure
- Deep Learning
- PyTorch
- SQL
- NLP
- TensorFlow
- GCP
- Docker
- Scikit-learn
- NumPy
- Pandas

The skill extraction is based on keyword matching within the available job-description text.

---

# 🤖 Machine Learning

The project includes a Machine Learning pipeline for salary prediction.

## Target Variable

The prediction target is:

```text
salary_usd