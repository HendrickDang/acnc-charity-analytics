# ACNC Charity Analytics

**Predicting Australian charity size and governance characteristics from the ACNC Charity Register.**

A complete data analytics pipeline built for PRT564 Data Analytics and Visualisation at Charles Darwin University, progressing from exploratory analysis through regression and classification modelling. The project uses publicly available data from the Australian Charities and Not-for-profits Commission (ACNC) to help understand the composition of Australia's charity sector and support evidence-based regulation.

---

## Project Overview

The Australian charity sector comprises more than 65,000 registered organisations regulated by the ACNC under a single national framework. This project applies data analytics to that sector, asking two related questions across its stages: what factors predict a charity's governance capacity (the number of responsible persons on its board), and can a charity's size category be predicted from its operational and financial characteristics?

The work is framed for the ACNC as the national regulator, whose interest lies in understanding sector composition and tailoring compliance and support activities — particularly for under-resourced jurisdictions such as the Northern Territory and for Indigenous-led organisations.

The repository documents one continuous analytics workflow that evolved across the unit's assessments:

**Preprocessing → EDA → Regression Modelling → Classification Modelling → Evaluation → Reporting**

---

## Datasets

Both datasets are publicly available from [data.gov.au](https://data.gov.au) and are **not committed to this repository** due to size; they can be downloaded directly from the source.

| Dataset | Records | Variables | Role |
|---------|---------|-----------|------|
| ACNC Charity Register | 65,114 | 69 | Primary dataset — charitable purposes, beneficiary groups, state operations, governance structure, charity size |
| ACNC 2023 Annual Information Statement (AIS) | 53,323 | 91 | Supplementary dataset — financial variables (revenue, expenses, assets) and staffing counts |

**Integration:** The two datasets are joined on Australian Business Number (ABN) using a **left join** (77.9% match rate), preserving all register records — including smaller and exempt charities — rather than biasing the sample toward larger organisations that lodge AIS returns.

---

## Pipeline

```
Raw ACNC Register (65,114 records)
        │
        ├─ AIS 2023 join on ABN (left join, 77.9% match)
        │
        ├─ Preprocessing
        │     • State standardisation → 58,235 valid records
        │     • Binary encoding (47 Y/blank columns → 1/0)
        │     • Missing financial values handled
        │     • Outlier and zero-target removal
        │
        ├─ Feature Engineering
        │     • Num_States_Operated, Num_Purposes, Num_Beneficiaries
        │     • Is_NT, Size_Encoded
        │     • Revenue_per_Staff, Total_Staff, Has_AIS, Rev_Exp_Ratio
        │
        ├─ Exploratory Data Analysis
        │     • Class distributions, correlations, outlier analysis
        │
        ├─ Regression Modelling (Assessment 2)
        │     • OLS, Ridge, Lasso — predicting responsible persons
        │
        ├─ Classification Modelling (Assessment 4)
        │     • Naive Bayes, Decision Tree, Random Forest — predicting charity size
        │
        └─ Evaluation & Reporting
              • Cross-validation, statistical tests, ROC-AUC, recommendations
```

---

## Repository Structure

```
acnc-charity-analytics/
├── README.md                       # This file
├── CONTRIBUTORS.md                 # Team roles and contributions
├── .gitignore
│
├── assignment1/                    # Exploratory Data Analysis
│   └── notebooks/
│       └── EDA_initial_analysis.ipynb
│
├── assignment2/                    # Regression Analysis
│   ├── regression_analysis.ipynb   # OLS / Ridge / Lasso pipeline
│   └── Preprocessing/              # Modular preprocessing scripts
│
├── assignment3/                    # Project planning
│
├── Project_Report/                 # Classification analysis (Assessment 4)
│                                   # Naive Bayes / Decision Tree / Random Forest
│
└── assets/
    └── charts/                     # Generated EDA and evaluation figures
```

---

## How the Workflow Evolved Across Assessments

**Assessment 1 — Project Plan & EDA.** Selected the ACNC Charity Register, defined the organisational context and stakeholders, framed research questions, and conducted initial exploratory analysis of sector composition (charity sizes, purposes, beneficiary groups, geographic spread, with a focus on the Northern Territory).

**Assessment 2 — Regression.** Built a regression pipeline to predict the number of responsible persons on a charity's board. Integrated the AIS dataset, engineered features, and compared OLS, Ridge, and Lasso models with cross-validation, statistical tests, and residual diagnostics. Key finding: Northern Territory charities have significantly larger governance boards, reflecting Aboriginal-controlled cultural governance structures.

**Assessment 4 — Classification.** Reframed the problem as predicting charity size (Small, Medium, Large). Extended preprocessing and feature engineering for classification, addressed severe class imbalance with balanced class weighting, and built and compared three classifiers.

---

## Key Results

### Regression (Assessment 2)
- Three models compared (OLS, Ridge, Lasso); OLS and Ridge statistically indistinguishable
- NT charities significantly larger governance (5.73 vs 5.11 responsible persons, p < 0.001)
- Charity size the dominant predictor of board size

### Classification (Assessment 4)

| Model | Accuracy | Macro F1 | ROC-AUC | 10-Fold CV |
|-------|----------|----------|---------|------------|
| Gaussian Naive Bayes | 49.08% | — | 0.760 | 41.49% |
| Decision Tree | 86.85% | 0.793 | 0.861 | 87.13% |
| **Random Forest** | **91.47%** | **0.863** | **0.966** | **91.29%** |

**Random Forest** is the recommended model — strongest across every metric, with stable cross-validation performance and balanced accuracy across all three size classes.

---

## Technologies

- **Python 3.9+**
- **pandas, numpy** — data manipulation
- **scikit-learn** — regression and classification models, cross-validation, metrics
- **scipy, statsmodels** — statistical testing and diagnostics
- **matplotlib, seaborn** — visualisation

### Running the Analysis

Download the datasets from [data.gov.au](https://data.gov.au) into a local `data/` folder, then open the relevant notebook:

```bash
pip install pandas numpy scikit-learn scipy statsmodels matplotlib seaborn

# Regression analysis
jupyter notebook assignment2/regression_analysis.ipynb

# Classification analysis
# (see Project_Report/)
```

---

## Team

Group 2 — PRT564 Data Analytics and Visualisation, Charles Darwin University, Semester 1 2026.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for individual roles and contributions.

---

## Data Source and Attribution

Australian Charities and Not-for-profits Commission (ACNC). *Charity Register* and *2023 Annual Information Statement*. data.gov.au. Used under the Creative Commons licences specified on data.gov.au.

This repository is submitted as academic coursework and is provided for assessment purposes.
