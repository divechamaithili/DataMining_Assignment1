# CMPE 255: Data Mining — Assignment 1

## Project Overview
This repository contains the complete deliverables for **CMPE 255 Assignment 1**, implementing an end-to-end clinical machine-learning pipeline for **Diabetes Risk Screening** across 100,000 patient records following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

The project addresses the asymmetric misclassification costs inherent to healthcare screening:
* **False Negatives (Missed Diabetics):** Leads to unmonitored chronic hyperglycemia, causing severe long-term microvascular (retinopathy, nephropathy) and macrovascular (cardiovascular failure, stroke) complications.
* **False Positives:** Triggers an inexpensive, non-invasive confirmatory laboratory draw (such as a fasting plasma glucose or oral glucose tolerance test).

Primary optimization prioritized **Sensitivity (Recall $\ge 90\%$)** and **Precision-Recall AUC (PR-AUC)** over raw classification accuracy on an imbalanced target (8.50% base prevalence).

---

## Directory Organization

```text
assignment-1/
├── README.md
├── part1_diabetes_prediction/
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   │   └── diabetes_prediction.ipynb
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── eda.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── figures/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_numerical_kde_by_target.png
│   │   ├── 04_categorical_target_rates.png
│   │   ├── 05_correlation_matrix.png
│   │   ├── 07_outlier_rigorous_analysis.png
│   │   ├── 08_model_comparison_roc_pr.png
│   │   ├── 09_confusion_matrices_grid.png
│   │   ├── 10_detailed_error_analysis.png
│   │   └── 11_model_interpretability_and_importance.png
│   ├── results/
│   │   ├── model_comparison_metrics.csv
│   │   └── model_metadata.json
│   └── prompts/
│       └── part1_prompts.md
└── part2_reproduction/
    └── README.md


    ## Dataset Acquisition & Preprocessing Strategy

### 1. Data Characterization
* **Source:** Healthcare Diabetes Prediction Dataset (100,000 records, 9 raw features).
* **Imbalance Profile:** 91,500 non-diabetic observations (91.50%) and 8,500 diabetic observations (8.50%) — a 10.76 : 1 class imbalance ratio.
* **Synthetic Discretization:** Exploratory inspection revealed that continuous-labeled laboratory biomarkers (`HbA1c_level` and `blood_glucose_level`) each take on exactly 18 distinct, uniformly spaced lattice values rather than natural Gaussian distributions.

### 2. Leakage Prevention Protocol
* **Partition-First Separation:** An 80/20 Stratified Train/Test split (`random_state=42`) was established before computing any summary statistics, scaling distributions, or fitting transformers.
* **Training-Only Deduplication:** 3,854 duplicate observations and 91 contradictory label combinations (identical feature profiles with differing target labels) were addressed strictly on the training partition ($N_{\text{train}} = 77,270$). The holdout test set ($N_{\text{test}} = 20,000$) was preserved untouched to ensure realistic population evaluation.

### 3. Missingness & Artifact Engineering
* **Missingness Indicator Method (MIM):** Exactly 25,495 entries (25.50%) in `bmi` were concentrated at precisely 27.32 (the global column mean/median from upstream imputation). A binary flag $\mathbb{I}_{\text{bmi\_was\_imputed}}$ was created to allow linear and tree models to isolate this point mass.
* **Informative Intake Missingness:** In `smoking_history`, 35,816 entries were recorded as `'No Info'`. Because this was heavily concentrated in pediatric visits ($<18$ years old) where diabetes prevalence is under 0.5%, missingness was preserved as an explicit category (`'unknown'`) rather than dropped or imputed.
* **Biomarker Interaction:** Engineered $\text{glucose\_x\_hba1c} = \text{blood\_glucose\_level} \times \text{HbA1c\_level}$ to capture acute-chronic glycemic synergy.
* **Comorbidity Burden:** Aggregated macrovascular risk score: $\text{comorbidity\_count} = \text{hypertension} + \text{heart\_disease}$.

---

## Benchmark Results (Holdout Test Set: $N = 20,000$, Prevalence = $8.50\%$)

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC | False Negatives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.9150 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0850 | 1,700 (100%) | 0 |
| **Stratified Baseline** | 0.8464 | 0.0955 | 0.0953 | 0.0954 | 0.5057 | 0.0864 | 1,538 (90.5%) | 1,535 |
| **KNN ($k=15$)** | 0.9617 | 0.9177 | 0.6035 | 0.7282 | 0.9433 | 0.8073 | 674 (39.7%) | 92 |
| **Linear SVM (Calibrated)** | 0.9594 | 0.8453 | 0.6394 | 0.7281 | 0.9630 | 0.8197 | 613 (36.1%) | 199 |
| **Decision Tree (Balanced)** | 0.8194 | 0.3151 | 0.9582 | 0.4742 | 0.9677 | 0.8178 | 71 (4.2%) | 3,541 |
| **Logistic Regression (Balanced)** | 0.8890 | 0.4267 | 0.8924 | 0.5774 | 0.9630 | 0.8197 | 183 (10.8%) | 2,038 |
| **Random Forest (Balanced)** | 0.9159 | 0.5030 | 0.9006 | 0.6454 | 0.9779 | 0.8813 | 169 (9.9%) | 1,513 |
| **HistGradientBoosting (Tuned)** | 0.9107 | 0.4862 | 0.9029 | 0.6321 | 0.9787 | 0.8852 | 165 (9.7%) | 1,622 |

---

## Recommended Model: Tuned `HistGradientBoostingClassifier`

* **The Accuracy Paradox:** K-Nearest Neighbors and Linear SVM produced the highest raw accuracy ($>95.9\%$), yet failed the primary clinical requirement by missing nearly $40\%$ of all diabetic cases ($>600$ False Negatives).
* **Selected Architecture:** `HistGradientBoostingClassifier` tuned via 5-fold Stratified Cross-Validation on PR-AUC achieved the strongest global discrimination (ROC-AUC: 0.9787, PR-AUC: 0.8852), detecting 90.29% of all diabetic cases while keeping precision at 48.62%.
* **Top Explanatory Features:** Test-set permutation importance established `glucose_x_hba1c` ($\Delta\text{PR-AUC} = +0.3925$) and standalone `HbA1c_level` ($\Delta\text{PR-AUC} = +0.3525$, Odds Ratio $= 5.66$) as primary determinants.

---

## Failure Analysis & Clinical Governance

* **The Prediabetic Twilight Zone:** Errors are concentrated in the borderline prediabetic coordinate window ($\text{HbA1c} \in [5.7\%, 6.6\%]$ and $\text{Glucose} \in [126, 160]\text{ mg/dL}$), where true empirical diabetes prevalence is only $6.5\%\text{--}9.2\%$.
* **Subgroup Variance:** Non-diabetic patients aged $\ge 65$ exhibited an elevated False Positive Rate (27.52%) due to the natural compounding of age, hypertension, and borderline glucose readings. Conversely, missed diabetic patients were younger (mean age 48 vs. 63) and lacked cardiovascular comorbidities (93.2% normotensive).
* **Governance Notice:** This pipeline is intended strictly as a risk-stratification screening triage tool to prioritize patients for confirmatory laboratory evaluation. It does not output medical diagnoses.

---

## Environment Setup & Reproduction

```bash
# 1. Clone repository
git clone [https://github.com/divechamaithili/DataMining_Assignment1.git](https://github.com/divechamaithili/DataMining_Assignment1.git)
cd DataMining_Assignment1/part1_diabetes_prediction

# 2. Virtual environment setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Pipeline execution
python src/eda.py
python src/train.py
python src/evaluate.pyß