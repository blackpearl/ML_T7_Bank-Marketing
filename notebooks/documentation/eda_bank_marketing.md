
# 📊 Exploratory Data Analysis (EDA): Bank Marketing Campaign Datasets

We analyze two datasets from a Portuguese bank's direct marketing campaigns to understand customer behavior and campaign effectiveness in subscribing to term deposits.

---

## ✅ Target Variable: `y`

The target indicates whether a client subscribed to a term deposit.

### Target Class Distribution
- Both datasets show **significant class imbalance**.
- Approximately **88.3%** of clients did **not** subscribe (`y = no`), while only **11.7%** said yes.

> 🟠 **Modeling Implication**: Class imbalance must be addressed (e.g., via resampling or class-weighting).

---

## 🧮 Numeric Feature Overview

| Feature   | Mean    | Min   | Max    | Notes                                                                 |
|-----------|---------|-------|--------|-----------------------------------------------------------------------|
| age       | 40.9    | 18    | 95     | Wide age range                                                        |
| balance   | 1362.27 | -8019 | 102127 | Strong right-skew; outliers present                                   |
| duration  | 258.16  | 0     | 4918   | Duration of last contact (very predictive, but not usable pre-call)   |
| campaign  | 2.76    | 1     | 63     | Number of contacts during campaign                                    |
| pdays     | 40.2    | -1    | 871    | -1 indicates no prior contact                                         |
| previous  | 0.58    | 0     | 275    | Number of contacts before this campaign                               |

> ⚠️ **Note**: `duration` is highly predictive but should be excluded in real-time models to prevent data leakage.

---

## 📈 Feature Distributions

### Age Distribution
```python
sns.histplot(data=bank_full, x='age', bins=30, kde=True)
```
- Most clients fall between **25–60 years**.

### Call Duration Distribution
```python
sns.histplot(data=bank_additional_full, x='duration', bins=30, kde=True)
```
- **Highly right-skewed** distribution: most calls are short.

---

## 🔢 Categorical Variables

- **Job**: 12 categories (e.g., management, technician, blue-collar)
- **Marital**: married, single, divorced
- **Education**: primary, secondary, tertiary, unknown
- **Default, Housing, Loan**: binary (yes/no)
- **Contact Method**: cellular, telephone, unknown
- **Month**: Campaign month (e.g., may, jul, aug)
- **Poutcome**: Outcome of previous campaign (success, failure, unknown, other)

> 🧠 **Note**: Categorical variables will need encoding (e.g., one-hot or label encoding).

---

## 🔗 Correlation Analysis

### `bank-full.csv` Correlation Matrix

|         | age       | balance   | duration  | campaign  | pdays     | previous  |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| age     | 1.00      | 0.098     | -0.005    | 0.005     | -0.024    | 0.001     |
| balance | 0.098     | 1.00      | 0.022     | -0.015    | 0.003     | 0.017     |
| duration| -0.005    | 0.022     | 1.00      | -0.085    | -0.002    | 0.001     |
| campaign| 0.005     | -0.015    | -0.085    | 1.00      | -0.089    | -0.033    |
| pdays   | -0.024    | 0.003     | -0.002    | -0.089    | 1.00      | 0.455     |
| previous| 0.001     | 0.017     | 0.001     | -0.033    | 0.455     | 1.00      |

> 🔍 **Key Insights**:
- **pdays** and **previous** show a **moderate correlation** (0.455)
- Most other pairs have **weak or negligible correlation** (|r| < 0.1)

### `bank-additional-full.csv` Correlation Highlights
- **euribor3m** ↔ **nr.employed**: Strong positive correlation
- **emp.var.rate** ↔ **euribor3m**: Strong negative correlation

> 🧭 These reflect macroeconomic patterns, indicating that interest rates and employment variables move together.

---

## 🧼 Data Quality & Preprocessing (Pending)
- Missing values are encoded (e.g., `unknown`)
- Requires:
  - Encoding of categorical variables
  - Handling of outliers (e.g., in `balance`, `duration`)
  - Feature scaling and transformation (as needed)
  - Addressing class imbalance

---

## 📦 Dataset Overview

| Dataset                  | Rows   | Columns | Notes                                            |
|--------------------------|--------|---------|--------------------------------------------------|
| `bank-full.csv`          | 45,211 | 17      | Older dataset; fewer macroeconomic indicators    |
| `bank-additional-full.csv` | 41,188 | 21      | Richer dataset; includes economic context fields |

---

## ✅ Summary & Next Steps

- **Class imbalance** and **outlier handling** are key for preprocessing.
- **Duration** is highly informative but should be excluded in pre-call models.
- **Categorical feature engineering** will play a significant role.
- **Macroeconomic features** in the additional dataset provide modeling depth.

> 🔧 Prepare data preprocessing pipeline and modeling strategy with attention to leakage, encoding, and class imbalance.
