# Strategic Marketing: Insights from the Bank Marketing Dataset


Data Science Institute - Cohort 6 - ML Team 07  Project Report


**Members**  
Rajthilak Chandran  
Feihong Nan  
Darling, Oscanoa  
Rehan, Ahmed  
Rituparna, Bera  
Jenniffer Carolina, Triana Martinez


# Business Case
To build a predictive model that helps the bank optimize its marketing campaign strategy by identifying customers most likely to subscribe to a term deposit. The goal is to increase campaign efficiency by reducing outreach to unlikely responders.


***In today’s competitive banking environment, direct marketing campaigns often yield low conversion rates. Traditional outreach strategies waste resources by targeting clients who are unlikely to respond positively. Our project addresses this challenge by using machine learning to predict which clients are most likely to subscribe to a term deposit.
By identifying high-potential leads, banks can focus their efforts on the right audience—leading to higher ROI, lower campaign costs, and improved customer satisfaction. This allows marketing and sales teams to make data-driven decisions that directly boost operational efficiency and revenue.***


# Project Objective


What value do our projects bring to the industry?


***Objective: To build a predictive model that helps the bank optimize its marketing campaign strategy by identifying customers most likely to subscribe to a term deposit. The goal is to increase campaign efficiency by reducing outreach to unlikely responders.***


We believe our project can help banks and financial institutions improve how they run their marketing campaigns. By using machine learning to predict which clients are most likely to subscribe to a term deposit, we can help reduce wasted outreach and focus efforts on people who are more likely to respond. This means better results for the business, lower costs, and a more personalized experience for customers. In a competitive industry like banking, being able to target the right people at the right time is a big advantage

# Project Structure

<img width="952" height="583" alt="image" src="https://github.com/user-attachments/assets/eef354e1-21fe-4804-9494-9d689ab063da" />

# Project Overview


**Who are your stakeholders and why do they care about your project?**


***Stakeholders***


Marketing Teams: Want to optimize campaign ROI


Sales Teams: Need qualified leads


Executives: Focused on strategic growth and cost reduction


Clients: Benefit from more relevant and personalized outreach


**How will you answer your business question with your chosen dataset?**


We’re using the Bank Marketing dataset from UCI, which contains detailed information about clients, their previous interactions with marketing campaigns, and several economic indicators. Our goal is to help the bank improve its conversion rate, the percentage of contacted clients who end up subscribing to a term deposit.


To do this, we started by exploring the dataset to uncover patterns and relationships that might influence a client’s decision. We then cleaned and prepared the data, making sure it was ready for modeling by handling categorical variables and addressing class imbalance.
Next, we built machine learning models to predict whether a client is likely to subscribe. These predictions allow the bank to focus its outreach on individuals who are more likely to respond positively. We evaluated our models using metrics like precision, recall, F1-score, and ROC-AUC to ensure they’re both accurate and practical.


To make our results interpretable, we used SHAP values to explain which features are most influential in the model’s predictions. This helps the bank understand what drives client behavior and supports more informed decision-making.


By identifying high-potential leads, our approach directly supports the bank’s ability to improve its conversion rate, which is a key performance indicator for marketing success.


**What are the risks and uncertainties?**




We’re aware that certain features like call duration might inflate our model’s performance, so we’re testing models with and without it to make sure our predictions are realistic. We’re also dealing with class imbalance, which could affect how well our model identifies potential subscribers. Our goal is to build a model that not only performs well technically but also helps improve the conversion rate in a way that’s reliable and scalable.


**What methods and technologies will you use?**


We’re using Python and tools like scikit-learn, xgboost, and shap to build and explain our models. Everything is tracked in GitHub, and we’re following best practices for collaboration. Our reproducible notebook shows how our model can be used to predict outcomes and support decisions that directly impact the bank’s conversion rate, a key metric for campaign success.


## Requirements
![Data Analysis](https://img.shields.io/badge/-Data_analysis-informational?style=for-the-badge&logo=GooglePodcasts&logoColor=white&color=FFC98B)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white)&nbsp;
![NumPy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)&nbsp;
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/-Machine_Learning-informational?style=for-the-badge&logo=AIOHTTP&logoColor=white&color=FFB284)


This project relies on a suite of Python libraries and frameworks that support end-to-end data science workflows, including data preprocessing, machine learning, model evaluation, interpretability, and visualization:


### **Core Libraries**


* **pandas** – Essential for structured data manipulation, cleaning, and tabular analysis  
* **NumPy** – Powers efficient numerical computations and matrix operations


### **Machine Learning and Modeling**


* **scikit-learn** – Provides robust tools for supervised learning, preprocessing, model selection, and validation. Notable components:  
  * Estimators: `LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `BernoulliNB`  
  * Pipelines and Transformers: `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`  
  * Evaluation and Tuning: `accuracy_score`, `confusion_matrix`, `classification_report`, `mean_squared_error`, `GridSearchCV`, `StratifiedKFold`  
* **xgboost** – Implements the scalable and performant `XGBClassifier` for gradient boosting  
* **lightgbm** – Provides high-speed gradient boosting via `LGBMClassifier`, optimized for large datasets  
* **imblearn (imbalanced-learn)** – Offers strategies for handling imbalanced data, including `SMOTE` and `SMOTENC`


### **Visualization and Diagnostics**


* **matplotlib** – Standard library for creating visualizations and plots  
* **scikit-plot** – Enhances plotting of classifier performance and metrics  
* **missingno** – Visualizes missing data patterns to support quality assessments  
* **ydata\_profiling** – Automates exploratory data analysis with comprehensive data profiling reports


### **Optimization and Interpretability**


* **hyperopt** – Facilitates hyperparameter tuning using Bayesian optimization via components such as `tpe`, `Trials`, `hp`, `fmin`, and `space_eval`  
* **shap** – Enables model interpretability through SHAP (SHapley Additive Explanations), highlighting feature contributions to predictions


### **System and Serialization**


* **os** – Provides operating system interface for file and environment management  
* **pickle** – Used to serialize and persist Python objects (e.g., trained models)  
* **Pathlib** – Offers intuitive, object-oriented file system navigation and path handling


## Dataset Overview - Understanding the Raw Data
**URL**: [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) https://archive.ics.uci.edu/dataset/222/bank+marketing


This dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is often to predict whether a client will subscribe to a term deposit (variable y).


**Size:**
* Examples: 45,211 (for the bank-full.csv)
* Features: 17 input features + 1 output


**Target (label):**


* y — whether the client subscribed to a term deposit: yes or no (binary classification)


**Main Features:**


* Client attributes: age, job, marital, education, default, housing, loan
* Campaign-related: contact, month, day_of_week, duration
* Social/economic context: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed



# 📊 Exploratory Data Analysis (EDA): Bank Marketing Campaign Datasets

We analyze two datasets from a Portuguese bank's direct marketing campaigns to understand customer behavior and campaign effectiveness in subscribing to term deposits.

---
## 📦 Dataset Overview

| Dataset                  | Rows   | Columns | Notes                                            |
|--------------------------|--------|---------|--------------------------------------------------|
| `bank-full.csv`          | 45,211 | 17      | Older dataset; fewer macroeconomic indicators    |
| `bank-additional-full.csv` | 41,188 | 21      | Richer dataset; includes economic context fields |

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

* Duration is a strong predictor but should be excluded if the goal is to predict before the call is made.
* Many categorical variables: will require encoding (e.g., one-hot or label encoding)
* Outliers: Significant outliers in balance and duration should be handled or scaled.
---

## 🧼 Data Quality & Preprocessing (Pending)
- Missing values are encoded (e.g., `unknown`)
- Requires:
  - Encoding of categorical variables
  - Handling of outliers (e.g., in `balance`, `duration`)
  - Feature scaling and transformation (as needed)
  - Addressing class imbalance

---




## Data Cleaning and Handling Missing Values - Preprocessing of the data
PENDING


## Model Development

We developed a streamlined pipeline to efficiently handle both preprocessing and model training tasks.
•	Numerical features were standardized using StandardScaler to bring all values to a similar scale. This avoids bias toward features with larger values and improves model convergence.
•	Categorical features were transformed using OneHotEncoder, converting categories into binary columns. This ensures the model treats them as distinct inputs without assuming any order.
•	To keep the workflow clean and consistent, we used a ColumnTransformer to apply all preprocessing steps in a single, unified process—helping avoid data leakage and improving reproducibility.
We trained and evaluated the following models:
•	Logistic Regression
•	Decision Tree
•	Random Forest
•	Neural Network (built using Keras)
After comparing their performance on metrics like accuracy, precision, recall, and F1-score, the Neural Network model outperformed the others and was selected as the final model for deployment.


## Handling Imbalanced Data


Raj


## Model Training and Evaluation - Jennifer

## 🔍 Model Evaluation – Logistic Regression (Baseline)  - Feihong

We trained a baseline **Logistic Regression** model to predict whether a client will subscribe to a term deposit (`y`).

### 📊 Evaluation Results

**Confusion Matrix**:

[[10828    137]

[ 1083   309]]

**Classification Report**:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **0 (no)**  | 0.91      | 0.99   | 0.95     | 10,965  |
| **1 (yes)** | 0.69      | 0.22   | 0.34     | 1,392   |

- **Overall Accuracy**: 90%  
- **Macro Avg F1**: 0.64  
- **Weighted Avg F1**: 0.88

### 📝 Key Observations

- The model performs well for the **majority class** (`no`), but poorly on the **minority class** (`yes`) due to significant **class imbalance**.
- Despite high overall accuracy, the **low recall for the 'yes' class (0.22)** indicates many false negatives.
- Logistic Regression provides a **simple, interpretable baseline** for further modeling.

### 🛠️ Recommendations for Improvement

- **Handle class imbalance**:
  - Use `class_weight='balanced'`
  - Apply oversampling techniques like **SMOTE**
- **Explore advanced models**:
  - Try **Random Forest**, **XGBoost**, or **Gradient Boosting** for better handling of complex patterns.
- **Use alternative evaluation metrics**:
  - Include **precision-recall curves** and **ROC-AUC** to better assess performance on the minority class.


## Model Deployment and Interpretation
After completing our data analysis, preprocessing, and model training phases, we will make our machine learning solution accessible and actionable for bank marketing teams. Our deployment strategy focuses on creating a user-friendly interface that translates model predictions into real business value.

### Deployment Strategy

Once we finalize our model selection and comparison (based on our evaluation metrics that we will analyze), we'll deploy the best-performing model using Streamlit with pickle files. This approach will give us a clean, interactive web application that marketing professionals can use without any ML technical background.

#### Why Streamlit for Our Use Case

We chose Streamlit because it allows us to quickly build and iterate on our application while maintaining the flexibility to demonstrate our model's capabilities. For a project focused on proving concept and demonstrating value, Streamlit provides the perfect balance of functionality and simplicity under tha /app directory.

Our deployment follows a straightforward architecture:
- **Model artifacts**: Saved using joblib for consistent loading
- **Preprocessing pipeline**: Encoders and scalers preserved from training
- **Interactive interface**: Real-time prediction with user-friendly inputs

#### Application Interface

![alt text](image.png)
Streamlit App - Main Interface

![alt text](image-1.png)
Streamlit App - Prediction Results

The application captures all the key customer features we identified during our analysis and provides immediate feedback on subscription probability. Users can adjust inputs and see how different customer characteristics affect the likelihood of term deposit subscription.

### Business Application and ROI Optimization

Here's where our technical work translates into actual business impact. The reason we're revisiting business value in the deployment section is simple: this is where the bank decides whether to invest resources based on expected returns.

#### Converting Predictions to Actionable Decisions

Our model doesn't just predict probabilities—it helps banks make smarter resource allocation decisions. Traditional telemarketing campaigns often have conversion rates around 11-12% (as we saw in our EDA). By targeting customers with higher predicted probabilities, we can potentially improve these rates significantly.

**The ROI calculation becomes straightforward:**
- **Cost per contact**: ~$5-10 (call center time, agent salary, phone costs)
- **Revenue per subscription**: ~$200-500 (depending on deposit amount and terms)
- **Break-even point**: Need conversion rates above 2-5%

If our model helps identify customers with 25-30% conversion probability (instead of the baseline 11%), the return on investment becomes substantial.

#### Practical Implementation

The application includes a threshold-based recommendation system. When a customer shows, let's say >60% subscription probability, we flag them as "High Priority" for immediate contact. This threshold can be adjusted based on:
- Campaign budget availability
- Call center capacity
- Seasonal factors (we found certain months perform better)

**Real-world usage scenario:**
1. Marketing team receives a list of potential customers
2. They input customer details into our app
3. Model provides probability score and recommendation
4. Team prioritizes high-probability customers for immediate outreach
5. Results feed back into our system for continuous improvement

### Model Interpretation and Explainability

Understanding *why* our model makes certain predictions is crucial for building trust with marketing teams and improving campaign strategies.

#### Feature Importance Insights

Our final model will provide clear insights into which customer characteristics matter most for term deposit subscriptions. Based on our initial analysis, we expect factors like:
- Previous campaign outcomes (success/failure history)
- Contact duration and timing
- Customer financial situation (balance, existing loans)
- Economic context (employment rates, interest rates)

#### Making Predictions Transparent

The Streamlit app shows both the raw customer data and the processed features that go into our model. This transparency helps users understand how their inputs translate into predictions and builds confidence in the system.

We're also prepared to integrate SHAP values if time permits, which would show exactly how each customer feature contributes to their specific prediction score.

### Production Considerations (Future Scope)

While our current focus is on demonstrating model effectiveness through local deployment, we recognize that scaling to production involves additional complexity.

#### Scaling and Performance

For production deployment, we'd need to consider:
- **API development**: Converting our Streamlit interface to REST APIs for integration with existing bank systems
- **Database integration**: Storing customer predictions and tracking campaign outcomes
- **Performance monitoring**: Ensuring response times meet business requirements

#### MLOPs Implementation (Optional)

If resources and timeline allow, we could implement MLOPs best practices:
- **Model versioning**: Track different model versions and their performance with MLflow
- **Automated retraining**: Schedule periodic updates with new campaign data  
- **Performance monitoring**: Track prediction accuracy against actual campaign results

However, these are enhancements rather than requirements for demonstrating our solution's value.

#### Cloud Deployment Overview

For production scale, AWS provides a straightforward path:
- **Compute**: EC2 or containerized deployment for the application
- **Storage**: S3 for model artifacts and campaign data
- **Database**: RDS for customer information and prediction history
- **Monitoring**: CloudWatch for application health and performance

The key is that our local deployment provides a solid foundation that can be migrated to cloud infrastructure when the business case justifies the additional investment.

### Documentation and Reproducibility

Our deployment will include comprehensive documentation to ensure:
- **Easy setup**: Clear instructions for running the application locally
- **Model reproducibility**: Steps to retrain models with updated data
- **Business integration**: Guidelines for incorporating predictions into existing workflows

This documentation approach ensures that our work can be maintained and extended by other team members or stakeholders.

### Expected Outcomes and Next Steps

The deployed application serves as a proof-of-concept that demonstrates how machine learning can improve marketing campaign efficiency. Success metrics include:
- **User adoption**: How frequently marketing teams use the tool
- **Prediction accuracy**: How well our model performs on new customers
- **Business impact**: Measurable improvements in conversion rates and ROI

Based on initial results and user feedback, we can then decide whether to invest in full production deployment with MLOPs implementation and cloud infrastructure.




## Further Model Assessment
Rehan


## Conclusion and Future Directions





