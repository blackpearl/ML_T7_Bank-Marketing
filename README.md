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


## Exploratory Analysis Summary for Predictive Modeling






| Feature   | Mean    | Min   | Max    | Notes                                                                 |
|-----------|---------|-------|--------|-----------------------------------------------------------------------|
| age       | 40.9    | 18    | 95     | Wide age range                                                        |
| balance   | 1362.27 | -8019 | 102127 | Strong right-skew; outliers present                                   |
| duration  | 258.16  | 0     | 4918   | Duration of last contact in seconds (very predictive, but not usable for real-time targeting) |
| campaign  | 2.76    | 1     | 63     | Number of contacts during campaign                                    |
| pdays     | 40.2    | -1    | 871    | -1 indicates no prior contact                                         |
| previous  | 0.58    | 0     | 275    | Number of contacts before this campaign  


**Categorical Variables (Samples)**
* Job: 12 categories (e.g., management, blue-collar, technician, etc.)
* Marital: married, single, divorced
* Education: primary, secondary, tertiary, unknown
* Default, Housing, Loan: binary yes/no
* Contact Method: cellular, telephone, unknown
* Month: Campaign month (e.g., may, jul, aug)
* Poutcome: Outcome of previous campaign (success, failure, other, unknown)


**Initial Obersavation**
* Highly Imbalanced Target: ~88.3% of clients did not subscribe (y = no)
 Only ~11.7% said yes


  **![][image1]**


[image1]: <data:image/png;base64,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>


* Duration is a strong predictor but should be excluded if the goal is to predict before the call is made.
* Many categorical variables: will require encoding (e.g., one-hot or label encoding)
* Outliers: Significant outliers in balance and duration should be handled or scaled.




## Data Cleaning and Handling Missing Values - Preprocessing of the data
PENDING


## Correlation between features
| index     | age         | balance     | duration    | campaign    | pdays       | previous    |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| age       | 1           | 0.097782739 | -0.004648428| 0.004760312 | -0.023758014| 0.001288319 |
| balance   | 0.097782739 | 1           | 0.02156038  | -0.014578279| 0.003435322 | 0.016673637 |
| duration  | -0.004648428| 0.02156038  | 1           | -0.084569503| -0.00156477 | 0.001203057 |
| campaign  | 0.004760312 | -0.014578279| -0.084569503| 1           | -0.088627668| -0.03285529 |
| pdays     | -0.023758014| 0.003435322 | -0.00156477 | -0.088627668| 1           | 0.454819635 |
| previous  | 0.001288319 | 0.016673637 | 0.001203057 | -0.03285529 | 0.454819635 | 1           |


This matrix shows the pairwise Pearson correlation coefficients between six numerical variables: **age**, **balance**, **duration**, **campaign**, **pdays**, and **previous**.


#### **🔍 Key Observations of the Correlation Matrix:**


* **Strongest Correlation:**


  * `pdays` and `previous`: **0.4548**


    * Indicates a **moderate positive correlation**. More days since last contact tends to associate with a higher number of previous contacts.


* **Weak Correlations (|r| \< 0.1):**


  * `age` with other variables: All correlations are weak, with the highest being with `balance` (**0.0978**).


  * `balance` shows minimal correlation with all other variables.


  * `duration`, `campaign`, and `pdays` also show weak or no linear relationship with most variables.


* **Negative Correlations (all weak):**


  * `duration` and `campaign`: **\-0.0846**


  * `campaign` and `pdays`: **\-0.0886**


  * `campaign` and `previous`: **\-0.0329**
## Model Development


Ritu


## Handling Imbalanced Data


Raj


## Model Training and Evaluation
Jennifer


## Model Deployment and Interpretation


Darling


## Further Model Assessment
Rehan


## Conclusion and Future Directions
This project demonstrated how machine learning can support targeted marketing efforts in the banking sector. By building and evaluating models to predict client subscription to term deposits, we developed a solution that improves campaign efficiency and decision-making.

Our final model, XGBoost, showed strong performance across evaluation metrics. With SHAP values and ranking curves, we ensured the results were both accurate and interpretable, allowing marketing teams to prioritize outreach based on likelihood to convert.

To strengthen future versions of the model, we recommend incorporating additional client features such as:

* **Customer tenure**: Number of years the client has held an account with the bank  
* **Product usage**: Total number of active financial products (e.g., savings, credit cards, loans)  
* **Channel preference**: Indicated communication preferences (email, phone, SMS) or history of responsiveness  
* **Credit standing**: Proxy indicators of financial health, such as internal risk rating or credit score band

These features could enhance model precision and reflect behavioral patterns not captured in the original dataset.

Next steps include retraining the model with newer data, integrating these additional features, and deploying the solution via real-time dashboards or APIs. These improvements will increase the model’s value in operational settings and support dynamic campaign strategies.




