# Employee Turnover Analytics

This project explores HR data to **predict which employees are likely to leave the organization**, using machine learning techniques and statistical analysis. It aims to equip HR teams with data-driven insights and retention strategies that reduce employee churn.

---

## Objectives

- Analyze key factors contributing to employee turnover.
- Build predictive models to identify employees at risk of leaving.
- Segment (cluster) employees based on satisfaction and evaluation patterns.
- Generate targeted **retention strategies** using risk-based segmentation.

---

## Dataset

The dataset used is the [HR Analytics dataset](https://www.kaggle.com/datasets) from Kaggle. It includes:

| Column Name              | Description                                           |
|--------------------------|-------------------------------------------------------|
| `satisfaction_level`     | Employee satisfaction score (0–1)                     |
| `last_evaluation`        | Last evaluation score (0–1)                           |
| `number_project`         | Number of projects assigned                           |
| `average_monthly_hours`  | Average monthly working hours                         |
| `time_spend_company`     | Tenure in the company (in years)                      |
| `work_accident`          | Whether the employee had a work accident              |
| `promotion_last_5years`  | Promotion in the last 5 years                         |
| `department`             | Employee's department                                 |
| `salary`                 | Categorical salary level (low, medium, high)          |
| `left`                   | Target variable (1: left the company, 0: stayed)      |

---

## Exploratory Data Analysis (EDA)

Key findings:

- **Lower satisfaction** is highly correlated with turnover.
- **Very high or very low project loads** can increase attrition.
- **Long tenure** may be associated with burnout or stagnation.
- **Promotions and salary** have a role in employee decisions but show weaker correlation.

Visualizations include:

- Correlation heatmap
- Distribution plots (satisfaction, evaluation, working hours)
- Project count vs turnover
- Cluster plots of employee groups based on satisfaction & evaluation

---

## Modeling Workflow

### Preprocessing
- Categorical variables encoded using `get_dummies`.
- Addressed **class imbalance** using **SMOTE**.
- Train-test split using **Stratified Sampling** (80/20).

### Models Trained
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

### Evaluation Metrics
- Precision, Recall, F1-score (via 5-fold cross-validation)
- Confusion Matrix
- ROC Curve & AUC

| Model                 | Accuracy | Precision | Recall | F1-Score | AUC  |
|-----------------------|----------|-----------|--------|----------|------|
| Logistic Regression   | 78%      | 0.77–0.80 | 0.76–0.81 | 0.78–0.79 | ~0.86 |
| Random Forest         | 98%      | 0.97–0.99 | 0.97–1.00 | 0.98      | ~0.99 |
| Gradient Boosting     | 96%      | 0.94–0.97 | 0.94–0.97 | 0.96      | ~0.97 |

**Random Forest** was selected as the best-performing model.

---

## Clustering (K-Means)

Employees who left were grouped into **3 clusters**:

| Cluster | Satisfaction | Evaluation | Interpretation                               |
|---------|--------------|------------|----------------------------------------------|
| 0       | Low          | Low        | Disengaged, poor performance                 |
| 1       | High         | High       | Top performers who left, likely for growth   |
| 2       | Low          | High       | Burned out high performers                   |

These insights help design **targeted retention strategies** per group.

---

## Risk Segmentation and Retention Strategy

Using the trained Random Forest model, each employee's **turnover probability** was calculated and categorized:

| Zone                | Probability Range     | Suggested Strategy                              |
|---------------------|-----------------------|-------------------------------------------------|
| Safe Zone (Green)   | < 20%                 | Continue engagement and recognition             |
| Low-Risk (Yellow)   | 20–60%                | Regular check-ins, growth discussions           |
| Medium-Risk (Orange)| 60–90%                | Re-engagement, mentorship, workload management  |
| High-Risk (Red)     | > 90%                 | Urgent 1-on-1s, compensation review, exit risk  |

A bar chart visualizes the distribution of employees across these zones.

---

## Installation & Usage

### Requirements
- Python 3.8+
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- imbalanced-learn (`pip install imbalanced-learn`)

### Run the notebook

```
> git clone https://github.com/yourusername/employee-turnover-analytics.git
> cd employee-turnover-analytics
> jupyter notebook

```

---

## Folder Structure (Recommended)

```
employee-turnover-analytics/
│
├── data/
│   └── HR_comma_sep.csv

├── notebooks/
│   └── Employee_Turnover_Analytics.ipynb
│
├── images/
│   └── charts, ROC curves, confusion matrix, etc.
│
├── outputs/
│   └── Employee_Turnover_Analytics.html
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Feature selection or PCA to reduce dimensionality
- Test on real company data with anonymization
- Explainability using SHAP or LIME

--- 

## License

This project is for educational purposes only. Feel free to reuse the code with proper attribution.
