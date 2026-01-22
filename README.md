# Lung Cancer Classification Using Snowflake and Logistic Regression

## Project Overview
This project implements a **Lung Cancer Classification system** by integrating **Snowflake Cloud Data Warehouse** with **Machine Learning**. Patient data is fetched directly from Snowflake using the Snowflake Python connector, preprocessed, and classified using **Logistic Regression**.

The project demonstrates real-world data engineering + machine learning integration using cloud-based databases.

---

## Key Highlights
- Cloud data retrieval using **Snowflake**
- Data preprocessing and label encoding
- Logistic Regression classification
- Model evaluation using multiple metrics
- End-to-end ML pipeline in Google Colab

---

## Dataset Source
- Data stored in **Snowflake Database**
- Table Name: `CANCER`
- Schema: `PUBLIC`

### Target Variable
- `LUNG_CANCER`
  - 0 → No Lung Cancer
  - 1 → Lung Cancer

### Features
- Demographic attributes (e.g., Gender, Age)
- Medical and behavioral indicators
- Total Features Used: **15**

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Snowflake Connector for Python
- Google Colab

---

## Workflow
1. Connect to Snowflake database
2. Fetch data using SQL query
3. Perform data inspection and validation
4. Encode categorical variables
5. Split dataset into training and testing sets (80:20)
6. Train Logistic Regression model
7. Evaluate model performance

---

## Snowflake Data Connection
```python
import snowflake.connector

conn = snowflake.connector.connect(
    user='USERNAME',
    password='PASSWORD',
    account='ACCOUNT_ID',
    database='NEWDB',
    schema='PUBLIC',
)
```

##  Model Training
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
```
## Model Evaluation
```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, lr.predict(x_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr.predict(x_test)))
print("Classification Report:\n", classification_report(y_test, lr.predict(x_test)))
```
Performance Metrics
Accuracy: 90.32%
Precision (Cancer): 0.89
Recall (Cancer): 1.00
F1-Score (Cancer): 0.94
Confusion Matrix
[[ 7  6]
 [ 0 49]]

## Project Structure
├── lung_cancer_classification.ipynb
├── README.md
(Data fetched directly from Snowflake — no local CSV required)

## Future Enhancements
Implement KNN and Naive Bayes models
Hyperparameter tuning
Handle class imbalance
Feature importance analysis
Model deployment using Flask or FastAPI
Automate Snowflake data ingestion pipeline

## Author
Saravanavel E
AI & Data Science Student
GitHub: https://github.com/SaravanavelE

## License
This project is intended for educational and academic purposes only.
This project is intended for educational and academic purposes only. warehouse='COMPUTE_WH'
)
