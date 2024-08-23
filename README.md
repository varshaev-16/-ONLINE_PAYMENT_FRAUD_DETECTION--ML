# -ONLINE_PAYMENT_FRAUD_DETECTION--ML
Objective

Create a machine learning system to detect and prevent fraudulent online transactions.


Summary

Using a Kaggle dataset, this project involves:

Gathering and cleaning transaction data.
Engineering features to differentiate legitimate and fraudulent activities.
Evaluating algorithms such as Random Forest, XGBoost, and Decision Trees.
Addressing class imbalance using techniques like SMOTE.
Training models on historical data and validating them using cross-validation.
Integrating a real-time detection model to flag suspicious transactions.
Assessing model performance using metrics such as accuracy, precision, and recall.

Problem Statement

With the rise in online transactions, fraudulent activities have also increased, leading to significant financial losses and reputational damage for businesses. Therefore, a robust system to detect and prevent fraud in real-time has become critical.

Dataset

The dataset used in this project is sourced from Kaggle and contains transaction data with the following features:

Step: Unit of time.
Type: Type of online transaction.
Amount: Transaction amount.
NameOrig: Customer starting the transaction.
OldbalanceOrg: Balance before the transaction.
NewbalanceOrg: Balance after the transaction.
NameDest: Account receiving the transaction.
OldbalanceDest: Initial balance of the receiver before the transaction.
NewbalanceDest: New balance of the receiver after the transaction.
IsFraud: Indicates whether the transaction is fraudulent.

Libraries and Tools

The following libraries were used:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Algorithms Used
Random Forest Classifier
XGBoost Classifier
Decision Tree Classifier

Results

The models developed in this project demonstrated strong performance in detecting fraudulent transactions:

Random Forest Model: Accuracy of 98.96%.
XGBoost Model: Accuracy of 99.03%.
Decision Tree Model: Accuracy of 98.63%.
These results highlight the models' effectiveness in identifying fraudulent activities with high accuracy.

Conclusion

The development of this machine learning system has yielded highly accurate models for detecting and preventing fraudulent online transactions. By leveraging data preprocessing, feature engineering, and advanced algorithms, this project provides a robust solution to enhance transaction security and protect financial systems from fraud.
