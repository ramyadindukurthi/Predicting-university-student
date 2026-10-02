# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:13:40 2025

@author: CMP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("StudentPerformanceFactors.csv")

df.head()
def calc_cgpa(score):
    if score >= 70:
        return 4
    elif score >= 60:
        return 3
    elif score >= 50:
        return 2
    elif score >= 40:
        return 1
    else:
        return 0


df['Previous_cgpa'] =np.array( [calc_cgpa(x) for x in df['Previous_Scores']], dtype=np.uint8)
df['Exam_Score_cgpa'] = np.array([calc_cgpa(y) for y in df['Exam_Score']], dtype=np.uint8)

# 1. Label Encoding (for binary categorical variables like 'Purchased')
label_encoder = LabelEncoder()
df['Access_to_Resources_Encoded'] = label_encoder.fit_transform(df['Access_to_Resources'])
df['Motivation_Level_Encoded'] = label_encoder.fit_transform(df['Motivation_Level'])
df['Family_Income_Encoded'] = label_encoder.fit_transform(df['Family_Income'])
df['Teacher_Quality_Encoded'] = label_encoder.fit_transform(df['Teacher_Quality'])
df['School_Type_Encoded'] = label_encoder.fit_transform(df['School_Type'])
df['Peer_Influence_Encoded'] = label_encoder.fit_transform(df['Peer_Influence'])
df['Parental_Education_Level_Encoded'] = label_encoder.fit_transform(df['Parental_Education_Level'])
df['Distance_from_Home_Encoded'] = label_encoder.fit_transform(df['Distance_from_Home'])
df['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])
df['Parental_Involvement_Encoded'] = label_encoder.fit_transform(df['Parental_Involvement'])
df['Extracurricular_Activities_Encoded'] = label_encoder.fit_transform(df['Extracurricular_Activities'])
df['Internet_Access_Encoded'] = label_encoder.fit_transform(df['Internet_Access'])
df['Learning_Disabilities_Encoded'] = label_encoder.fit_transform(df['Learning_Disabilities'])

# Drop multiple columns 
columns_to_drop = ['Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home', 'Gender', 'Parental_Involvement', 'Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']
df = df.drop(columns=columns_to_drop)
df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
# Drop any rows with missing values (optional, if your data has NaNs)
df = df.dropna()


# Split the dataset into input features (X) and target variable (y)
y = df['Exam_Score_cgpa']
X = df.drop(columns=['Exam_Score', 'Exam_Score_cgpa', 'Previous_Scores'])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# y_pred = [round(x) for x in y_pred]

print('After adjustment:\n')

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Optionally, print actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison_df.head()
print(list(X_test.columns))
print(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate a classification report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)




# Optionally, print actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison_df.head()
import joblib
# Save the model to a file
joblib.dump(model, 'random_forest_model.pkl')

from sklearn.tree import DecisionTreeClassifier

# Create and train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model using accuracy and classification report
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

from sklearn.neighbors import KNeighborsClassifier

# Create and train a K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict on the test data
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model using accuracy and classification report
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))


from sklearn.ensemble import VotingClassifier

# Create the Random Forest and XGBoost models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = rf_model

# Create a voting classifier (combine RF and XGBoost)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='hard')
voting_model.fit(X_train, y_train)

# Predict on the test data
y_pred_voting = voting_model.predict(X_test)

# Evaluate the model using accuracy and classification report
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Hybrid Model Accuracy (Random Forest + XGBoost): {voting_accuracy * 100:.2f}%")
print("\nHybrid Model Classification Report:")
print(classification_report(y_test, y_pred_voting))

import joblib
import numpy as np
import pandas as pd
# Load the model from the file
loaded_rf_model = joblib.load('random_forest_model.pkl')

feature_names = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity', 'Previous_cgpa', 'Access_to_Resources_Encoded', 'Motivation_Level_Encoded', 'Family_Income_Encoded', 'Teacher_Quality_Encoded', 'School_Type_Encoded', 'Peer_Influence_Encoded', 'Parental_Education_Level_Encoded', 'Distance_from_Home_Encoded', 'Gender_Encoded', 'Parental_Involvement_Encoded', 'Extracurricular_Activities_Encoded', 'Internet_Access_Encoded', 'Learning_Disabilities_Encoded']
X_test_manual = pd.DataFrame([[5, 24, 10, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0]], columns=feature_names) # reduce score to 0 to 1
# Now you can use the loaded model
predictions = loaded_rf_model.predict(X_test_manual)
print(predictions)
