import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv('data.csv')  # Replace with actual data loading
# X = df[feature_columns]  # Define feature columns
# y = df['target_column']  # Define target column
# Example placeholders for data
X, y = np.random.rand(100, 10), np.random.rand(100) * 100000  # Dummy data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment
experiment_name = "Data_Science_Salaries_Experiment"
mlflow.set_experiment(experiment_name)

# Hyperparameter tuning for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

best_model = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_gb_model = grid_search.best_estimator_

# Log the best model's results
with mlflow.start_run():
    mlflow.log_param("model", "Gradient Boosting (Optimized)")
    mlflow.log_params(grid_search.best_params_)

    y_pred_best = best_gb_model.predict(X_test)

    rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    mlflow.log_metric("rmse", rmse_best)
    mlflow.log_metric("mae", mae_best)
    mlflow.log_metric("r2", r2_best)
    mlflow.sklearn.log_model(best_gb_model, "SalaryPredictorModel")  # Use a registered model name

    print(f"Optimized Model Metrics:")
    print(f"RMSE: {rmse_best}")
    print(f"MAE: {mae_best}")
    print(f"R²: {r2_best}")

# Streamlit App
st.title("Salary Prediction App")

# Define feature names
feature_names = [
    "experience",      # Years of experience
    "education_level", # Education level (encoded)
    "job_title_Data_Scientist",
    "job_title_Data_Analyst",
    "job_title_Machine_Learning_Engineer",
    "country_US",
    "country_CA",
    "country_UK",
    "gender_Male",
    "gender_Female",
]

# Load the best model from MLflow
model = best_gb_model  # Adjust model name

# User input for features
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", min_value=0.0)

# Convert user input to DataFrame
input_data = pd.DataFrame(user_input, index=[0])

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

# Display model performance metrics
st.subheader("Model Performance Metrics")
metrics = {
    "RMSE": st.session_state.get("rmse", "Not available"),
    "MAE": st.session_state.get("mae", "Not available"),
    "R²": st.session_state.get("r2", "Not available")
}
for metric, value in metrics.items():
    st.write(f"{metric}: {value}")

# Feature Importance visualization
st.subheader("Feature Importance")
feature_importance = best_gb_model.feature_importances_
features = np.array(feature_names)
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(features[indices], feature_importance[indices], align='center')
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
st.pyplot(plt)
