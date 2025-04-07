# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\code\\AP lab\\kindergarten immunizations.csv"
df = pd.read_csv(file_path, encoding="latin1")

# Select relevant features
features = ["ENROLLMENT", "COUNT"]  # Numeric features
target = "PERCENT"  # Predict the actual vaccination percentage

# Handle missing values
df[features] = df[features].fillna(df[features].median())
df[target] = df[target].fillna(df[target].median())

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Scale the numeric data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train Random Forest Regressor model with Grid Search
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = (y_test == y_pred).mean()  # Accuracy calculation

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2%}")

# Calculate total vaccinated children
df["VACCINATED_CHILDREN"] = (df["PERCENT"] / 100) * df["ENROLLMENT"]
total_vaccinated = df["VACCINATED_CHILDREN"].sum()
print(f"Total Vaccinated Children: {total_vaccinated:.0f}")

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # 45-degree line
plt.xlabel("Actual Vaccination %")
plt.ylabel("Predicted Vaccination %")
plt.title("Random Forest Regression: Actual vs Predicted Vaccination %")
plt.show()

# Feature Importance
importance = best_model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=features, y=importance, palette="viridis")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest Model")
plt.show()

# Scatter Plot: Distribution of Vaccinated Children
plt.figure(figsize=(8, 6))
plt.scatter(df["ENROLLMENT"], df["VACCINATED_CHILDREN"], alpha=0.6, color="green")
plt.xlabel("Total Enrollment")
plt.ylabel("Vaccinated Children")
plt.title("Distribution of Vaccinated Children")
plt.show()