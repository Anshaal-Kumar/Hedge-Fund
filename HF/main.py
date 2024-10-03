# numpy
# pandas
# scikit
# matplotlib
# openpyxl
# tensorflow

import pandas as pd
import numpy as np

#graph plotting
import matplotlib.pyplot as plt

#split test
from sklearn.model_selection import train_test_split

#evaluate 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#fine tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Visualize code 
import matplotlib.pyplot as plt
import seaborn as sns




# Load your data
file_path = "../archive/SP500.csv"
df = pd.read_csv(file_path)

# Print initial dataset shape
print("Initial shape of DataFrame:", df.shape)

# Check for NaNs in the initial dataset
print("\nNumber of missing values in each column:\n", df.isnull().sum())

# DATA PREPROCESSING
# Convert 'DATE' column to datetime format without coercing NaNs
if 'DATE' in df.columns:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Drop rows where 'DATE' is NaN, if necessary
df_cleaned = df.dropna(subset=['DATE'])

# Convert remaining columns to numeric (without touching 'DATE')
df_cleaned[['SP500', 'SP500TR']] = df_cleaned[['SP500', 'SP500TR']].apply(pd.to_numeric, errors='coerce')

# Check the DataFrame after conversion to numeric
print("\nDataFrame after converting to numeric:\n", df_cleaned.head())

# Check for NaNs before dropping rows
print("\nNumber of missing values after conversion:\n", df_cleaned.isnull().sum())

# Drop rows with missing values in the numeric columns
df_cleaned = df_cleaned.dropna()

# Check how many rows and columns are left after cleaning
print("\nShape of DataFrame after cleaning:", df_cleaned.shape)

# If no rows are left after cleaning, print a message and exit
if df_cleaned.shape[0] == 0:
    print("No data left after cleaning. Please check your dataset and preprocessing steps.")
else:
    # Separate features and target variable
    target_column = 'SP500'  # Replace with your actual target column name
    if target_column not in df_cleaned.columns:
        print(f"Target column '{target_column}' not found in the data.")
    else:
        X = df_cleaned.drop(columns=[target_column, 'DATE'])  # Drop 'DATE' from features
        y = df_cleaned[target_column]  # Target variable

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Print shapes of the datasets
        print("Training set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)
        print("Training target shape:", y_train.shape)
        print("Testing target shape:", y_test.shape)

        # Initialize the Linear Regression model
        model = LinearRegression()

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

# EVALUATE MODEL
# Add your code for model evaluation here

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Output results
        print(f"Mean Squared Error: {mse}")
        print(f"R-Squared: {r2}")

        # Plot the actual vs predicted values
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Stock Prices")
        plt.show()




# FINE TUNING 
# Add your code for fine-tuning here
# Define the model
ridge = Ridge()

# Set the hyperparameter grid
param_grid = {'alpha': [0.1, 1, 10, 100]}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_



# PREDICT
# Add your prediction code here
# Assuming you have already trained your model and have the following variables:
# model (the trained model)
# X_test (the test features)
# y_test (the actual target values for the test set)

# Make predictions
y_pred = model.predict(X_test)

# Print the predicted values
print("Predicted values:", y_pred)

# Print the actual values for comparison
print("Actual values:", y_test.values)

# Evaluate the model's performance using metrics like Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")

# Optional: Visualize the predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot(y_test, y_test, color='red', linewidth=2, label='Perfect Prediction Line')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


# VISUALIZE
# Add your visualization code here

# Assuming y_test contains actual values and y_pred contains predicted values

# 1. Scatter Plot of Predicted vs. Actual Values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of equality
plt.title('Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid()
plt.show()
plt.savefig('predicted_vs_actual.png')  # Save the plot as an image


# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')  # Line at 0
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# 3. Line Plot of Actual vs. Predicted Values Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['DATE'].iloc[-len(y_test):], y_test, label='Actual Values', color='blue')
plt.plot(df['DATE'].iloc[-len(y_test):], y_pred, label='Predicted Values', color='orange')
plt.title('Actual vs. Predicted Values Over Time')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# DEPLOY
# Add your deployment code here
