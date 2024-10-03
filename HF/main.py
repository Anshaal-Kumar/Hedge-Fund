# numpy
# pandas
# scikit
# matplotlib
# openpyxl
# tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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


# EVALUATE MODEL
# Add your code for model evaluation here

# FINE TUNING 
# Add your code for fine-tuning here

# PREDICT
# Add your prediction code here

# VISUALIZE
# Add your visualization code here

# DEPLOY
# Add your deployment code here
