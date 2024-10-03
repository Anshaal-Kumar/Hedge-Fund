# numpy
# pandas
# scikit
# matplotlib
# openpyxl
# tensorflow

import pandas as pd
import numpy as np 
import yfinance as yf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score

# PUTTING DATA IN EXCEL
file_path = "../archive/SP500.csv"

# DATA PROCESSING 
try:
    # Load all rows and columns
    df = pd.read_csv(file_path)

    # Set display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping for large DataFrames

    # Drop duplicates
    df_cleaned = df.drop_duplicates()

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:  # Catch any other exceptions that may arise
    print(f"An error occurred: {e}")

# SPLITTING DATA 
try:
    # Separate features and target variable
    target_column = 'SP500'  # Replace with your actual target column name
    X = df_cleaned.drop(columns=[target_column])  # Features
    y = df_cleaned[target_column]  # Target variable

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print shapes of the datasets
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    # Print a sample of the training and testing data
    print("\nTraining Features:\n", X_train.head())
    print("\nTesting Features:\n", X_test.head())
    print("\nTraining Target:\n", y_train.head())
    print("\nTesting Target:\n", y_test.head())

except Exception as e:
    print(f"An error occurred during data splitting: {e}")

# Print the cleaned DataFrame
print("\nCleaned DataFrame:\n", df_cleaned)

# ALGO MODEL SELECTION
# Add your code for model selection here

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
