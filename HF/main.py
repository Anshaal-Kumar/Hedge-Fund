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

    # Print the entire DataFrame
    # print(df)

    # Drop duplicates
    df_cleaned = df.drop_duplicates()
    print("\nDataFrame after dropping duplicates:\n", df_cleaned)

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:  # Catch any other exceptions that may arise
    print(f"An error occurred: {e}")

# SPLITTING DATA 
# Add your code for splitting data here

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
