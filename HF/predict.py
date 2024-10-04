import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Load new data for prediction
new_data = {
    'SP500TR': [0.5, 1.0, 1.5],  # Example feature values
    # Add other features here as needed, e.g., 'feature2': [value1, value2, value3]
}
df_new = pd.DataFrame(new_data)

# Make predictions
predictions = model.predict(df_new)

# Output predictions
print("Predicted values:", predictions)

# Example to evaluate predictions against true values
# If you have true values (e.g., from a CSV)
# df_test = pd.read_csv('test_data.csv')
# X_test = df_test.drop(columns=['SP500', 'DATE'])  # Adjust according to your actual dataset
# y_test = df_test['SP500']

# Make predictions on the test set
# y_pred = model.predict(X_test)

# Calculate evaluation metrics
# mse = mean_squared_error(y_test, y_pred)
# r_squared = r2_score(y_test, y_pred)

# Print evaluation results
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r_squared}")
