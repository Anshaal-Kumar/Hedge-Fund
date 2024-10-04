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

# If you have a test dataset for evaluation, load it
# Assuming you have the actual values to compare with the predictions
# For example, if you have actual values corresponding to your new data
actual_values = [1.1, 1.05, 1.02]  # Replace this with your actual values

# Calculate the R-squared value
r_squared = r2_score(actual_values, predictions)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(actual_values, predictions)

# Convert R-squared to percentage accuracy
percentage_accuracy = r_squared * 100

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")
print(f"Percentage Accuracy: {percentage_accuracy:.2f}%")
