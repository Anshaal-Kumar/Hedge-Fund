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

