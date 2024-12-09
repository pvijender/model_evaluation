import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Load and prepare main training data
df_train = pd.read_csv('Car Price Data for evaluating models.csv')
X = df_train.drop(['car_ID', 'price', 'CarName'], axis=1)
y = df_train['price']

# Load the cars to predict
cars_to_predict = pd.read_csv('Car Price Data for evaluating models_11records.csv')
X_predict = cars_to_predict.drop(['car_ID', 'price', 'CarName'], axis=1)
actual_prices = cars_to_predict['price']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Train models and make predictions
predictions = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit model on full training data
    pipeline.fit(X, y)
    
    # Make predictions
    preds = pipeline.predict(X_predict)
    predictions[name] = preds

# Create results DataFrame
results = pd.DataFrame({
    'CarName': cars_to_predict['CarName'],
    'Actual Price': actual_prices
})

# Add predictions from each model
for name, preds in predictions.items():
    results[name] = preds

# Calculate error metrics for each model
error_metrics = []
for name in models.keys():
    mae = np.mean(np.abs(results['Actual Price'] - results[name]))
    mape = np.mean(np.abs((results['Actual Price'] - results[name]) / results['Actual Price'])) * 100
    r2 = r2_score(results['Actual Price'], results[name])
    error_metrics.append({
        'Model': name,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    })

error_df = pd.DataFrame(error_metrics)

# Format results for display
pd.set_option('display.float_format', lambda x: '${:,.2f}'.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else '{:.4f}'.format(x) if isinstance(x, float) else str(x))

print("\nPredictions for each car:")
print(results.to_string(index=False))
print("\nError Metrics:")
print(error_df.to_string(index=False))

# Save results to CSV
results.to_csv('car_price_predictions.csv', index=False)
error_df.to_csv('prediction_errors.csv', index=False)