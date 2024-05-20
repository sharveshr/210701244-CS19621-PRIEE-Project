import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('lawyer_recommendation_dataset.csv', encoding="ISO-8859-1")

# List all unique locations
unique_locations = data['Location'].unique()
print("Unique Locations:")
print(unique_locations)

# List all unique expertise
unique_expertise = data['Expertise'].unique()
print("\nUnique Expertise:")
print(unique_expertise)

# Encode categorical variables
label_encoder_location = LabelEncoder()
label_encoder_crime_branch = LabelEncoder()
data['location'] = label_encoder_location.fit_transform(data['Location'])
data['crime_branch'] = label_encoder_crime_branch.fit_transform(data['Expertise'])

# Split data into features and target
X = data[['location', 'crime_branch']]
y = data['Fees']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'lawyer_price_model.joblib')
print("Model saved to 'lawyer_price_model.joblib'")

# Load the model
model = joblib.load('lawyer_price_model.joblib')
print("Model loaded from 'lawyer_price_model.joblib'")

# Predict prices on the test set and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

def predict_price(location, crime_branch):
    location_encoded = label_encoder_location.transform([location])[0]
    crime_branch_encoded = label_encoder_crime_branch.transform([crime_branch])[0]
    features = pd.DataFrame([[location_encoded, crime_branch_encoded]], columns=['location', 'crime_branch'])
    predicted_price = model.predict(features)[0]
    return predicted_price

def recommend_lawyers(predicted_price, location, crime_branch, price_range=10000):
    min_price = predicted_price - price_range
    max_price = predicted_price + price_range
    recommended_lawyers = data[(data['Fees'] >= min_price) & (data['Fees'] <= max_price) & (data['Location'] == location) & (data['Expertise'] == crime_branch)]
    recommended_lawyers = recommended_lawyers.sort_values(by='Rating', ascending=False)
    return recommended_lawyers[['Lawyer Name', 'Location', 'Expertise', 'Fees', 'Rating']]

# Example usage
location = 'Mysore'
crime_branch = 'Criminal Defense Law'
predicted_price = predict_price(location, crime_branch)
print(f'Predicted Price: ₹{predicted_price:.2f}')

recommended_lawyers = recommend_lawyers(predicted_price, location, crime_branch)
print('Recommended Lawyers:')
print(recommended_lawyers)

# Plot graphs to understand the model
plt.figure(figsize=(12, 6))

# Plot true vs predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True Fees')
plt.ylabel('Predicted Fees')
plt.title('True vs Predicted Fees')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line

# Plot residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.xlabel('Predicted Fees')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Fees')
plt.axhline(y=0, color='red', linestyle='--')

plt.tight_layout()
plt.show()
