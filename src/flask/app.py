from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load data and model
data = pd.read_csv('lawyer_recommendation_dataset.csv', encoding="ISO-8859-1")
model = joblib.load('lawyer_price_model.joblib')

# Encode categorical variables
label_encoder_location = LabelEncoder()
label_encoder_crime_branch = LabelEncoder()
data['location'] = label_encoder_location.fit_transform(data['Location'])
data['crime_branch'] = label_encoder_crime_branch.fit_transform(data['Expertise'])

# Get unique locations and expertise
unique_locations = sorted(data['Location'].unique())
unique_expertise = sorted(data['Expertise'].unique())

def predict_price(location, crime_branch):
    location_encoded = label_encoder_location.transform([location])[0]
    crime_branch_encoded = label_encoder_crime_branch.transform([crime_branch])[0]
    features = pd.DataFrame([[location_encoded, crime_branch_encoded]], columns=['location', 'crime_branch'])
    predicted_price = model.predict(features)[0]
    return predicted_price

def recommend_lawyers(predicted_price, location, crime_branch, price_range=10000):
    min_price = predicted_price - price_range
    max_price = predicted_price + price_range
    recommended_lawyers = data[(data['Fees'] >= min_price) & (data['Fees'] <= max_price) &
                               (data['Location'] == location) & (data['Expertise'] == crime_branch)]
    recommended_lawyers = recommended_lawyers.sort_values(by='Rating', ascending=False)
    return recommended_lawyers[['Lawyer Name', 'Location', 'Expertise', 'Fees', 'Rating']]

@app.route('/')
def index():
    return render_template('index.html', locations=unique_locations, expertise=unique_expertise)

@app.route('/result', methods=['POST'])
def result():
    location = request.form['location']
    crime_branch = request.form['crime_branch']
    predicted_price = predict_price(location, crime_branch)
    recommended_lawyers = recommend_lawyers(predicted_price, location, crime_branch)
    return render_template('result.html', predicted_price=predicted_price, lawyers=recommended_lawyers.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
