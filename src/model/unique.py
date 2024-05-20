import pandas as pd

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
