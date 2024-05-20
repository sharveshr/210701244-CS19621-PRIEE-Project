import csv
from faker import Faker
import random

fake = Faker('en_IN')

def generate_lawyer():
    name = fake.name()
    expertise = random.choice(["Personal Injury Law", "Family Law", "Business Law", "Criminal Defense Law", "Real Estate Law", "Immigration Law", "Intellectual Property Law", "Employment Law", "Environmental Law", "Medical Malpractice Law"])
    location = fake.city_name()
    rating = round(random.uniform(3.5, 5.0), 1)
    fees = f"{random.randint(20000, 50000)}"
    return [name, expertise, location, rating, fees]

def generate_dataset(num_lawyers):
    dataset = [["Lawyer Name", "Expertise", "Location", "Rating", "Fees"]]
    for _ in range(num_lawyers):
        dataset.append(generate_lawyer())
    return dataset

def save_dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset)

if __name__ == "__main__":
    num_lawyers = 10000
    dataset = generate_dataset(num_lawyers)
    save_dataset_to_csv(dataset, "lawyer_recommendation_dataset.csv")
