import pandas as pd

# Load the CSV file
df = pd.read_csv('vehicles.csv')

# Define the list of columns you want to keep
columns_to_keep = ['price', 'year', 'manufacturer', 'model', 'odometer', 'transmission']

# Keep only these columns
df = df[columns_to_keep]

# Remove all rows where at least one item is missing
df = df.dropna()

# Save the modified DataFrame back to a new CSV file
df.to_csv('vehicles-new.csv', index=False)
