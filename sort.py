import pandas as pd

# Load the CSV file
df = pd.read_csv('vehicles-new.csv')

# List of columns to process
columns = ['manufacturer', 'model', 'transmission']

for column in columns:
    # Get unique values of the column
    unique_values = pd.DataFrame(df[column].unique())

    # Write unique values to a new CSV file
    unique_values.to_csv(f'options/{column}_unique.csv', index=False, header=False)
