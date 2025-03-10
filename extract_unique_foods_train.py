import pandas as pd

# Load the Excel file
df = pd.read_excel('annotations_train.xlsx')

# Extract unique food items
unique_foods = df['food_name'].unique()

# Convert unique food items to a DataFrame
unique_foods_df = pd.DataFrame(unique_foods, columns=['food_name'])



# Display the number of unique food items and the first few rows of the DataFrame
num_unique_foods = len(unique_foods)
print(f"Number of unique food items: {num_unique_foods}")
print("Unique food items in the training dataset:")
print(unique_foods_df)