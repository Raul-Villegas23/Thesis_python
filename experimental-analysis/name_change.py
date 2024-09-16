import pandas as pd

# Load the CSV file into a DataFrame
def load_data(csv_file):
    return pd.read_csv(csv_file)

# Function to replace names with participant numbers
def replace_names_with_participant_numbers(df):
    # Get unique names
    unique_names = df['name'].unique()

    # Initialize participant number counter
    participant_number = 1
    
    # Loop over each unique name and replace the name with participant numbers (3 times each)
    for name in unique_names:
        # Filter the DataFrame for rows with the current name and assign participant numbers
        name_rows = df[df['name'] == name].index
        df.loc[name_rows, 'name'] = [participant_number] * 3
        participant_number += 1  # Increment participant number for next name

    return df

# Save the updated DataFrame back to a CSV file
def save_data(df, output_file):
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Load your CSV file
    csv_file = 'merged_nasa_tlx_with_maze_results.csv'
    df = load_data(csv_file)

    # Replace names with participant numbers
    df_updated = replace_names_with_participant_numbers(df)

    # Save the updated DataFrame to a new CSV
    save_data(df_updated, 'updated.csv')

    # Print the first few rows to verify
    print(df_updated.head(10))
