import csv
import random

# Define the range for random time samples in seconds (e.g., time for task completion)
time_range = (20, 120)  # Example range: between 20 and 120 seconds

# Read the existing CSV file
csv_filename = "experiment_conditions.csv"
new_csv_filename = "experiment_time_trials_with_time.csv"

# Read the existing CSV file and add a new column for Time(s)
experiment_data = []
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header
    experiment_data = [row for row in reader]

# Add a new "Time(s)" column with random values
header.append("Time(s)")
for row in experiment_data:
    random_time = random.uniform(time_range[0], time_range[1])  # Generate random time within the range
    row.append(f"{random_time:.2f}")  # Append random time, formatted to 2 decimal places

# Write the updated data to a new CSV file
with open(new_csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the new header with the added "Time(s)" column
    writer.writerow(header)
    
    # Write the updated rows with time samples
    for row in experiment_data:
        writer.writerow(row)

print(f"CSV file '{new_csv_filename}' has been created successfully with random time samples.")
