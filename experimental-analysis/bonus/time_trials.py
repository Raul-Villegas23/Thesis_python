import csv

# Define input and output file names
input_csv_filename = "experiment_conditions.csv"  # The existing CSV with velocities and delays
output_csv_filename = "experiment_time_trials_with_time.csv"  # Final output CSV with time included

# Read the input CSV with the pre-defined conditions
def read_conditions(input_file):
    conditions = []
    with open(input_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            conditions.append(row)
    return conditions, reader.fieldnames

# Automatically map the necessary columns based on common names
def auto_map_columns(headers):
    # Define keywords for each type of column
    keywords = {
        'participant': ['participant', 'id', 'subject', 'participant number'],
        'run': ['run', 'trial'],
        'velocity': ['velocity', 'speed'],
        'delay': ['delay', 'latency']
    }

    # Function to match columns based on keywords
    def find_column(header_list, possible_keywords):
        for header in header_list:
            for keyword in possible_keywords:
                if keyword.lower() in header.lower():  # Case insensitive matching
                    return header
        return None  # Return None if no match is found

    # Automatically map each column
    column_mapping = {
        'participant': find_column(headers, keywords['participant']),
        'run': find_column(headers, keywords['run']),
        'velocity': find_column(headers, keywords['velocity']),
        'delay': find_column(headers, keywords['delay'])
    }

    # Ensure all columns were found, raise an error if not
    if None in column_mapping.values():
        missing_columns = [key for key, value in column_mapping.items() if value is None]
        raise ValueError(f"Missing columns for: {', '.join(missing_columns)}")

    print("Automatically mapped columns: ", column_mapping)
    return column_mapping

# Add time for each trial based on the conditions
def record_trial_times(conditions, column_mapping):
    for condition in conditions:
        participant = condition[column_mapping['participant']]
        run = condition[column_mapping['run']]
        velocity = condition[column_mapping['velocity']]
        delay = condition[column_mapping['delay']]
        
        # Input time manually
        total_time = input(f"Enter time taken (in seconds) for {participant}, Run {run}, Velocity {velocity} m/s, Delay {delay} ms: ")
        condition["Time(s)"] = total_time
    return conditions

# Write the final CSV with time included
def write_results(conditions, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=conditions[0].keys())
        writer.writeheader()
        writer.writerows(conditions)

def main():
    print("Experiment Time Trial Recorder")

    # Step 1: Read the predefined conditions from the CSV file
    conditions, headers = read_conditions(input_csv_filename)
    
    # Step 2: Automatically detect the column mapping for participant, run, velocity, and delay
    column_mapping = auto_map_columns(headers)
    
    # Step 3: Record the time for each trial
    conditions_with_times = record_trial_times(conditions, column_mapping)
    
    # Step 4: Write the final results (including time) to a new CSV
    write_results(conditions_with_times, output_csv_filename)
    print(f"Time trials recorded successfully and saved to {output_csv_filename}.")

if __name__ == "__main__":
    main()
