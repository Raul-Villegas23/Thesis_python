import csv
import random
import itertools

# Define the participants, velocities (m/s), and delay magnitudes (ms)
participants = list(range(1, 9))  # 8 participants, just numbers now
velocities = [2.2, 3.6, 5.0, 6.0]  # 4 velocities in m/s
delays = [0, 250, 500, 1000, 2000]  # 5 delays in ms

# Create all possible combinations of velocities and delays
conditions = list(itertools.product(velocities, delays))

# Number of runs per participant
runs_per_participant = 3
total_runs_needed = len(participants) * runs_per_participant

# Ensure the total number of runs exceeds the unique conditions
if total_runs_needed > len(conditions):
    print(f"Warning: Not enough unique conditions ({len(conditions)}) for {total_runs_needed} total runs. Some conditions will be reused.")

# Shuffle the conditions to ensure randomization
random.shuffle(conditions)

# Assign conditions to each participant, allowing repetition across participants but ensuring no duplicates within each participant
experiment_data = []

for participant in participants:
    participant_conditions = []
    
    # Randomly select conditions for this participant, ensuring no duplicates for the same participant
    available_conditions = conditions.copy()  # Copy the full condition list
    for _ in range(runs_per_participant):
        condition = random.choice(available_conditions)
        
        # Remove the chosen condition from the available list for this participant to avoid repetition
        available_conditions.remove(condition)
        
        participant_conditions.append(condition)
        experiment_data.append([participant, len(participant_conditions), condition[0], condition[1]])

# Define the CSV file name
csv_filename = "experiment_conditions.csv"

# Write the data to a CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["Participant", "Run", "Velocity(m/s)", "Delay(ms)"])
    
    # Write the randomized conditions for each participant
    for row in experiment_data:
        writer.writerow(row)

print(f"Experiment conditions CSV file '{csv_filename}' has been created successfully.")
