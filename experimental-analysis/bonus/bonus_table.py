import csv
import random
import itertools

# Define the participants, velocities (m/s), and delay magnitudes (ms)
participants = list(range(1, 9))  # 8 participants
velocities = [2.2, 3.6, 5.0, 6.0]  # 4 velocities in m/s
delays = [0, 250, 500, 1000, 2000]  # 5 delays in ms

# Create all possible combinations of velocities and delays
conditions = list(itertools.product(velocities, delays))

# Number of runs per participant (3 runs per participant)
runs_per_participant = 3
total_runs_needed = len(participants) * runs_per_participant

# Shuffle the conditions to ensure randomization
random.shuffle(conditions)

# Ensure that all conditions are tested at least once before any repetition
experiment_data = []

# Step 1: Assign each unique combination to participants
for participant in participants:
    for run in range(1, runs_per_participant + 1):
        if conditions:
            # Pop the next condition from the shuffled list
            condition = conditions.pop(0)
        else:
            # Re-shuffle the conditions when they run out
            conditions = list(itertools.product(velocities, delays))
            random.shuffle(conditions)
            condition = conditions.pop(0)
        
        # Append the participant, run number, velocity, and delay to the experiment data
        experiment_data.append([participant, run, condition[0], condition[1]])

# Define the CSV file name
csv_filename = "experiment_conditions.csv"

# Write the data to a CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["Participant", "Run", "Velocity(m/s)", "Delay(ms)"])
    
    # Write the randomized conditions for each participant
    writer.writerows(experiment_data)

print(f"Experiment conditions CSV file '{csv_filename}' has been created successfully.")
