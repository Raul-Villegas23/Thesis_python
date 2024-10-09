import pandas as pd

# Load each CSV file
experiment_df = pd.read_csv('experiment_time_trials_with_time.csv')
maze_df = pd.read_csv('maze_results.csv')
nasa_tlx_df = pd.read_csv('nasa_tlx_results.csv')

# Rename 'Participant Number' to 'Participant' in nasa_tlx_df for consistency
nasa_tlx_df = nasa_tlx_df.rename(columns={'Participant Number': 'Participant'})

# Concatenate all DataFrames, ignoring missing columns
merged_df = pd.concat([experiment_df, maze_df, nasa_tlx_df], axis=1)

# Save the concatenated DataFrame to a new CSV file
merged_df.to_csv('complete.csv', index=False)

print("CSV files successfully merged into 'complete.csv'")
