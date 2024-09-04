import pandas as pd

# Load the participants info triplicated data
participants_info_triplicated_file = 'participants_info_triplicated.csv'
participants_info_triplicated_data = pd.read_csv(participants_info_triplicated_file)

# Load the NASA TLX data
nasa_tlx_file = 'nasa_tlx_data.csv'
nasa_tlx_data = pd.read_csv(nasa_tlx_file)

# Load the participants delays data
participants_delays_file = 'participants_delays.csv'
participants_delays_data = pd.read_csv(participants_delays_file)
# Ignore the first column of the CSV file
participants_delays_data = participants_delays_data.iloc[:, 1:]

# Load the participants time scores data
participants_time_scores_file = 'participants_times.csv'
participants_time_scores_data = pd.read_csv(participants_time_scores_file)

# Load the maze results data
maze_results_file = 'maze_results.csv'
maze_results_data = pd.read_csv(maze_results_file)
# Ignore the first column of the CSV file
maze_results_data = maze_results_data.iloc[:, 1:]

# Add an index to each DataFrame based on the row number
participants_info_triplicated_data['Row Index'] = participants_info_triplicated_data.index
nasa_tlx_data['Row Index'] = nasa_tlx_data.index
participants_delays_data['Row Index'] = participants_delays_data.index
participants_time_scores_data['Row Index'] = participants_time_scores_data.index
maze_results_data['Row Index'] = maze_results_data.index

# Merge the participants info data with the NASA TLX data
merged_data = pd.merge(participants_info_triplicated_data, nasa_tlx_data, how='left', on='Row Index')

# Merge the resulting data with participants delays data
merged_data = pd.merge(merged_data, participants_delays_data, how='left', on='Row Index', suffixes=('', '_delays'))

# Merge the resulting data with participants time scores data
merged_data = pd.merge(merged_data, participants_time_scores_data, how='left', on='Row Index', suffixes=('', '_times'))

# Merge the resulting data with maze results data
merged_data = pd.merge(merged_data, maze_results_data, how='left', on='Row Index', suffixes=('', '_maze'))

# Drop the Row Index column
merged_data.drop(columns=['Row Index'], inplace=True)

# Display the merged dataframe to verify the result
print("\nMerged Data:")
print(merged_data.head())

# Save the merged data to a new CSV file
merged_file = 'merged_nasa_tlx_with_maze_results.csv'
merged_data.to_csv(merged_file, index=False)
