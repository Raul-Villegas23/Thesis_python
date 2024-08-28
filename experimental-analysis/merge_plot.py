import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from the CSV file
csv_file = 'merged_nasa_tlx_with_time_scores.csv'
data = pd.read_csv(csv_file)

# Calculate performance based on time scores
# Normalize time scores so that the lowest time gets 100% and the highest gets 0%
# Calculate performance based on time scores
min_time = data['Time Score'].min()
max_time = data['Time Score'].max()
data['Driving Performance'] = 100 * (max_time - data['Time Score']) / (max_time - min_time)

# Group the data by 'Delay' and calculate the mean Driving Performance for each delay
grouped_data = data.groupby('Delay')['Driving Performance'].mean().reset_index()

# Set the style
sns.set(style="whitegrid")

# Create a figure for the area plot
plt.figure(figsize=(14, 8))

# Create the area plot
sns.lineplot(x='Delay', y='Driving Performance', data=grouped_data, color='blue', linewidth=2.5)
plt.fill_between(grouped_data['Delay'], grouped_data['Driving Performance'], color='skyblue', alpha=0.4)
sns.regplot(x='Delay', y='Driving Performance', data=grouped_data, scatter=False, color='blue', line_kws={"linewidth": 2, "linestyle": "--"})
# Customize the plot
plt.title('Average Driving Performance vs. Time Delays', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('Average Driving Performance (%)', fontsize=16)
plt.ylim(0, 100)  # Ensures the y-axis goes from 0 to 100
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Add custom styling
plt.gca().patch.set_facecolor('#f7f7f7')  # Set background color
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Tight layout
plt.tight_layout()

# Show the plot
plt.show()

# Now, create a single figure that compares each NASA TLX dimension score vs time delays
dimensions = [
    "Mental Demand", 
    "Physical Demand", 
    "Temporal Demand", 
    "Performance", 
    "Effort", 
    "Frustration"
]

# Create a new figure for the dimension comparison
plt.figure(figsize=(14, 8))

# Plot each dimension with both line and scatter
palette = sns.color_palette("husl", len(dimensions))
for i, dimension in enumerate(dimensions):
    sns.lineplot(x='Delay', y=dimension, data=data, marker='o', color=palette[i], label=dimension, linewidth=2.5)
    sns.scatterplot(x='Delay', y=dimension, data=data, color=palette[i], s=100, edgecolor='black')
    sns.regplot(x='Delay', y=dimension, data=data, scatter=False, color=palette[i], line_kws={"linewidth": 2, "linestyle": "--"})

# Customize the plot
plt.title('Comparison of NASA TLX Dimension Scores vs Time Delays', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('NASA TLX Score', fontsize=16)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(title='NASA TLX Dimensions', fontsize=12, title_fontsize=14, loc='upper right', frameon=True, framealpha=0.9, shadow=True)

# Add a custom background color to the plot
plt.gca().patch.set_facecolor('#f0f0f0')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Now add an area plot for the mean of each dimension across time delays
# Group the data by 'Delay' and calculate the mean for each dimension
grouped_data_dimensions = data.groupby('Delay')[dimensions].mean().reset_index()

# Create a new figure for the area plot of dimensions
plt.figure(figsize=(14, 8))

# Plot each dimension with a line and fill (area plot)
for i, dimension in enumerate(dimensions):
    sns.lineplot(x='Delay', y=dimension, data=grouped_data_dimensions, color=palette[i], label=dimension, linewidth=2.5)
    plt.fill_between(grouped_data_dimensions['Delay'], grouped_data_dimensions[dimension], color=palette[i], alpha=0.2)
    sns.regplot(x='Delay', y=dimension, data=grouped_data_dimensions, scatter=False, color=palette[i], line_kws={"linewidth": 2, "linestyle": "--"})

# Customize the plot
plt.title('Mean NASA TLX Dimension Scores vs Time Delays', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('Mean NASA TLX Score', fontsize=16)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(title='NASA TLX Dimensions', fontsize=12, title_fontsize=14, loc='upper right', frameon=True, framealpha=0.9, shadow=True)

# Add a custom background color to the plot
plt.gca().patch.set_facecolor('#f0f0f0')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['Delay', 'Driving Performance'] + dimensions].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of Time Delays, Performance, and NASA TLX Dimensions', fontsize=18, fontweight='bold')
plt.show()

# 2. Pair Plot for deeper insights into relationships between variables
# sns.pairplot(data[['Delay', 'Driving Performance'] + dimensions], kind='reg', diag_kind='kde')
# plt.suptitle('Pair Plot of Time Delays, Performance, and NASA TLX Dimensions', fontsize=18, fontweight='bold', y=1.02)
# plt.show()

# 3. Box Plots for distribution and density comparison
plt.figure(figsize=(14, 10))
for i, dimension in enumerate(dimensions):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Delay', y=dimension, data=data)
    plt.title(f'{dimension} Distribution Across Time Delays')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# 4. Distribution Plots for Performance and Mental Demand
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['Driving Performance'], kde=True, color='blue', bins=20)
plt.title('Distribution of Driving Performance', fontsize=16, fontweight='bold')
plt.xlabel('Driving Performance (%)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(data['Mental Demand'], kde=True, color='orange', bins=20)
plt.title('Distribution of Mental Demand', fontsize=16, fontweight='bold')
plt.xlabel('Mental Demand Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

