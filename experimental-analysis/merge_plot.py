import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
# Load the data from the CSV file
csv_file = 'merged_nasa_tlx_with_maze_results.csv'
data = pd.read_csv(csv_file)

# Convert Maze score from percentage string to float
data['Maze score'] = data['Maze score'].str.rstrip('%').astype(float)

# Define function to plot line and fill with regression line and weights
def plot_with_regression(x, y, data, color, xlabel, ylabel, title, weights=None):
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=x, y=y, data=data, color=color, linewidth=2.5)
    plt.fill_between(data[x], data[y], color=color, alpha=0.4)
    sns.regplot(x=x, y=y, data=data, scatter=False, color=color, line_kws={"linewidth": 2, "linestyle": "--"})
    
    # Add weights as text annotations
    if weights:
        weight_text = "\n".join([f"{dim}: {wt*100:.1f}%" for dim, wt in weights.items()])
        plt.text(0.98, 0.02, weight_text, fontsize=12, color='black', ha='right', va='bottom', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.gca().patch.set_facecolor('#f7f7f7')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot Maze Score vs. Time Delays
grouped_maze_data = data.groupby('Delays')['Maze score'].mean().reset_index()
plot_with_regression(
    x='Delays', y='Maze score', data=grouped_maze_data,
    color='orange', xlabel='Time Delay (ms)', ylabel='Maze Score (%)',
    title='Maze Score vs. Time Delays'
)

# Assign weights to each NASA TLX dimension and calculate weighted scores
weights = {
    "Mental Demand": 0.25,
    "Physical Demand": 0.0,
    "Temporal Demand": 0.25,
    "Performance": 0.00,
    "Effort": 0.25,
    "Frustration": 0.25
}
weights = {k: v / sum(weights.values()) for k, v in weights.items()}

for dimension, weight in weights.items():
    data[f'Weighted {dimension}'] = data[dimension] * weight

data['Weighted NASA TLX Score'] = data[[f'Weighted {dim}' for dim in weights.keys()]].sum(axis=1)

# Calculate performance based on time scores and overall performance
min_time, max_time = data['Time Scores'].min(), data['Time Scores'].max()
data['Driving Performance'] = 100 * (max_time - data['Time Scores']) / (max_time - min_time)
data['Overall Performance'] = (data['Driving Performance'] + data['Maze score']) / 2

# Plot Driving Performance vs. Time Delays
grouped_driving_data = data.groupby('Delays')['Driving Performance'].mean().reset_index()
plot_with_regression(
    x='Delays', y='Driving Performance', data=grouped_driving_data,
    color='blue', xlabel='Time Delay (ms)', ylabel='Driving Performance (%)',
    title='Driving Performance vs. Time Delays'
)

# Plot Overall Performance vs. Time Delays
grouped_data = data.groupby('Delays').agg({
    'Driving Performance': 'mean',
    'Maze score': 'mean',
    'Overall Performance': 'mean'
}).reset_index()
plot_with_regression(
    x='Delays', y='Overall Performance', data=grouped_data,
    color='green', xlabel='Time Delay (ms)', ylabel='Overall Performance (%)',
    title='Overall Performance (Driving + Maze) vs. Time Delays'
)


# Comparison of NASA TLX Dimension Scores vs Time Delays
dimensions = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]
palette = sns.color_palette("husl", len(dimensions))

plt.figure(figsize=(14, 8))
for i, dimension in enumerate(dimensions):
    sns.lineplot(x='Delays', y=dimension, data=data, marker='o', color=palette[i], label=dimension, linewidth=2.5)
    sns.scatterplot(x='Delays', y=dimension, data=data, color=palette[i], s=100, edgecolor='black')
    sns.regplot(x='Delays', y=dimension, data=data, scatter=False, color=palette[i], line_kws={"linewidth": 2, "linestyle": "--"})
plt.title('Comparison of NASA TLX Dimension Scores vs Time Delays', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('NASA TLX Score', fontsize=16)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(title='NASA TLX Dimensions', fontsize=12, title_fontsize=14, loc='upper right', frameon=True, framealpha=0.9, shadow=True)
plt.gca().patch.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()

# Area plot for mean NASA TLX dimension scores across time delays
grouped_data_dimensions = data.groupby('Delays')[dimensions].mean().reset_index()
plt.figure(figsize=(14, 8))
for i, dimension in enumerate(dimensions):
    sns.lineplot(x='Delays', y=dimension, data=grouped_data_dimensions, color=palette[i], label=dimension, linewidth=2.5)
    plt.fill_between(grouped_data_dimensions['Delays'], grouped_data_dimensions[dimension], color=palette[i], alpha=0.2)
    sns.regplot(x='Delays', y=dimension, data=grouped_data_dimensions, scatter=False, color=palette[i], line_kws={"linewidth": 2, "linestyle": "--"})
plt.title('Mean NASA TLX Dimension Scores vs Time Delays', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('Mean NASA TLX Score', fontsize=16)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(title='NASA TLX Dimensions', fontsize=12, title_fontsize=14, loc='upper right', frameon=True, framealpha=0.9, shadow=True)
plt.gca().patch.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['Delays', 'Driving Performance', 'Maze score', 'Overall Performance', 'Weighted NASA TLX Score'] + dimensions].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of Time Delays, Performance, Maze, and NASA TLX Dimensions', fontsize=18, fontweight='bold')
plt.show()

# Box plots for distribution and density comparison
plt.figure(figsize=(14, 10))
for i, dimension in enumerate(dimensions):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Delays', y=dimension, data=data)
    plt.title(f'{dimension} Distribution Across Time Delays')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot unweighted NASA TLX Score vs. Time Delays
grouped_data = data.groupby('Delays')['Overall Score'].mean().reset_index()
plot_with_regression(
    x='Delays', y='Overall Score', data=grouped_data,
    color='red', xlabel='Time Delay (ms)', ylabel='NASA TLX Score',
    title='NASA TLX Score vs. Time Delays'
)

# Plot Weighted NASA TLX Score vs. Time Delays
grouped_weighted_data = data.groupby('Delays')['Weighted NASA TLX Score'].mean().reset_index()
plot_with_regression(
    x='Delays', y='Weighted NASA TLX Score', data=grouped_weighted_data,
    color='purple', xlabel='Time Delay (ms)', ylabel='Weighted NASA TLX Score',
    title='Weighted NASA TLX Score vs. Time Delays', weights=weights
)

# Fit regression models to estimate critical delay
def find_critical_delay(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    predictions = model.predict(x.reshape(-1, 1))
    residuals = y - predictions
    critical_delay = x[np.argmin(residuals)]  # Delay with the highest residual (performance drop)
    return critical_delay

delays = data['Delays'].values
overall_performance = data['Overall Performance'].values
critical_delay = find_critical_delay(delays, overall_performance)

# Plot performance metrics with critical delay threshold
plt.figure(figsize=(14, 8))

# Plot Driving Performance vs. Time Delays
sns.lineplot(x='Delays', y='Driving Performance', data=data, color='blue', label='Driving Performance', linewidth=2.5)
sns.scatterplot(x='Delays', y='Driving Performance', data=data, color='blue', s=100, edgecolor='black')

# Plot Maze Score vs. Time Delays
sns.lineplot(x='Delays', y='Maze score', data=data, color='orange', label='Maze Score', linewidth=2.5)
sns.scatterplot(x='Delays', y='Maze score', data=data, color='orange', s=100, edgecolor='black')

# Plot Overall Performance vs. Time Delays
sns.lineplot(x='Delays', y='Overall Performance', data=data, color='green', label='Overall Performance', linewidth=2.5)
sns.scatterplot(x='Delays', y='Overall Performance', data=data, color='green', s=100, edgecolor='black')

# Plot NASA TLX Score vs. Time Delays
sns.lineplot(x='Delays', y='Overall Score', data=data, color='red', label='NASA TLX Score', linewidth=2.5)
sns.scatterplot(x='Delays', y='Overall Score', data=data, color='red', s=100, edgecolor='black')

# Add critical delay line
plt.axvline(x=critical_delay, color='purple', linestyle='--', linewidth=2, label=f'Estimated Critical Delay ({int(critical_delay)} ms)')

plt.title('Performance Metrics vs. Time Delays with Critical Delay Threshold', fontsize=20, fontweight='bold')
plt.xlabel('Time Delay (ms)', fontsize=16)
plt.ylabel('Performance (%)', fontsize=16)
plt.ylim(0, 100)
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(title='Metrics', fontsize=12, title_fontsize=14)
plt.gca().patch.set_facecolor('#f7f7f7')
plt.tight_layout()
plt.show()