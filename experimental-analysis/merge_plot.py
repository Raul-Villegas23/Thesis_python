import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from scipy.interpolate import griddata
import statsmodels.api as sm 

# Load and preprocess data
def load_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns={
        'Maze score': 'Maze Score (%)',
        'Time Scores': 'Time Scores (ms)'
    }, inplace=True)
    data['Maze Score (%)'] = data['Maze Score (%)'].str.rstrip('%').astype(float)
    return data

# Calculate performance based on time scores and overall performance
def calculate_performance_metrics(data):
    min_time, max_time = data['Time Scores (ms)'].min(), data['Time Scores (ms)'].max()
    data['Driving Performance'] = 100 * (max_time - data['Time Scores (ms)']) / (max_time - min_time)
    data['Overall Performance'] = (data['Driving Performance'] + data['Maze Score (%)']) / 2
    return data

# Plot with regression line and optional weights
def plot_with_regression(x, y, data, color, xlabel, ylabel, title, weights=None):
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=x, y=y, data=data, color=color, linewidth=2.5)
    plt.fill_between(data[x], data[y], color=color, alpha=0.4)
    sns.regplot(x=x, y=y, data=data, scatter=False, color=color, line_kws={"linewidth": 2, "linestyle": "--"})
    
    if weights:
        weight_text = "\n".join([f"{dim}: {wt*100:.1f}%" for dim, wt in weights.items()])
        plt.text(0.98, 0.02, weight_text, fontsize=12, color='black', ha='right', va='bottom',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
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

# Calculate weighted NASA TLX scores
def calculate_weighted_scores(data, weights):
    for dimension, weight in weights.items():
        data[f'Weighted {dimension}'] = data[dimension] * weight
    data['Weighted NASA TLX Score'] = data[[f'Weighted {dim}' for dim in weights.keys()]].sum(axis=1)
    return data
# Plot 3D surface
def plot_3d_surface(data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid data for surface plot
    X = data['Delays'].values
    Y = data['Driving Performance'].values
    Z = data['Maze Score (%)'].values

    # Create a meshgrid for X and Y
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    
    # Interpolate Z values onto the grid
    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method='cubic')
    
    # Plot the surface
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Delays (ms)', fontsize=15, fontweight='bold', labelpad=15)
    ax.set_ylabel('Driving Performance (%)', fontsize=15, fontweight='bold', labelpad=15)
    ax.set_zlabel('Maze Score (%)', fontsize=15, fontweight='bold', labelpad=15)
    ax.set_title('3D Surface Plot of Delays, Driving Performance, and Maze Score', fontsize=18, fontweight='bold', pad=20)

    # Add a color bar
    cbar = plt.colorbar(surf, pad=0.1)
    cbar.set_label('Maze Score (%)', fontsize=13, fontweight='bold')
    
    plt.show()

def calculate_and_plot_learning_effect(data):
    data = data.sort_values(by=['Participant Number'])
    data['Trial'] = data.groupby('name').cumcount() + 1

    # Calculate mean performance metrics across trials
    mean_performance = data.groupby('Trial').agg({
        'Driving Performance': 'mean',
        'Maze Score (%)': 'mean',
        'Overall Performance': 'mean',
        'Weighted NASA TLX Score': 'mean'  # Use Weighted NASA TLX Score instead of Overall Score
    }).reset_index()

    # Calculate learning effects
    learning_effect_performance = (mean_performance.iloc[2, 1:4] - mean_performance.iloc[0, 1:4]) / mean_performance.iloc[0, 1:4] * 100
    learning_effect_tlx = (mean_performance.iloc[2, 4] - mean_performance.iloc[0, 4]) / mean_performance.iloc[0, 4] * 100

    plt.figure(figsize=(14, 8))

    # Plot the performance metrics with trendlines
    sns.lineplot(x='Trial', y='Driving Performance', data=mean_performance, marker='o', color='blue', label='Driving Performance', linewidth=2.5)
    sns.regplot(x='Trial', y='Driving Performance', data=mean_performance, scatter=False, color='blue', line_kws={"linestyle": "--", "linewidth": 2})

    sns.lineplot(x='Trial', y='Maze Score (%)', data=mean_performance, marker='o', color='orange', label='Maze Score', linewidth=2.5)
    sns.regplot(x='Trial', y='Maze Score (%)', data=mean_performance, scatter=False, color='orange', line_kws={"linestyle": "--", "linewidth": 2})

    sns.lineplot(x='Trial', y='Overall Performance', data=mean_performance, marker='o', color='green', label='Overall Performance', linewidth=2.5)
    sns.regplot(x='Trial', y='Overall Performance', data=mean_performance, scatter=False, color='green', line_kws={"linestyle": "--", "linewidth": 2})

    # Plot the Weighted NASA TLX Score with trendlines
    sns.lineplot(x='Trial', y='Weighted NASA TLX Score', data=mean_performance, marker='o', color='purple', label='Weighted NASA TLX Score', linewidth=2.5)
    sns.regplot(x='Trial', y='Weighted NASA TLX Score', data=mean_performance, scatter=False, color='purple', line_kws={"linestyle": "--", "linewidth": 2})

    # Confidence intervals
    sns.lineplot(x='Trial', y='Driving Performance', data=data, estimator='mean', ci='sd', color='blue', alpha=0.3)
    sns.lineplot(x='Trial', y='Maze Score (%)', data=data, estimator='mean', ci='sd', color='orange', alpha=0.3)
    sns.lineplot(x='Trial', y='Overall Performance', data=data, estimator='mean', ci='sd', color='green', alpha=0.3)
    sns.lineplot(x='Trial', y='Weighted NASA TLX Score', data=data, estimator='mean', ci='sd', color='purple', alpha=0.3)

    # Annotate the percentage change in Weighted NASA TLX Score
    plt.text(1.5, mean_performance['Weighted NASA TLX Score'].max() + 5, 
             f'Change in Weighted NASA TLX Score: {learning_effect_tlx:.2f}%', 
             fontsize=14, color='purple', fontweight='bold', ha='left')
    
    # Annotate the percentage change in overall performance
    plt.text(1.5, mean_performance['Overall Performance'].max() - 10,
                f'Change in Overall Performance: {learning_effect_performance[2]:.2f}%', 
                fontsize=14, color='green', fontweight='bold', ha='left')
    

    plt.title('Learning Effect Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Performance (%) / Weighted NASA TLX Score', fontsize=18)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(title='Metrics', fontsize=14, title_fontsize=16)
    plt.gca().patch.set_facecolor('#f7f7f7')
    plt.tight_layout()
    plt.show()

    print("Learning Effect (Percentage Improvement from Trial 1 to Trial 3):")
    print(learning_effect_performance.round(2))
    print(f"Change in Weighted NASA TLX Score: {learning_effect_tlx:.2f}%")


# Plot radar chart for different professions
def plot_radar_chart(data, profession):
    df = data[data['profession'] == profession]
    mean_scores = df[["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]].mean()

    labels = mean_scores.index
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    values = mean_scores.tolist()
    values += values[:1]
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.fill(angles, values, color='orange', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(f'Radar Chart of NASA TLX Dimensions for {profession}', size=20, color='black', y=1.1)
    plt.show()

# Plot trend analysis over participant number
def plot_trend_analysis(data):
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Participant Number', y='Overall Performance', data=data, marker='o', color='green', label='Overall Performance', linewidth=2.5)
    sns.lineplot(x='Participant Number', y='Maze Score (%)', data=data, marker='o', color='orange', label='Maze Score', linewidth=2.5)
    sns.lineplot(x='Participant Number', y='Driving Performance', data=data, marker='o', color='blue', label='Driving Performance', linewidth=2.5)

    plt.title('Trend Analysis Over Participant Number', fontsize=20, fontweight='bold')
    plt.xlabel('Participant Number', fontsize=16)
    plt.ylabel('Performance (%)', fontsize=16)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(title='Metrics', fontsize=12, title_fontsize=14)
    plt.gca().patch.set_facecolor('#f7f7f7')
    plt.tight_layout()
    plt.show()

# Comparison of NASA TLX Dimension Scores vs Time Delays
def plot_nasa_tlx_comparison(data):
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
def plot_nasa_tlx_area(data):
    dimensions = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]
    palette = sns.color_palette("husl", len(dimensions))
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
def plot_correlation_heatmap(data):
    dimensions = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[['Delays', 'Driving Performance', 'Maze Score (%)'] + dimensions].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Heatmap of Time Delays, Performance, Maze, and NASA TLX Dimensions', fontsize=18, fontweight='bold')
    plt.show()

# Box plots for distribution and density comparison
def plot_boxplots(data):
    dimensions = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration"]

    plt.figure(figsize=(14, 10))
    for i, dimension in enumerate(dimensions):
        plt.subplot(3, 2, i + 1)
        sns.boxplot(x='Delays', y=dimension, data=data)
        plt.title(f'{dimension} Distribution Across Time Delays')
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Fit regression models to estimate critical delay
def find_critical_delay(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    predictions = model.predict(x.reshape(-1, 1))
    residuals = y - predictions
    critical_delay_index = np.argmin(residuals)  # Delay with the highest residual (performance drop)
    critical_delay = x[critical_delay_index]
    return critical_delay, model.coef_[0], model.intercept_

# Plot performance metrics with critical delay threshold
def plot_with_critical_delay(data):
    delays = data['Delays'].values
    overall_performance = data['Overall Performance'].values
    critical_delay, slope, intercept = find_critical_delay(delays, overall_performance)

    # Display the mathematical formula of the regression line
    formula = f"Performance = {slope:.2f} * Delay + {intercept:.2f}"
    print(f"Critical Delay: {critical_delay} ms")
    print(f"Regression Formula: {formula}")

    plt.figure(figsize=(14, 8))

    # Plot Driving Performance vs. Time Delays
    sns.lineplot(x='Delays', y='Driving Performance', data=data, color='blue', label='Driving Performance', linewidth=2.5)
    sns.regplot(x='Delays', y='Driving Performance', data=data, scatter=False, color='blue', line_kws={"linewidth": 2, "linestyle": "--"})

    # Plot Maze Score vs. Time Delays
    sns.lineplot(x='Delays', y='Maze Score (%)', data=data, color='orange', label='Maze Score', linewidth=2.5)
    sns.regplot(x='Delays', y='Maze Score (%)', data=data, scatter=False, color='orange', line_kws={"linewidth": 2, "linestyle": "--"})

    # Plot Overall Performance vs. Time Delays
    sns.lineplot(x='Delays', y='Overall Performance', data=data, color='green', label='Overall Performance', linewidth=2.5)
    sns.regplot(x='Delays', y='Overall Performance', data=data, scatter=False, color='green', line_kws={"linewidth": 2, "linestyle": "--"})

    # Plot Weighted NASA TLX Score vs. Time Delays
    sns.lineplot(x='Delays', y='Weighted NASA TLX Score', data=data, color='purple', label='Weighted NASA TLX Score', linewidth=2.5)
    sns.regplot(x='Delays', y='Weighted NASA TLX Score', data=data, scatter=False, color='purple', line_kws={"linewidth": 2, "linestyle": "--"})

    # Add critical delay line
    plt.axvline(x=critical_delay, color='red', linestyle='--', linewidth=2, label=f'Estimated Critical Delay ({int(critical_delay)} ms)')

    # Annotate the critical delay value on the plot
    plt.text(critical_delay + 50, 5, f'{int(critical_delay)} ms', color='red', fontsize=14, fontweight='bold')

    plt.title('Performance Metrics vs. Time Delays with Critical Delay Threshold', fontsize=20, fontweight='bold')
    plt.xlabel('Time Delay (ms)', fontsize=16)
    plt.ylabel('Performance (%)', fontsize=16)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(title='Metrics', fontsize=12, title_fontsize=14)
    plt.gca().patch.set_facecolor('#f7f7f7')
    plt.tight_layout()

    plt.show()

def plot_all_metrics_separately(data):
    # Create the 'Trial' column if it does not exist
    if 'Trial' not in data.columns:
        data = data.sort_values(by=['Participant Number'])  # Sort by participant number first
        data['Trial'] = data.groupby('name').cumcount() + 1  # Add trial numbers for each participant

    participants = data['name'].unique()

    # Define a color palette for each participant
    palette = sns.color_palette("husl", len(participants))

    # 1. Plot Driving Performance for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Driving Performance', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='o')
    plt.title('Driving Performance Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
    plt.ylabel('Driving Performance (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 2. Plot Maze Score for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Maze Score (%)', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='x')
    plt.title('Maze Score Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
    plt.ylabel('Maze Score (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Plot Weighted NASA TLX Score for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Weighted NASA TLX Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='D')
    plt.title('Weighted NASA TLX Score Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
    plt.ylabel('Weighted NASA TLX Score (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_all_metrics_vs_delays(data):
    participants = data['name'].unique()

    # Define a color palette for each participant
    palette = sns.color_palette("husl", len(participants))

    # 1. Plot Driving Performance for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Driving Performance', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='o')
    plt.title('Driving Performance vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Driving Performance (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 2. Plot Maze Score for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Maze Score (%)', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='x')
    plt.title('Maze Score vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Maze Score (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Plot Weighted NASA TLX Score for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Weighted NASA TLX Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='D')
    plt.title('Weighted NASA TLX Score vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Weighted NASA TLX Score (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

def statistical_analysis(data):
    # Ensure the correct column names are used
    print("Available columns: ", data.columns)

    # Strip any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Convert categorical variables (profession and gaming experience) into one-hot encoded columns
    data_encoded = pd.get_dummies(data, columns=['profession', 'gaming experience'], drop_first=True)

    # Ensure all numeric columns are properly cast as float/int
    data_encoded['Delays'] = pd.to_numeric(data_encoded['Delays'], errors='coerce')
    data_encoded['age'] = pd.to_numeric(data_encoded['age'], errors='coerce')
    data_encoded['Driving Performance'] = pd.to_numeric(data_encoded['Driving Performance'], errors='coerce')
    data_encoded['Maze Score (%)'] = pd.to_numeric(data_encoded['Maze Score (%)'], errors='coerce')
    data_encoded['Weighted NASA TLX Score'] = pd.to_numeric(data_encoded['Weighted NASA TLX Score'], errors='coerce')

    # Convert boolean columns to integers (0 or 1)
    boolean_columns = data_encoded.select_dtypes(include=['bool']).columns
    data_encoded[boolean_columns] = data_encoded[boolean_columns].astype(int)

    # Drop rows with missing or NaN values
    data_encoded = data_encoded.dropna(subset=['Driving Performance', 'Maze Score (%)', 'Weighted NASA TLX Score', 'Delays', 'age'])

    # Define independent variables (Delays, age, and encoded categorical variables)
    X = data_encoded[['Delays', 'age', 'gaming experience_yes', 'profession_phd student', 'profession_phd researcher', 'profession_software engineer']]
    X = sm.add_constant(X)  # Add constant for intercept

    # Open a file to save the regression results
    with open("regression_results.txt", "w") as f:

        # Regression model for Driving Performance
        y = np.asarray(data_encoded['Driving Performance'])
        model = sm.OLS(y, X).fit()
        f.write("=== Regression Results: Driving Performance ===\n")
        f.write(str(model.summary()))
        f.write("\n\n")

        # Plot Driving Performance vs. Delays and save plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x=data_encoded['Delays'], y=data_encoded['Driving Performance'], scatter_kws={'s': 50}, line_kws={"color": "red", "alpha": 0.7})
        plt.title('Driving Performance vs. Delays', fontsize=16)
        plt.xlabel('Delays (ms)', fontsize=14)
        plt.ylabel('Driving Performance (%)', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('driving_performance_vs_delays.png')  # Save plot
        plt.show()

        # Regression model for Maze Score
        y = np.asarray(data_encoded['Maze Score (%)'])
        model = sm.OLS(y, X).fit()
        f.write("=== Regression Results: Maze Score ===\n")
        f.write(str(model.summary()))
        f.write("\n\n")

        # Plot Maze Score vs. Delays and save plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x=data_encoded['Delays'], y=data_encoded['Maze Score (%)'], scatter_kws={'s': 50}, line_kws={"color": "blue", "alpha": 0.7})
        plt.title('Maze Score vs. Delays', fontsize=16)
        plt.xlabel('Delays (ms)', fontsize=14)
        plt.ylabel('Maze Score (%)', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('maze_score_vs_delays.png')  # Save plot
        plt.show()

        # Regression model for Weighted NASA TLX Score
        y = np.asarray(data_encoded['Weighted NASA TLX Score'])
        model = sm.OLS(y, X).fit()
        f.write("=== Regression Results: Weighted NASA TLX Score ===\n")
        f.write(str(model.summary()))
        f.write("\n\n")

        # Plot Weighted NASA TLX Score vs. Delays and save plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x=data_encoded['Delays'], y=data_encoded['Weighted NASA TLX Score'], scatter_kws={'s': 50}, line_kws={"color": "green", "alpha": 0.7})
        plt.title('Weighted NASA TLX Score vs. Delays', fontsize=16)
        plt.xlabel('Delays (ms)', fontsize=14)
        plt.ylabel('Weighted NASA TLX Score', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('nasa_tlx_vs_delays.png')  # Save plot
        plt.show()



# Main script execution
if __name__ == "__main__":
    csv_file = 'merged_nasa_tlx_with_maze_results.csv'
    data = load_and_preprocess_data(csv_file)

    # Define weights for NASA TLX dimensions
    weights = {
        "Mental Demand": 0.50,
        "Physical Demand": 0.0,
        "Temporal Demand": 0.0,
        "Performance": 0.00,
        "Effort": 0.0,
        "Frustration": 0.50
    }
    weights = {k: v / sum(weights.values()) for k, v in weights.items()}
    data = calculate_weighted_scores(data, weights)

    # Calculate performance metrics
    data = calculate_performance_metrics(data)

    # Plot metrics against time delays
    grouped_data = data.groupby('Delays').agg({
        'Driving Performance': 'mean',
        'Maze Score (%)': 'mean',
        'Overall Performance': 'mean'
    }).reset_index()
    # plot_with_regression('Delays', 'Maze Score (%)', grouped_data, 'orange', 'Time Delay (ms)', 'Maze Score (%)', 'Maze Score vs. Time Delays')
    # plot_with_regression('Delays', 'Driving Performance', grouped_data, 'blue', 'Time Delay (ms)', 'Driving Performance (%)', 'Driving Performance vs. Time Delays')
    # plot_with_regression('Delays', 'Overall Performance', grouped_data, 'green', 'Time Delay (ms)', 'Overall Performance (%)', 'Overall Performance (Driving + Maze) vs. Time Delays')

    # # Plot NASA TLX dimension comparisons
    # plot_trend_analysis(data)
    # plot_3d_surface(data)
    # plot_nasa_tlx_comparison(data)
    # plot_nasa_tlx_area(data)
    # plot_correlation_heatmap(data)
    # plot_boxplots(data)

    # # Calculate and plot learning effect
    # calculate_and_plot_learning_effect(data)

    # # # Radar charts for professions
    # for profession in ['phd student', 'software engineer', 'student', 'master student']:
    #     plot_radar_chart(data, profession)

    # # Plot performance with critical delay
    # plot_with_critical_delay(data)

    # # Plot individual participant performance
    # plot_all_metrics_separately(data)

    # # Plot individual participant performance vs. delays
    # plot_all_metrics_vs_delays(data)

    # Call the function with your dataset
    statistical_analysis(data)