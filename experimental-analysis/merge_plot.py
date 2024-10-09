import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from scipy.interpolate import griddata
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.oneway import anova_oneway

# Load and preprocess data
def load_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns={
        'Maze score': 'Maze_Score',
        'Time Scores': 'Time_Scores',
        'Driving Performance': 'Driving_Performance',
        'Overall Performance': 'Overall_Performance',
        'Overall Score': 'NASA_TLX_Score',
        'Weighted NASA TLX Score': 'Weighted_NASA_TLX_Score',
        'Mental Demand': 'Mental_Demand',
        'Physical Demand': 'Physical_Demand',
        'Temporal Demand': 'Temporal_Demand',
        'Performance': 'Performance',
        'Effort': 'Effort',
        'Frustration': 'Frustration'
    }, inplace=True)
    data['Maze_Score'] = data['Maze_Score'].str.rstrip('%').astype(float)
    return data

# Calculate performance based on time scores and overall performance
def calculate_performance_metrics(data):
    # Make sure 'Time_Scores' is used after renaming
    min_time, max_time = data['Time_Scores'].min(), data['Time_Scores'].max()

    # Calculate Driving Performance using the renamed 'Time_Scores' column
    data['Driving_Performance'] = 100 * (max_time - data['Time_Scores']) / (max_time - min_time)

    # Calculate Overall Performance using the renamed 'Maze_Score'
    data['Overall_Performance'] = (data['Driving_Performance'] + data['Maze_Score']) / 2
    
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

def calculate_and_plot_performance_by_delays(data):
    
    # Sort data by 'Delays' and 'Participant Number'
    data = data.sort_values(by=['Participant Number', 'Delays'])

    plt.figure(figsize=(14, 8))

    # Create box plot for performance metrics (no aggregation, use original data)
    data_long = pd.melt(data, id_vars=['Delays'], 
                        value_vars=['Driving_Performance', 'Maze_Score'], 
                        var_name='Metric', value_name='Performance_Value')  # Renamed value_name

    sns.boxplot(x='Delays', y='Performance_Value', hue='Metric', data=data_long, showfliers=False)

    # Plot settings
    plt.title('Performance by Delays (Box Plot)', fontsize=22, fontweight='bold')
    plt.xlabel('Time Delay (ms)', fontsize=18)
    plt.ylabel('Performance (%)', fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(title='Metrics', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.show()


# Calculate weighted NASA TLX scores
def calculate_weighted_scores(data, weights):
    for dimension, weight in weights.items():
        if dimension in data.columns:
            data[f'Weighted_{dimension}'] = data[dimension] * weight
        else:
            raise KeyError(f"Column '{dimension}' not found in data.")
    
    # Sum up the weighted scores to create the final NASA TLX score
    weighted_columns = [f'Weighted_{dim}' for dim in weights.keys()]
    data['Weighted_NASA_TLX_Score'] = data[weighted_columns].sum(axis=1)
    
    return data


def calculate_and_plot_learning_effect_barplot(data):

    data = data.sort_values(by=['Participant Number'])
    data['Trial'] = data.groupby('name').cumcount() + 1

    # Calculate mean performance metrics across trials
    mean_performance = data.groupby('Trial').agg({
        'Driving_Performance': 'mean',
        'Maze_Score': 'mean',
        'Overall_Performance': 'mean',
        'Weighted_NASA_TLX_Score': 'mean'
    }).reset_index()

    # Calculate learning effects
    learning_effect_performance = (mean_performance.iloc[2, 1:4] - mean_performance.iloc[0, 1:4]) / mean_performance.iloc[0, 1:4] * 100
    learning_effect_tlx = (mean_performance.iloc[2, 4] - mean_performance.iloc[0, 4]) / mean_performance.iloc[0, 4] * 100

    plt.figure(figsize=(14, 8))

    # Bar plot for mean performance
    mean_performance_long = pd.melt(mean_performance, id_vars='Trial', 
                                    value_vars=['Driving_Performance', 'Maze_Score'], 
                                    var_name='Metric', value_name='Performance')
    sns.barplot(x='Trial', y='Performance', hue='Metric', data=mean_performance_long)

    # Add trend lines using sns.pointplot (which aligns automatically)
    sns.pointplot(x='Trial', y='Driving_Performance', data=mean_performance, 
                color='blue', label='Driving Performance Trend', markers='o', linestyles='--', ci=None)

    sns.pointplot(x='Trial', y='Maze_Score', data=mean_performance, 
                color='orange', label='Maze Score Trend', markers='o', linestyles='--', ci=None)


    plt.title('Learning Effect Across Trials with Trend Line', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18)
    plt.ylabel('Performance (%)', fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(title='Metrics', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.show()

    print("Learning Effect (Percentage Improvement from Trial 1 to Trial 3):")
    print(learning_effect_performance.round(2))
    print(f"Change in Weighted NASA TLX Score: {learning_effect_tlx:.2f}%")


# Correlation heatmap
def plot_correlation_heatmap(data):
    dimensions = ["Mental_Demand", "Physical_Demand", "Temporal_Demand", "Performance", "Effort", "Frustration"]
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[['Delays', 'Driving_Performance', 'Maze_Score'] + dimensions].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Heatmap of Time Delays, Performance, Maze, and NASA TLX Dimensions', fontsize=18, fontweight='bold')
    plt.show()

# Box plots for distribution and density comparison
def plot_boxplots(data):
    dimensions = ["Mental_Demand", "Physical_Demand", "Temporal_Demand", "Performance", "Effort", "Frustration"]

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


def plot_all_metrics_separately(data):
    # Create the 'Trial' column if it does not exist
    if 'Trial' not in data.columns:
        data = data.sort_values(by=['Participant Number'])  # Sort by participant number first
        data['Trial'] = data.groupby('name').cumcount() + 1  # Add trial numbers for each participant

    participants = data['name'].unique()

    # Define a color palette for each participant
    palette = sns.color_palette("husl", len(participants))

    # 1. Plot Driving_Performance for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Driving_Performance', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='o')
    plt.title('Driving_Performance Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
    plt.ylabel('Driving_Performance (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 2. Plot Maze Score for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Maze_Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='x')
    plt.title('Maze Score Across Trials', fontsize=22, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
    plt.ylabel('Maze_Score', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Plot Weighted NASA TLX Score for all participants
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Trial', y='Weighted_NASA_TLX_Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='D')
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

    # 1. Plot Driving_Performance for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Driving_Performance', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='o')
    plt.title('Driving_Performance vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Driving_Performance (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 2. Plot Maze Score for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Maze_Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='x')
    plt.title('Maze Score vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Maze_Score', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # 3. Plot Weighted NASA TLX Score for all participants vs Delays
    plt.figure(figsize=(14, 8))
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(x='Delays', y='Weighted_NASA_TLX_Score', data=participant_data, color=palette[i], label=f'{participant}', linestyle='-', marker='D')
    plt.title('Weighted NASA TLX Score vs. Delays', fontsize=22, fontweight='bold')
    plt.xlabel('Delays (ms)', fontsize=18, fontweight='bold')
    plt.ylabel('Weighted NASA TLX Score (%)', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Participants', title_fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_all_metrics_separately_boxplot(data):
    # Ensure the 'Trial' column exists; create it if not
    if 'Trial' not in data.columns:
        data = data.sort_values(by=['Participant Number'])  # Sort by participant number first
        data['Trial'] = data.groupby('name').cumcount() + 1  # Add trial numbers for each participant

    metrics = ["Driving_Performance", "Maze_Score", "Weighted_NASA_TLX_Score"]

    plt.figure(figsize=(14, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)
        sns.boxplot(x='Trial', y=metric, data=data)  # Removed the hue argument for simplicity
        plt.title(f'{metric} Distribution Across Trials', fontsize=16, fontweight='bold')
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel(f'{metric}', fontsize=12)
        plt.ylim(0, 100)  # Assuming percentage-based scaling for each metric
        plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_all_metrics_vs_delays_lineplot(data):
    participants = data['name'].unique()

    # Define a color palette for each participant
    palette = sns.color_palette("husl", len(participants))

    # Create a figure with 2 subplots (subfigures) arranged vertically
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))  # 2 rows, 1 column

    # 1. Plot Driving_Performance for all participants vs Delays
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(
            x='Delays', 
            y='Driving_Performance', 
            data=participant_data, 
            color=palette[i], 
            label=f'P{participant}', 
            linestyle='-', 
            marker='o', 
            ax=axs[0], 
            linewidth=3,  # Thicker lines to highlight differences
            markersize=8  # Larger markers
        )
    axs[0].set_title('Driving_Performance vs. Delays', fontsize=22, fontweight='bold')
    axs[0].set_xlabel('Delays', fontsize=16)
    axs[0].set_ylabel('Driving_Performance (%)', fontsize=16)
    
    # Set a dynamic y-axis limit based on data to highlight participant differences
    max_perf = data['Driving_Performance'].max()
    min_perf = data['Driving_Performance'].min()
    axs[0].set_ylim(min_perf - 5, max_perf + 5)

    axs[0].grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Participants', title_fontsize=10)

    # 2. Plot Maze Score for all participants vs Delays
    for i, participant in enumerate(participants):
        participant_data = data[data['name'] == participant]
        sns.lineplot(
            x='Delays', 
            y='Maze_Score', 
            data=participant_data, 
            color=palette[i], 
            label=f'P{participant}', 
            linestyle='-', 
            marker='x', 
            ax=axs[1], 
            linewidth=3,  # Thicker lines to highlight differences
            markersize=8  # Larger markers
        )
    axs[1].set_title('Maze Score vs. Delays', fontsize=22, fontweight='bold')
    axs[1].set_xlabel('Delays', fontsize=16)
    axs[1].set_ylabel('Maze Score', fontsize=16)
    
    # Set a dynamic y-axis limit based on data to highlight participant differences
    max_maze = data['Maze_Score'].max()
    min_maze = data['Maze_Score'].min()
    axs[1].set_ylim(min_maze - 5, max_maze + 5)

    axs[1].grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Participants', title_fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust right margin for legend
    plt.show()



def statistical_analysis(data):
    """
    Perform statistical analysis on the provided data including encoding categorical variables,
    fitting regression models, and plotting results.
    """

    # Ensure the correct column names are used
    print("Available columns: ", data.columns)

    # Strip any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Convert categorical variables (profession and gaming experience) into one-hot encoded columns
    data_encoded = pd.get_dummies(data, columns=['profession', 'gaming experience'], drop_first=True)

    # Ensure all numeric columns are properly cast as float/int
    numeric_columns = ['Delays', 'age', 'Driving_Performance', 'Maze_Score', 'Weighted_NASA_TLX_Score']
    for col in numeric_columns:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')

    # Convert boolean columns to integers (0 or 1)
    boolean_columns = data_encoded.select_dtypes(include=['bool']).columns
    data_encoded[boolean_columns] = data_encoded[boolean_columns].astype(int)

    # Drop rows with missing or NaN values
    data_encoded = data_encoded.dropna(subset=numeric_columns)

    # Define independent variables
    gaming_experience_columns = [col for col in data_encoded.columns if 'gaming experience_' in col]
    X = data_encoded[['Delays', 'age'] + gaming_experience_columns]
    X = sm.add_constant(X)  # Add constant for intercept

    # Define dependent variables
    try:
        with open("regression_results.txt", "w") as f:
            for outcome in ['Driving_Performance', 'Maze_Score', 'Weighted_NASA_TLX_Score', 'Physical_Demand', 'Mental_Demand', 'Temporal_Demand', 'Performance', 'Effort', 'Frustration', 'gaming experience_yes']:
                y = np.asarray(data_encoded[outcome])
                model = sm.OLS(y, X).fit()
                f.write(f"=== Regression Results: {outcome} ===\n")
                f.write(str(model.summary()))
                f.write("\n\n")
                
                # Plot
                plt.figure(figsize=(10, 6))
                sns.regplot(x=data_encoded['Delays'], y=data_encoded[outcome], scatter_kws={'s': 50}, line_kws={"color": "red", "alpha": 0.7})
                plt.title(f'{outcome} vs. Delays', fontsize=16)
                plt.xlabel('Delays (ms)', fontsize=14)
                plt.ylabel(outcome, fontsize=14)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'figures/{outcome.replace(" ", "_")}_vs_Delays.png')
                plt.show()
                plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")



def comprehensive_regression_analysis(data):
    # Preprocessing
    data.columns = data.columns.str.strip()
    
    # One-hot encoding for categorical variables
    data_encoded = pd.get_dummies(data, columns=['profession', 'gaming experience'], drop_first=True)
    
    # Convert selected columns to numeric
    numeric_cols = ['Delays', 'age', 'Driving_Performance', 'Maze_Score', 'Weighted_NASA_TLX_Score']
    for col in numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')
    
    # Convert boolean columns to integers
    boolean_columns = data_encoded.select_dtypes(include=['bool']).columns
    data_encoded[boolean_columns] = data_encoded[boolean_columns].astype(int)
    
    # Drop rows with missing values in key columns
    data_encoded = data_encoded.dropna(subset=['Driving_Performance', 'Maze_Score', 'Weighted_NASA_TLX_Score', 'Delays', 'age'])
    
    # Standardize numeric features to improve condition number and reduce multicollinearity issues
    data_encoded[['Delays', 'age']] = (data_encoded[['Delays', 'age']] - data_encoded[['Delays', 'age']].mean()) / data_encoded[['Delays', 'age']].std()

    # Define independent variables
    X = data_encoded[['Delays', 'age', 'gaming experience_yes']]
    X = sm.add_constant(X)  # Add a constant term for the intercept
    
    # Define dependent variable
    y = data_encoded['Driving_Performance']
    
    # OLS Model
    model_ols = sm.OLS(y, X).fit()
    print("=== OLS Regression Results ===")
    print(model_ols.summary())
    
    # VIF Calculation to detect multicollinearity
    vif_df = calculate_vif(X)
    print("\n=== Variance Inflation Factors ===")
    print(vif_df)
    
    # Residual Diagnostics
    residuals = model_ols.resid
    plt.figure(figsize=(12, 5))
    
    # Residuals vs Fitted Values plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=model_ols.fittedvalues, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    
    # Residuals distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=15, color='purple')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # If multicollinearity detected (VIF > 5), consider removing variables
    if vif_df['VIF'].max() > 5:
        print("Warning: Multicollinearity detected. Consider removing highly collinear variables.")
        # You can optionally remove highly collinear variables here and refit the model

    # Fit Robust Linear Model (RLM) to handle outliers
    model_rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    print("\n=== Robust Linear Model: Driving_Performance ===")
    print(model_rlm.summary())
    
    # Fit Generalized Least Squares (GLS) to handle heteroscedasticity
    # Assume variance inversely proportional to squared Delays (you can change this assumption)
    weights = 1 / (data_encoded['Delays'] ** 2)
    model_gls = sm.GLS(y, X, sigma=weights).fit()
    print("\n=== GLS Regression Results: Driving_Performance ===")
    print(model_gls.summary())

# Helper function for VIF
def calculate_vif(X):
    # Drop the constant to avoid infinite VIF for constant term
    X_no_const = X.drop('const', axis=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_no_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i) 
                       for i in range(X_no_const.shape[1])]
    return vif_data

    
def output_summary_statistics(data):
    # Specify the columns for which we want to calculate mean, std, min, and max
    metrics = ['Driving_Performance', 'Maze_Score', 'NASA_TLX_Score', 'Weighted_NASA_TLX_Score']
    
    # Calculate mean, standard deviation, min, and max for each metric
    summary_stats = data[metrics].agg(['mean', 'std', 'min', 'max']).T
    
    # Rename columns for clarity
    summary_stats.columns = ['Mean', 'Standard Deviation', 'Minimum', 'Maximum']
    
    # Display the summary statistics table
    print("Summary Statistics for Driving Performance, Maze Score, and NASA TLX Scores:")
    print(summary_stats)
    
    return summary_stats


def calculate_and_plot_statistics(data):
    # Calculate means and standard deviations
    metrics = ['Driving_Performance', 'Maze_Score', 'NASA_TLX_Score', 'Weighted_NASA_TLX_Score']
    
    # Individual NASA TLX dimensions
    nasa_dimensions = ["Mental_Demand", "Physical_Demand", "Temporal_Demand", "Performance", "Effort", "Frustration"]
    
    all_metrics = metrics + nasa_dimensions
    
    # Initialize an empty list to store stats
    stats_list = []
    
    for metric in all_metrics:
        mean_val = data[metric].mean()
        std_val = data[metric].std()
        stats_list.append({'Metric': metric, 'Mean': round(mean_val, 2), 'Standard Deviation': round(std_val, 2), 'Max': round(data[metric].max(), 2), 'Min': round(data[metric].min(), 2)})
    
    stats_df = pd.DataFrame(stats_list)
    
    # Now, plot the table
    fig, ax = plt.subplots(figsize=(16, len(all_metrics)*0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Means and Standard Deviations of Metrics', fontsize=16)
    plt.show()


def calculate_baseline_performance(data, metric='Driving_Performance'):
    """
    This function calculates the baseline performance for each participant based on
    their performance at the shortest delay.
    """
    # Ensure the data is sorted by participant number and delays
    data = data.sort_values(by=['Participant Number', 'Delays'])
    
    # Group by participant and calculate the baseline as the performance at the shortest delay
    baseline_data = data.groupby('Participant Number').first()  # First row after sorting by delay
    
    # Extract the baseline performance (we can use Overall Performance or another metric)
    baseline_performance = baseline_data[['name', metric]].reset_index()
    
    return baseline_performance

def plot_baseline_performance(baseline_performance, y = 'Driving_Performance'):
    """
    This function plots the baseline performance for each participant.
    """
    plt.figure(figsize=(12, 8))
    
    # Plotting the baseline performance
    sns.barplot(x='name', y=y, data=baseline_performance, palette='turbo')
    
    # Adding titles and labels
    plt.title('Baseline Performance for Each Participant', fontsize=20, fontweight='bold')
    plt.xlabel('Participant', fontsize=16)
    plt.ylabel('Baseline Driving performance (seconds)', fontsize=16)
    
    # Rotating x-axis labels for better visibility
    plt.xticks(rotation=45)
    
    # Display plot
    plt.tight_layout()
    plt.show()

def perform_anova(data, dependent_var, independent_var):
    """
    Perform one-way ANOVA to compare the means of the dependent variable across different levels of the independent variable.
    """
    # Use simplified column names for the ANOVA formula
    formula = f'{dependent_var} ~ C({independent_var})'

    # Fit the OLS model and perform ANOVA
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(f"=== ANOVA Results for {dependent_var} vs. {independent_var} ===")
    print(anova_table)

    # Check the p-value using .iloc to avoid the FutureWarning
    p_value = anova_table['PR(>F)'].iloc[0]  # Fix: use .iloc[0] to access the first element by position
    if p_value < 0.05:
        print(f"\nStatistically significant difference in {dependent_var} across {independent_var} groups (p-value = {p_value:.4f}).")
    else:
        print(f"\nNo statistically significant difference in {dependent_var} across {independent_var} groups (p-value = {p_value:.4f}).")

def check_anova_assumptions(model, data, dependent_var, independent_var):
    """
    Check assumptions for ANOVA: normality of residuals, homogeneity of variances.
    """
    # 1. Residuals from the ANOVA model
    residuals = model.resid
    
    # 2. Normality of residuals (Q-Q plot and Shapiro-Wilk test)
    plt.figure(figsize=(10, 6))

    # Q-Q plot
    plt.subplot(1, 2, 1)
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title('Q-Q Plot of Residuals')

    # Histogram of residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=10)
    plt.title('Histogram of Residuals')
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test for normality
    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test for normality: W={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
    if shapiro_test.pvalue < 0.05:
        print("Warning: Residuals are not normally distributed (p-value < 0.05).")
    else:
        print("Residuals are normally distributed (p-value >= 0.05).")

    # 3. Homogeneity of variances (Levene's Test and Bartlett's Test)
    groups = [data[dependent_var][data[independent_var] == level] for level in data[independent_var].unique()]

    # Levene's test
    levene_test = stats.levene(*groups)
    print(f"Levene's test for homogeneity of variances: W={levene_test.statistic}, p-value={levene_test.pvalue}")
    if levene_test.pvalue < 0.05:
        print("Warning: Variances are not equal across groups (p-value < 0.05).")
    else:
        print("Variances are equal across groups (p-value >= 0.05).")

    # Bartlett's test (more sensitive to normality)
    bartlett_test = stats.bartlett(*groups)
    print(f"Bartlett's test for homogeneity of variances: W={bartlett_test.statistic}, p-value={bartlett_test.pvalue}")
    if bartlett_test.pvalue < 0.05:
        print("Warning: Variances are not equal across groups (p-value < 0.05).")
    else:
        print("Variances are equal across groups (p-value >= 0.05).")

def perform_non_parametric_tests(data, dependent_var, independent_var):
    """
    Perform Kruskal-Wallis test and Welch's ANOVA to handle violations of normality and homogeneity.
    """
    # Prepare the groups for Kruskal-Wallis
    groups = [data[dependent_var][data[independent_var] == level] for level in data[independent_var].unique()]

    # 1. Kruskal-Wallis Test (Non-parametric alternative to ANOVA)
    kruskal_test = stats.kruskal(*groups)
    print(f"Kruskal-Wallis test for {dependent_var} vs {independent_var}: H-statistic={kruskal_test.statistic}, p-value={kruskal_test.pvalue}")
    if kruskal_test.pvalue < 0.05:
        print(f"Significant differences in {dependent_var} across {independent_var} groups (p-value < 0.05).")
    else:
        print(f"No significant differences in {dependent_var} across {independent_var} groups (p-value >= 0.05).")

    # 2. Welch’s ANOVA (Robust to heterogeneity of variances)
    welch_anova = anova_oneway(groups, use_var='unequal', welch_correction=True)
    print(f"Welch's ANOVA for {dependent_var} vs {independent_var}: F-statistic={welch_anova.statistic}, p-value={welch_anova.pvalue}")
    if welch_anova.pvalue < 0.05:
        print(f"Significant differences in {dependent_var} across {independent_var} groups (p-value < 0.05).")
    else:
        print(f"No significant differences in {dependent_var} across {independent_var} groups (p-value >= 0.05).")



# Main script execution
if __name__ == "__main__":
    csv_file = 'updated.csv'
    data = load_and_preprocess_data(csv_file)

    # Define weights for NASA TLX dimensions
    weights = {
        "Mental_Demand": 0.50,
        "Physical_Demand": 0.0,
        "Temporal_Demand": 0.0,
        "Performance": 0.00,
        "Effort": 0.0,
        "Frustration": 0.50
    }
    weights = {k: v / sum(weights.values()) for k, v in weights.items()}
    data = calculate_weighted_scores(data, weights)

    # Calculate performance metrics
    data = calculate_performance_metrics(data)

    # Perform ANOVA
    dependent_vars = ['Driving_Performance', 'Maze_Score', 'Weighted_NASA_TLX_Score', 'Physical_Demand', 'Mental_Demand', 'Temporal_Demand', 'Performance', 'Effort', 'Frustration']
    independent_var = 'Delays'
    for var in dependent_vars:
     # Perform ANOVA
        perform_anova(data, var, independent_var)
        model=ols(f'{var} ~ C({independent_var})', data=data).fit()
        # Check ANOVA assumptions
        check_anova_assumptions(model, data, var, independent_var)
    
    for var in dependent_vars:
        print(f"--- Non-parametric tests for {var} ---")
        perform_non_parametric_tests(data, var, independent_var)

    # Corrected column names for aggregation
    grouped_data = data.groupby('Delays').agg({
        'Driving_Performance': 'mean',
        'Maze_Score': 'mean',  # Corrected name
        'Overall_Performance': 'mean',
    }).reset_index()
    
    # ## Plotting the data with regression lines
    # plot_with_regression('Delays', 'Maze_Score', grouped_data, 'orange', 'Time Delay (ms)', 'Maze_Score', 'Maze Score vs. Time Delays')
    # plot_with_regression('Delays', 'Driving_Performance', grouped_data, 'blue', 'Time Delay (ms)', 'Driving_Performance (%)', 'Driving_Performance vs. Time Delays')
    # plot_with_regression('Delays', 'Overall_Performance', grouped_data, 'green', 'Time Delay (ms)', 'Overall_Performance', 'Overall Performance (Driving + Maze) vs. Time Delays')

    # #  Calculate and plot statistics by delay
    # calculate_and_plot_performance_by_delays(data)

    # plot_correlation_heatmap(data)
    # plot_boxplots(data)
    # calculate_and_plot_learning_effect_barplot(data)

    # # # Plot individual participant performance
    # plot_all_metrics_separately(data)
    # plot_all_metrics_separately_boxplot(data)

    # # Plot individual participant performance vs. delays
    # plot_all_metrics_vs_delays(data)
    # plot_all_metrics_vs_delays_lineplot(data)

    # Call the function with your dataset
    statistical_analysis(data)

    # #Comprehensive regression analysis
    # comprehensive_regression_analysis(data)

    # # # #Output summary statistics
    # output_summary_statistics(data)

    # # # #Calculate and plot statistics
    # calculate_and_plot_statistics(data)

    # # Calculate baseline performance
    # baseline_performance = calculate_baseline_performance(data, metric='Time_Scores')
    # plot_baseline_performance(baseline_performance, y='Time_Scores')



    

