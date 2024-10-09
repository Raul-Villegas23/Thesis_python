import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm 

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


def compare_gaming_experience(data):
    # Group the data by 'gaming experience' and 'Delays', and calculate the mean for the selected metrics
    comparison_data = data.groupby(['gaming experience', 'Delays'])[['Time_Scores', 'Maze_Score', 'NASA_TLX_Score']].mean().reset_index()

    # Display the comparison table
    print("Comparison of gaming experience with delay conditions:")
    print(comparison_data)

    # Plot the comparison for each delay condition and each metric
    metrics = {
        'Time_Scores': 'Time Scores (seconds)', 
        'Maze_Score': 'Maze Score (%)', 
        'NASA_TLX_Score': 'NASA TLX Overall Score'
    }
    
    for metric, label in metrics.items():
        plt.figure(figsize=(12, 6))
        
        # Create a barplot for each delay condition, comparing participants with and without gaming experience
        sns.barplot(x='Delays', y=metric, hue='gaming experience', data=comparison_data, ci=None)
        
        # Add titles and labels
        plt.title(f'Comparison of {label} by Gaming Experience across Delay Conditions', fontsize=16)
        plt.xlabel('Delays (ms)', fontsize=14)
        plt.ylabel(f'Mean {label}', fontsize=14)
        
        # Display the legend and layout
        plt.legend(title='Gaming Experience', loc='upper right')
        plt.tight_layout()
        
        # Save the plot as a PNG file
        plt.savefig(f'figures/gaming_comparison_{metric}.png')
        # Show the plot
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
    numeric_columns = ['Delays', 'age', 'Driving_Performance', 'Maze_Score']
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
            for outcome in ['Driving_Performance', 'Maze_Score', 'Physical_Demand', 'Mental_Demand', 'Temporal_Demand', 'Performance', 'Effort', 'Frustration', 'gaming experience_yes']:
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


# Main script execution
if __name__ == "__main__":
    csv_file = 'updated.csv'
    data = load_and_preprocess_data(csv_file)

    # Calculate performance metrics
    data = calculate_performance_metrics(data)

    # Corrected column names for aggregation
    grouped_data = data.groupby('Delays').agg({
        'Driving_Performance': 'mean',
        'Maze_Score': 'mean',  # Corrected name
        'Overall_Performance': 'mean',
    }).reset_index()
    
    # # Call the function with your dataset
    # statistical_analysis(data)

    # Call the function with your dataset
    compare_gaming_experience(data)