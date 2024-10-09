import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import Button, VBox, Checkbox, Output

# Load the data
data = pd.read_csv('path_to_your_data.csv')

# Initialize the output area for results
output = Output()

# Create checkboxes for column selection
checkboxes = [Checkbox(value=False, description=col) for col in data.columns]

# Function to run the analysis
def run_analysis(b):
    selected_columns = [cb.description for cb in checkboxes if cb.value]
    
    # Display selected columns
    with output:
        output.clear_output()
        print("Selected columns:", selected_columns)
        
        # Check if any columns are selected
        if not selected_columns:
            print("No columns selected. Please select at least one column.")
            return
        
        # Create a new DataFrame with selected columns
        analysis_data = data[selected_columns].copy()

        # Convert percentage strings to floats if they exist in selected columns
        for col in analysis_data.select_dtypes(include=['object']).columns:
            if analysis_data[col].str.contains('%').any():
                analysis_data[col] = analysis_data[col].str.rstrip('%').astype('float') / 100

        # Remove non-numeric columns for regression analysis
        numeric_data = analysis_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            print("No numeric data for analysis. Please select numeric columns.")
            return
        
        # Run OLS regression
        X = numeric_data.drop(columns=numeric_data.columns[-1])  # Use all but the last column as features
        y = numeric_data[numeric_data.columns[-1]]  # Use the last column as the target
        
        # Add a constant to the model (intercept)
        X = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Print model summary
        print(model.summary())

        # Create a correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

# Create the button to run analysis
run_button = Button(description='Run Analysis', style=widgets.ButtonStyle())
run_button.on_click(run_analysis)

# Create a layout
layout = VBox(children=[*checkboxes, run_button, output])

# Display the interface
display(layout)
