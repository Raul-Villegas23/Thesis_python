import tkinter as tk
from tkinter import messagebox
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Initialize participant count
participant_count = 1

# Check if CSV file exists and initialize participant_count
csv_file = 'nasa_tlx_results.csv'
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows:
            # Update participant_count based on the last participant number in the file
            participant_count = int(rows[-1][0]) + 1
else:
    # If file does not exist, create it with headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow([
            "Mental Demand", 
            "Physical Demand", 
            "Temporal Demand", 
            "Performance", 
            "Effort", 
            "Frustration", 
            "Overall Score"
        ])

# Function to save the entered data to a CSV file
def save_to_csv():
    global participant_count
    try:
        # Collecting input values
        mental_demand = int(mental_demand_entry.get())
        physical_demand = int(physical_demand_entry.get())
        temporal_demand = int(temporal_demand_entry.get())
        performance = int(performance_entry.get())
        effort = int(effort_entry.get())
        frustration = int(frustration_entry.get())
        
        # Validate input values
        if not all(0 <= x <= 100 for x in [mental_demand, physical_demand, temporal_demand, performance, effort, frustration]):
            raise ValueError("All values must be between 0 and 100")
        
        # Calculate overall score
        overall_score = (mental_demand + physical_demand + temporal_demand + performance + effort + frustration) / 6
        
        # Data to write
        data = [
            participant_count,
            mental_demand,
            physical_demand,
            temporal_demand,
            performance,
            effort,
            frustration,
            overall_score
        ]
        
        # Write data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        # Show success message
        messagebox.showinfo("Success", "Data saved successfully!")
        
        # Increment participant count for the next entry
        participant_count += 1
        
        # Clear inputs
        clear_inputs()
        
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Function to clear input fields
def clear_inputs():
    mental_demand_entry.delete(0, tk.END)
    physical_demand_entry.delete(0, tk.END)
    temporal_demand_entry.delete(0, tk.END)
    performance_entry.delete(0, tk.END)
    effort_entry.delete(0, tk.END)
    frustration_entry.delete(0, tk.END)

def plot_data():
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            data = list(reader)

            if not data:
                messagebox.showerror("Error", "No data to plot.")
                return

            participants = [int(row[0]) for row in data]
            mental_demand = [int(row[1]) for row in data]
            physical_demand = [int(row[2]) for row in data]
            temporal_demand = [int(row[3]) for row in data]
            performance = [int(row[4]) for row in data]
            effort = [int(row[5]) for row in data]
            frustration = [int(row[6]) for row in data]
            overall_score = [float(row[7]) for row in data]

            # Calculate statistics
            subscales = [mental_demand,physical_demand ,temporal_demand, performance, effort, frustration, overall_score] # Excluding Physical Demand
            subscale_names = ["Mental Demand","Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration", "Overall Score"] # Excluding Physical Demand

            mean_values = [np.mean(subscale) for subscale in subscales]
            std_dev_values = [np.std(subscale) for subscale in subscales]

            # Separate window for subscale plots with trend lines
            plt.figure(figsize=(10, 6))
            for i, subscale in enumerate(subscales[:-1]):  # Exclude the overall score from individual subscale plots
                plt.plot(participants, subscale, label=subscale_names[i], marker='o')
                z = np.polyfit(participants, subscale, 1)
                p = np.poly1d(z)
                plt.plot(participants, p(participants), linestyle='--')
            plt.title("NASA-TLX Subscales with Trend Lines", fontsize=14)
            plt.xlabel("Participant", fontsize=12)
            plt.ylabel("Score (0-100)", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=2.0)
            plt.show()

            # Separate window for overall score plot
            plt.figure(figsize=(10, 6))
            plt.plot(participants, overall_score, label="Overall Score", marker='o', color='red')
            z = np.polyfit(participants, overall_score, 1)
            p = np.poly1d(z)
            plt.plot(participants, p(participants), linestyle='--', color='red')
            plt.title("NASA-TLX Overall Score with Trend Line", fontsize=14)
            plt.xlabel("Participant", fontsize=12)
            plt.ylabel("Overall Score", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=2.0)
            plt.show()

            # Separate window for histogram of subscales
            plt.figure(figsize=(10, 6))
            for i, subscale in enumerate(subscales[:-1]):  # Exclude overall score from subscale histograms
                plt.hist(subscale, alpha=0.5, bins=10, label=subscale_names[i])
            plt.title("Histogram of NASA-TLX Subscales", fontsize=14)
            plt.xlabel("Score (0-100)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=2.0)
            plt.show()

            # Separate window for box and whiskers plot, now including the overall score
            plt.figure(figsize=(10, 6))
            plt.boxplot(subscales, labels=subscale_names)
            plt.title("Box and Whiskers Plot of NASA-TLX Subscales and Overall Score", fontsize=14)
            plt.ylabel("Score (0-100)", fontsize=12)
            plt.xticks(rotation=15, fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=2.0)
            plt.show()

            # Separate window for normal distribution plots (excluding Physical Demand)
            plt.figure(figsize=(10, 6))
            x = np.linspace(0, 100, 1000)
            for i, subscale in enumerate(subscales[:-1]):  # Exclude overall score from normal distribution plot
                mu, std = stats.norm.fit(subscale)
                p = stats.norm.pdf(x, mu, std)
                plt.plot(x, p, label=f'{subscale_names[i]} (mean={mu:.2f}, std={std:.2f})')
            plt.title("Normal Distributions of NASA-TLX Subscales", fontsize=14)
            plt.xlabel("Score (0-100)", fontsize=12)
            plt.ylabel("Probability Density", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=2.0)
            plt.show()

            # Display mean and standard deviation in a message box
            stats_message = "Subscale Statistics:\n"
            for i, name in enumerate(subscale_names):
                stats_message += f"{name}: Mean = {mean_values[i]:.2f}, Std Dev = {std_dev_values[i]:.2f}\n"
            messagebox.showinfo("Statistics", stats_message)
            print(stats_message)
    else:
        messagebox.showerror("Error", "CSV file does not exist. Please save some data first.")

# Initialize the main application window
app = tk.Tk()
app.title("NASA TLX Survey")

# Create labels, questions, and entry fields for each NASA-TLX subscale
tk.Label(app, text="Mental Demand (0-100):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
mental_demand_entry = tk.Entry(app)
mental_demand_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How mentally demanding was the task?").grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Physical Demand (0-100):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
physical_demand_entry = tk.Entry(app)
physical_demand_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How physically demanding was the task?").grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Temporal Demand (0-100):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
temporal_demand_entry = tk.Entry(app)
temporal_demand_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How hurried or rushed was the pace of the task?").grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Performance (0-100):").grid(row=6, column=0, padx=10, pady=5, sticky="w")
performance_entry = tk.Entry(app)
performance_entry.grid(row=6, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How successful were you in accomplishing what you were asked to do?").grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Effort (0-100):").grid(row=8, column=0, padx=10, pady=5, sticky="w")
effort_entry = tk.Entry(app)
effort_entry.grid(row=8, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How hard did you have to work to accomplish your level of performance?").grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="w")

tk.Label(app, text="Frustration (0-100):").grid(row=10, column=0, padx=10, pady=5, sticky="w")
frustration_entry = tk.Entry(app)
frustration_entry.grid(row=10, column=1, padx=10, pady=5, sticky="w")
tk.Label(app, text="How insecure, discouraged, irritated, stressed, and annoyed were you?").grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky="w")

# Button to submit the scores
submit_button = tk.Button(app, text="Submit", command=save_to_csv)
submit_button.grid(row=12, column=0, columnspan=2, pady=10, padx=50)

# Button to plot the data
plot_button = tk.Button(app, text="Plot Data", command=plot_data)
plot_button.grid(row=13, column=0, columnspan=2, pady=10, padx=50)

# Run the application
app.mainloop()
