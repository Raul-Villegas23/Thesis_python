import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # For generating the square wave

# Define the grid coordinates of the correct path (each step is 10 meters)
correct_path = [
    (1, 0), (2, 0), (3, 0), (4, 0),  # Moving right
    (4, 1), (4, 2),                  # Moving up
    (3, 2), (2, 2), (1, 2), (0, 2),  # Moving left
    (0, 3), (0, 4),                  # Moving up
    (1, 4), (2, 4),                  # Moving right
    (2, 5), (2, 6),                  # Moving up
    (3, 6), (4, 6), (5, 6)           # Moving right
]

# Scale the path so each step is 10 meters
correct_path = [(x * 10, y * 10) for x, y in correct_path]  # Scaling coordinates by 10 meters

# Convert the path coordinates into continuous positions over time (over 60 seconds)
num_points = len(correct_path)
time = np.linspace(0, 60, num_points)  # Simulating over 60 seconds

# Extract x and y coordinates from the correct path
path_x, path_y = zip(*correct_path)
path_x = np.array(path_x)
path_y = np.array(path_y)

# Interpolate the path to get smooth transitions between the grid points
interpolated_time = np.linspace(0, 60, 500)  # 500 points over 60 seconds for smoother path
p_target_x = np.interp(interpolated_time, time, path_x)  # Interpolating x positions
p_target_y = np.interp(interpolated_time, time, path_y)  # Interpolating y positions

# Define delays and their corresponding alpha values from the dataset
delays_alpha = {
    0: 20.0,
    160: 6.25,
    400: 2.5,
    600: 1.67,
    1000: 1.0,
    1400: 0.71,
    1800: 0.56
}

# Extract the delay values and alpha (lag speed) values for plotting
delay_values = sorted(delays_alpha.keys())
alpha_values = [delays_alpha[delay] for delay in delay_values]

# Initialize the camera position (start at the same position as the target for simplicity)
p_camera_x = {delay: np.zeros_like(p_target_x) for delay in delay_values}
p_camera_y = {delay: np.zeros_like(p_target_y) for delay in delay_values}

# Initial camera positions are the same as the target at the first time step
for delay in delay_values:
    p_camera_x[delay][0] = p_target_x[0]
    p_camera_y[delay][0] = p_target_y[0]

# Run the simulation based on the provided algorithm for each delay (using corresponding alpha)
for delay, alpha in zip(delay_values, alpha_values):
    for t in range(1, len(interpolated_time)):
        # Calculate the difference between target and camera positions (for x and y)
        delta_p_x = p_target_x[t] - p_camera_x[delay][t-1]
        delta_p_y = p_target_y[t] - p_camera_y[delay][t-1]
        
        # Interpolation factor based on lag speed (alpha) and time step
        f = 1 - np.exp(-alpha * (interpolated_time[t] - interpolated_time[t-1]))
        
        # Update camera positions using the first-order delay model
        p_camera_x[delay][t] = p_camera_x[delay][t-1] + f * delta_p_x
        p_camera_y[delay][t] = p_camera_y[delay][t-1] + f * delta_p_y

# Generate a square wave for the target path (second plot)
frequency = 0.1  # Controls the frequency of the square wave (slow changes for the square path)
p_square_wave = signal.square(2 * np.pi * frequency * interpolated_time)

# Initialize the camera positions for the square wave target (for different delays)
p_camera_square = {delay: np.zeros_like(p_square_wave) for delay in delay_values}

# Initial camera positions (square wave simulation)
for delay in delay_values:
    p_camera_square[delay][0] = p_square_wave[0]

# Simulate the camera positions for the square wave target path
for delay, alpha in zip(delay_values, alpha_values):
    for t in range(1, len(interpolated_time)):
        # Calculate the difference between target and camera position for square wave
        delta_p = p_square_wave[t] - p_camera_square[delay][t-1]
        f = 1 - np.exp(-alpha * (interpolated_time[t] - interpolated_time[t-1]))
        p_camera_square[delay][t] = p_camera_square[delay][t-1] + f * delta_p

# Plot both figures (Maze path and Square wave path)
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Target position with the maze-like path
axs[0].plot(p_target_x, p_target_y, label="Target Path (Maze, 10 meters per step)", linestyle='dashed', color='black')
for delay in delay_values:
    axs[0].plot(p_camera_x[delay], p_camera_y[delay], label=f'Delay = {delay} ms')

axs[0].set_title('First-Order Delay Simulation with Maze Path (Lagged Camera Positions)')
axs[0].set_xlabel('X Position (Meters)')
axs[0].set_ylabel('Y Position (Meters)')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plot 2: Square wave target with lagged camera responses
axs[1].plot(interpolated_time, p_square_wave, label="Target Path (Square Wave)", linestyle='dashed', color='black')
for delay in delay_values:
    axs[1].plot(interpolated_time, p_camera_square[delay], label=f'Delay = {delay} ms')

axs[1].set_title('First-Order Delay Simulation with Square Wave Target Path')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Position')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Show the plots
plt.tight_layout()
plt.show()
