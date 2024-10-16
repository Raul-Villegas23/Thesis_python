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
    (3, 6), (4, 6)       # Moving right
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

# Add Gaussian noise to the path to simulate real-world noise (small deviations)
noise_std_dev = 0.25  # Standard deviation of noise (adjust as needed)
noise_x = np.random.normal(0, noise_std_dev, path_x.shape)  # Noise for X
noise_y = np.random.normal(0, noise_std_dev, path_y.shape)  # Noise for Y

# Add noise to the path
noisy_path_x = path_x + noise_x
noisy_path_y = path_y + noise_y

# Interpolate the path to get smooth transitions between the grid points
interpolated_time = np.linspace(0, 60, 1000)  # 1000 points over 60 seconds for smoother path
p_target_x = np.interp(interpolated_time, time, noisy_path_x)  # Interpolating x positions
p_target_y = np.interp(interpolated_time, time, noisy_path_y)  # Interpolating y positions

# Define delays and their corresponding alpha values
delays_alpha = {  # High alpha (instantaneous response) for 0 delay
    80: 12.5,
    600: 1.67,
    1800: 0.56
}

# Extract the delay values and alpha (lag speed) values for plotting
delay_values = sorted(delays_alpha.keys())
alpha_values = [delays_alpha[delay] for delay in delay_values]

# Initialize the camera position (start at the same position as the target for simplicity)
p_camera_x = {delay: np.zeros_like(p_target_x) for delay in delay_values}
p_camera_y = {delay: np.zeros_like(p_target_y) for delay in delay_values}

# Function to get delayed target position
def get_delayed_index(t, delay, time_step):
    delayed_time = interpolated_time[t] - delay / 1000.0  # Convert delay to seconds
    if delayed_time < 0:
        return 0  # If the delay pushes us before the start of the simulation, stay at the first position
    return np.searchsorted(interpolated_time, delayed_time)

# Simulate the camera movement based on delayed target positions
for delay, alpha in zip(delay_values, alpha_values):
    for t in range(1, len(interpolated_time)):
        # Get the delayed index for the target position
        delayed_index = get_delayed_index(t, delay, interpolated_time[t] - interpolated_time[t-1])

        # Calculate the difference between delayed target and camera positions (for x and y)
        delta_p_x = p_target_x[delayed_index] - p_camera_x[delay][t-1]
        delta_p_y = p_target_y[delayed_index] - p_camera_y[delay][t-1]
        
        # Interpolation factor based on lag speed (alpha) and time step
        f = 1 - np.exp(-alpha * (interpolated_time[t] - interpolated_time[t-1]))
        
        # Update camera positions using the first-order delay model
        p_camera_x[delay][t] = p_camera_x[delay][t-1] + f * delta_p_x
        p_camera_y[delay][t] = p_camera_y[delay][t-1] + f * delta_p_y

# Generate a square wave for the target path (second plot)
frequency = 0.03  # Controls the frequency of the square wave (slow changes for the square path)
p_square_wave = signal.square(2 * np.pi * frequency * interpolated_time)

# Initialize the camera positions for the square wave target (for different delays)
p_camera_square = {delay: np.zeros_like(p_square_wave) for delay in delay_values}

# Simulate the camera positions for the square wave target path
for delay, alpha in zip(delay_values, alpha_values):
    for t in range(1, len(interpolated_time)):
        # Get the delayed index for the square wave target position
        delayed_index = get_delayed_index(t, delay, interpolated_time[t] - interpolated_time[t-1])

        # Calculate the difference between delayed target and camera position for square wave
        delta_p = p_square_wave[delayed_index] - p_camera_square[delay][t-1]
        f = 1 - np.exp(-alpha * (interpolated_time[t] - interpolated_time[t-1]))
        p_camera_square[delay][t] = p_camera_square[delay][t-1] + f * delta_p

# Plotting

# Define color map for delays
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot 1: Target position with the noisy maze-like path
fig1 = plt.figure(figsize=(10, 6))
plt.plot(p_target_x, p_target_y, label="Target Path", linestyle='dashed', color='black', linewidth=2, marker='o', markersize=4)
for i, delay in enumerate(delay_values):
    plt.plot(p_camera_x[delay], p_camera_y[delay], label=f'Delay = {delay} ms', color=colors[i], linewidth=2, linestyle='-', marker='x')

plt.title('First-Order Delay Simulation with Maze Path (Lagged Camera Positions)', fontsize=16, fontweight='bold')
plt.xlabel('X Position (Meters)', fontsize=12)
plt.ylabel('Y Position (Meters)', fontsize=12)
plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()

# Plot 2: Square wave target with lagged camera responses
fig2 = plt.figure(figsize=(10, 6))
plt.plot(interpolated_time, p_square_wave, label="Target Path (Square Wave)", linestyle='dashed', color='black', linewidth=2, marker='o', markersize=4)
for i, delay in enumerate(delay_values):
    plt.plot(interpolated_time, p_camera_square[delay], label=f'Delay = {delay} ms', color=colors[i], linewidth=2, linestyle='-', marker='x')

plt.title('First-Order Delay Simulation with Square Wave Target Path', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, framealpha=0.7)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()

# Show both plots in separate windows
plt.show()
