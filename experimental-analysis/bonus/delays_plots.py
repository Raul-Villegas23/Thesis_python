import numpy as np
import matplotlib.pyplot as plt

# Parameters
desired_times = [1.0, 0.5, 0.25]  # Desired time constants in seconds
delta_t = 0.01                    # Time step
total_time = 5                    # Total simulation time
n_steps = int(total_time / delta_t)  # Number of time steps

# Target position (constant for simplicity)
p_target = 1.0

# Time vector
time = np.linspace(0, total_time, n_steps)

# Initialize target positions
target_positions = [p_target] * n_steps  # Store target positions

# Plot setup
plt.figure(figsize=(10, 6))

# Loop over each desired time constant (1 sec, 500 ms, 250 ms)
for desired_time in desired_times:
    # Calculate alpha for the given desired time constant
    alpha = 1 / desired_time
    
    # Initialize camera position
    p_camera = 0.0  # Camera starts at 0
    camera_positions = [p_camera]  # Store camera positions over time
    
    # Simulation loop
    for t in range(1, n_steps):
        # Calculate difference between target and camera
        delta_p = p_target - p_camera
        
        # Compute interpolation factor based on alpha and delta_t
        f = 1 - np.exp(-alpha * delta_t)
        
        # Update camera position
        p_camera = p_camera + f * delta_p
        
        # Store the updated position
        camera_positions.append(p_camera)
    
    # Plot the camera positions for this alpha
    plt.plot(time, camera_positions, label=f'Camera Position (Time Constant = {desired_time}s)', linewidth=2)

# Plot target position
plt.plot(time, target_positions, label='Target Position', color='red', linestyle='--', linewidth=2)

# Configure plot
plt.title('Camera Lag Update (First-Order Lag) for Different Time Constants', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Position', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

# Show plot
plt.show()
