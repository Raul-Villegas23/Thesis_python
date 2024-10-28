import numpy as np
import matplotlib.pyplot as plt

# Parameters
time_steps = 200  # Number of time steps to observe the stop-and-move effect
update_rate = 1 / 60  # Update rate in seconds (60 Hz)
delays = [50, 500, 1000]  # Delays in milliseconds
stop_time = 2  # Time (in seconds) after which the target stops moving

# Generate a sample "target" movement (sinusoidal pattern until stop_time)
time = np.arange(time_steps) * update_rate
target_movement = np.sin(2 * np.pi * time / max(time)) * 10  # Max 10 units displacement
stop_index = int(stop_time / update_rate)
target_movement[stop_index:] = target_movement[stop_index]  # Make the target stationary after stop_time

# Helper function to simulate delay without smoothing
def simulate_delay_no_smoothing(target, delay_ms):
    delay_steps = int(delay_ms / 1000 / update_rate)  # Convert delay to steps
    delayed_movement = np.zeros_like(target)
    for i in range(len(target)):
        # Apply delay by copying past values directly
        if i < delay_steps:
            delayed_movement[i] = target[0]  # Start at initial position until delay has passed
        else:
            delayed_movement[i] = target[i - delay_steps]
    return delayed_movement

# Helper function to simulate delay with smoothing (exponential smoothing)
def simulate_delay_with_smoothing(target, delay_ms):
    delay_s = delay_ms / 1000  # Convert to seconds
    alpha = update_rate / (delay_s + update_rate)  # Smoothing factor based on delay
    delayed_movement = np.zeros_like(target)
    delayed_movement[0] = target[0]
    for i in range(1, len(target)):
        delayed_movement[i] = alpha * target[i] + (1 - alpha) * delayed_movement[i - 1]
    return delayed_movement

# Simulate delays without smoothing and with smoothing
delayed_movements_no_smoothing = [simulate_delay_no_smoothing(target_movement, delay) for delay in delays]
delayed_movements_with_smoothing = [simulate_delay_with_smoothing(target_movement, delay) for delay in delays]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot no-smoothing subfigure
axs[0].plot(time, target_movement, label="Target Movement (Stop-and-Move)", linewidth=2, linestyle='--', color='black')
for i, delay in enumerate(delays):
    axs[0].plot(time, delayed_movements_no_smoothing[i], label=f"Delayed Movement ({delay} ms)", linewidth=1.5)
axs[0].set_title("Without Smoothing")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Position")
axs[0].legend()
axs[0].grid(True)

# Plot smoothing subfigure
axs[1].plot(time, target_movement, label="Target Movement (Stop-and-Move)", linewidth=2, linestyle='--', color='black')
for i, delay in enumerate(delays):
    axs[1].plot(time, delayed_movements_with_smoothing[i], label=f"Delayed Movement ({delay} ms)", linewidth=1.5)
axs[1].set_title("With Smoothing")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Robotic Telepresence Platform with and without Smoothing (Stop-and-Move Strategy)")
plt.show()
