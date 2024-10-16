import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import csv
import os
import time  # For tracking timing of user actions

def draw_maze():
    # Define grid size and initialize the maze array
    grid_size = 7
    maze = np.zeros((grid_size, grid_size))
    EMPTY, SELECTED, START, FINISH, CORRECT, INCORRECT = 0, 3, 1, 2, 4, 5

    # Start and finish points
    maze[0, 0] = START
    maze[6, 6] = FINISH

    # Define the correct path (excluding start and finish)
    correct_path = {
        (1, 0), (2, 0), (3, 0), (4, 0),
        (4, 1), (4, 2), (3, 2), (2, 2), (1, 2),
        (0, 2), (0, 3), (0, 4), (1, 4), (2, 4),
        (2, 5), (2, 6), (3, 6), (4, 6), (5, 6)
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'blue', 'cyan', 'gray'])
    norm = plt.cm.colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)
    img = ax.imshow(maze, cmap=cmap, norm=norm, aspect='equal')

    # Setting up the grid lines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)

    user_path = set()
    click_times = []  # To store timing of each click
    selection_sequence = []  # To store order of block selections
    error_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                         fontsize=14, ha='center', va='center', color='black', weight='bold')
    start_time = time.time()  # Record the start time of the task

    def onclick(event):
        # Check if the click is within the plot area, not on the button
        if event.inaxes == ax:
            if event.xdata is not None and event.ydata is not None:
                ix, iy = int(event.xdata + 0.5), int(event.ydata + 0.5)
                if 0 <= ix < grid_size and 0 <= iy < grid_size and (iy, ix) not in {(0, 0), (6, 6)}:
                    if (iy, ix) in user_path:
                        # Deselect the block if it was already selected
                        user_path.remove((iy, ix))
                        maze[iy, ix] = EMPTY
                    else:
                        # Select the block if it was not already selected
                        user_path.add((iy, ix))
                        maze[iy, ix] = SELECTED
                        selection_sequence.append((iy, ix))  # Track selection sequence
                        click_times.append(time.time())  # Record the time of this selection
                    img.set_data(maze)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)

    def finalize(event=None):
        # Function to evaluate and display results after user completes selection
        correct_selections = user_path.intersection(correct_path)
        incorrect_selections = user_path.difference(correct_path)
        missed_selections = correct_path.difference(user_path)

        # Color the path correctly for visual feedback
        for y, x in user_path:
            if (y, x) in correct_path:
                maze[y, x] = CORRECT  # Cyan for correct
            else:
                maze[y, x] = INCORRECT  # Gray for incorrect
        img.set_data(maze)

        # Define weights for error scoring
        w_i = 0.5  # Weight for incorrect selections
        w_m = 0.5  # Weight for missed selections

        # Calculate weighted error
        total_selections = len(user_path)
        total_correct_blocks = len(correct_path)
        if total_selections > 0:
            incorrect_weighted = w_i * (len(incorrect_selections) / total_selections)
        else:
            incorrect_weighted = 0
        missed_weighted = w_m * (len(missed_selections) / total_correct_blocks)

        weighted_error = (incorrect_weighted + missed_weighted) * 100
        score = 100 - weighted_error

        # Calculate task completion time
        end_time = time.time()
        total_time = end_time - start_time  # Total time in seconds

        # Update the error percentage text on the plot
        error_text.set_text(f'Score: {score:.2f}%')

        # Ensure the figure is updated
        fig.canvas.draw_idle()

        # Save the maze score to CSV
        file_exists = os.path.exists('maze_results.csv')

        with open('maze_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Maze Score'])  # Write header if the file doesn't exist
            writer.writerow([f'{score:.2f}'])

        # Print evaluation results
        print(f"Score: {score:.2f}%")

    # Button to finalize selections
    ax_button = plt.axes([0.7, 0.025, 0.25, 0.075])
    button = Button(ax_button, 'Finalize and Check')
    button.on_clicked(finalize)

    plt.show()

draw_maze()
