import numpy as np
import matplotlib.pyplot as plt

# تعریف پارامترهای الگوریتم
num_fish = 50                # تعداد ماهی‌ها
num_iterations = 100         # تعداد تکرارها
dim = 2                      # ابعاد فضای جستجو
visual_distance = 10         # محدوده دید ماهی
step_size = 1                # اندازه گام هر حرکت
crowd_factor = 0.5           # عامل تراکم برای کنترل رفتار گروهی
space_bounds = [0, 100]      # محدوده فضای جستجو برای هر بعد


def objective_function(x):
    return sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

# Random initialization of Fishes
fish_positions = np.random.uniform(space_bounds[0], space_bounds[1], (num_fish, dim))

# Random initialization of Best position
best_position = np.array([0, 0])

# Limit the positions in the search range
def keep_within_bounds(position):
    return np.clip(position, space_bounds[0], space_bounds[1])


def plot_fish_positions(iteration, fish_positions, best_position):
    plt.figure(figsize=(6, 6))
    plt.xlim(space_bounds[0], space_bounds[1])
    plt.ylim(space_bounds[0], space_bounds[1])

    plt.scatter(fish_positions[:, 0], fish_positions[:, 1], c='blue', label='Fish Positions', alpha=0.5)
    plt.scatter(best_position[0], best_position[1], c='red', marker='x', label='Best Position')
    plt.title(f"Iteration: {iteration}")
    plt.legend()
    plt.show()



# Main AFSA
for iteration in range(num_iterations):
    for i in range(num_fish):

        # Random Searching Behavior
        new_position = fish_positions[i] + (np.random.rand(dim) - 0.5) * step_size
        new_position = keep_within_bounds(new_position)

        if objective_function(new_position) < objective_function(fish_positions[i]):
            fish_positions[i] = new_position

        # Following Behavior
        for j in range(num_fish):
            if i != j and np.linalg.norm(fish_positions[j] - fish_positions[i]) < visual_distance:
                if objective_function(fish_positions[j]) < objective_function(fish_positions[i]):
                    direction = (fish_positions[j] - fish_positions[i]) * step_size
                    fish_positions[i] = keep_within_bounds(fish_positions[i] + direction)
                    break

        # Swarming Behavior
        neighbors = [
            fish_positions[j] for j in range(num_fish)
            if np.linalg.norm(fish_positions[j] - fish_positions[i]) < visual_distance
        ]

        if neighbors:
            center_of_neighbors = np.mean(neighbors, axis=0)
            if (objective_function(center_of_neighbors) / len(neighbors)) < (objective_function(fish_positions[i]) * crowd_factor):
                direction = (center_of_neighbors - fish_positions[i]) * step_size
                fish_positions[i] = keep_within_bounds(fish_positions[i] + direction)

        # Prey Behavior
        new_position = fish_positions[i] + (np.random.rand(dim) - 0.5) * step_size
        new_position = keep_within_bounds(new_position)

        if objective_function(new_position) < objective_function(fish_positions[i]):
            fish_positions[i] = new_position

    
    if iteration % 2 == 0:
        # find best position
        best_position = fish_positions[np.argmin([objective_function(pos) for pos in fish_positions])]

        # plot fish position
        plot_fish_positions(iteration, fish_positions, best_position)

# final best position
best_position = fish_positions[np.argmin([objective_function(pos) for pos in fish_positions])]
best_value = objective_function(best_position)

print("Best Position:", best_position, "Best Value:", best_value)
