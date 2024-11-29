import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from tmu.models.classification.vanilla_classifier import TMClassifier
import time
import matplotlib.patches as patches
import logging
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from tqdm import tqdm
import json

# Set logging level for matplotlib to WARNING or ERROR
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def create_samples_with_centered_reliability(
    num_samples=1000,
    num_values=500,
    std_dev=100,
    min_reliability=0.1,
    max_reliability=1,
    noise_function="exponential"
):
    assert num_samples % 2 == 0, "num_samples should be even to divide equally between classes."
    assert 0 <= min_reliability < max_reliability <= 1, "Reliability must be between 0 and 1, with min < max."

    half_values = num_values // 2
    class_0 = np.hstack((np.ones((num_samples // 2, half_values)), np.zeros((num_samples // 2, half_values))))
    class_1 = np.hstack((np.zeros((num_samples // 2, half_values)), np.ones((num_samples // 2, half_values))))

    matrix = np.vstack((class_0, class_1))
    center = num_values / 2

    # Define different noise intensity functions
    x = np.arange(num_values)
    if noise_function == "exponential":
        noise_intensity = 1 - np.exp(-((x - center) ** 2) / (2 * std_dev ** 2))
    elif noise_function == "linear":
        noise_intensity = np.abs(x - center) / center
    elif noise_function == "quadratic":
        noise_intensity = ((x - center) / center) ** 2
    elif noise_function == "sinusoidal":
        noise_intensity = (1 + np.sin(2 * np.pi * (x / num_values - 0.5))) / 2
    elif noise_function == "uniform":
        noise_intensity = np.ones(num_values)
    else:
        raise ValueError(f"Invalid noise function: {noise_function}")

    # Scale noise intensity to match min and max reliability
    reliability = 1 - noise_intensity
    reliability_min = reliability.min()
    reliability_max = reliability.max()

    # Handle cases where reliability_min == reliability_max
    if reliability_max == reliability_min:
        scaled_reliability = np.full_like(reliability, min_reliability)
    else:
        scaled_reliability = min_reliability + (reliability - reliability_min) * \
                             (max_reliability - min_reliability) / (reliability_max - reliability_min)

    noise_intensity = 1 - scaled_reliability  # Convert back to noise intensity

    # Ensure no NaN or Inf values
    if np.any(np.isnan(noise_intensity)) or np.any(np.isinf(noise_intensity)):
        raise ValueError("Noise intensity contains invalid values.")
    noise_intensity = np.nan_to_num(noise_intensity, nan=0, posinf=0, neginf=0)  # Replace NaN/Inf with 0

    # Normalize probabilities
    noise_intensity_sum = np.sum(noise_intensity)
    if noise_intensity_sum == 0:
        noise_intensity = np.ones_like(noise_intensity) / len(noise_intensity)  # Uniform if zero sum
    else:
        noise_intensity /= noise_intensity_sum

    noise_matrix = np.zeros((num_samples, num_values), dtype=int)
    for i in range(num_samples):
        noise_positions = np.random.choice(num_values, size=num_values, p=noise_intensity)
        noise_matrix[i, noise_positions] = 1

    noisy_matrix = np.logical_xor(matrix, noise_matrix).astype(int)
    return noisy_matrix, noise_matrix




def count_ta_states(tm):
    number_of_literals = tm.clause_banks[0].number_of_features
    ta_states_array = np.zeros((tm.number_of_classes, 2, tm.number_of_clauses // 2, number_of_literals), dtype=np.uint32)

    for class_idx in range(tm.number_of_classes):
        for polarity in [0, 1]:
            for clause_idx in range(tm.number_of_clauses // 2):
                for bit in range(number_of_literals):
                    action = tm.get_ta_action(clause=clause_idx, ta=bit, the_class=class_idx, polarity=polarity)
                    ta_states_array[class_idx, polarity, clause_idx, bit] = action
    return ta_states_array
def calculate_bit_frequencies(ta_states):
    global_frequencies = np.zeros(ta_states.shape[-1])
    for class_idx in range(ta_states.shape[0]):
        for polarity in range(ta_states.shape[1]):
            for clause_idx in range(ta_states.shape[2]):
                global_frequencies += ta_states[class_idx, polarity, clause_idx]
    return global_frequencies
def calculate_bit_frequencies_with_polarity(ta_states):
    positive_frequencies = np.zeros(ta_states.shape[-1])  # Polarity 0
    negative_frequencies = np.zeros(ta_states.shape[-1])  # Polarity 1

    for class_idx in range(ta_states.shape[0]):
        for clause_idx in range(ta_states.shape[2]):
            positive_frequencies += ta_states[class_idx, 0, clause_idx]
            negative_frequencies += ta_states[class_idx, 1, clause_idx]

    return positive_frequencies, negative_frequencies
def count_clause_weights(tm, ta_states_array=None):
    if ta_states_array is None:
        ta_states_array = count_ta_states(tm)

    number_of_clauses = tm.number_of_clauses // 2
    number_of_literals = tm.clause_banks[0].number_of_features

    positive_weights = np.zeros(number_of_literals)
    negative_weights = np.zeros(number_of_literals)

    for class_idx in range(tm.number_of_classes):
        for polarity in [0, 1]:
            for clause_idx in range(number_of_clauses):
                weight = tm.get_weight(the_class=class_idx, polarity=polarity, clause=clause_idx)
                
                for bit in range(number_of_literals):
                    # Check if the literal is used in the clause (ta_state is non-zero)
                    if ta_states_array[class_idx, polarity, clause_idx, bit] != 0:
                        if polarity == 0:
                            positive_weights[bit] += weight
                        else:
                            negative_weights[bit] += weight

    return positive_weights, negative_weights
def select_random_bits(total_bits, num_bits):
    return random.sample(range(total_bits), num_bits)
def replace_least_freq(selected_bits, total_bits, bit_frequencies, num_replace):
    selected_bits = list(set(selected_bits))
    
    sorted_bits = np.argsort(bit_frequencies)
    removed_bits = sorted_bits[:num_replace]

    selected_bits = np.delete(selected_bits, removed_bits)    
    remaining_bits = list(set(range(total_bits)) - set(selected_bits))
    if len(remaining_bits) < num_replace:
        raise ValueError("Not enough remaining bits to replace.")
    
    new_bits = random.sample(remaining_bits, num_replace)
    selected_bits = np.append(selected_bits,new_bits)
    
    return selected_bits[:total_bits], removed_bits
def plot_distribution(matrix, noise_matrix):
    noise_distribution = np.sum(noise_matrix, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(noise_distribution, label='Noise Distribution', marker='x', linestyle='--', color='r')
    plt.title("Noise Distribution Across Positions")
    plt.xlabel("Position")
    plt.ylabel("Sum of 1's (Noise Intensity)")
    plt.legend()
    plt.show()


epochs = 50
max_included_literals = 32

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import json

# Directory to save animations and results
output_dir = "tm_parameter_effects"
os.makedirs(output_dir, exist_ok=True)

# Default parameter values
default_params = {
    "num_samples": 500,
    "num_values": 500,
    "std_dev": 500 / 5,
    "min_reliability": 0,
    "max_reliability": 1,
    "noise_function": "linear",
    "clauses": 100,
    "T": 10,
    "s": 5,
    "num_bits_ratio": 2,
    "num_replace_ratio": 2
}

# Parameter variations
parameter_variations = {
    "num_samples": [250, 500, 1000],
    "num_values": [250, 500, 1000],
    "std_dev": [1, 2, 5, 10],  # ratio, so calculate as num_values/std_dev
    "min_reliability": [0.1, 0.3, 0.5],
    "max_reliability": [1, 0.7, 0.5],
    "noise_function": ["sinusoidal", "exponential", "linear", "quadratic", "uniform"],
    "clauses": [10, 100, 300],
    "T": [1, 10, 100, 1000],
    "s": [1, 3, 5, 10],
    "num_bits_ratio": [1.1, 2, 5, 10, 50],
    "num_replace_ratio": [1, 2, 5, 10, 50, 10, 20]
}

# Total combinations
total_combinations = sum(len(v) for v in parameter_variations.values())
print(f"Total combinations: {total_combinations}")

# Function to run experiments and create animations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import json

# Directory to save animations and results
output_dir = "tm_parameter_effects"
os.makedirs(output_dir, exist_ok=True)

# Function to create and save animationsfrom matplotlib.animation import FFMpegWriter
from matplotlib.animation import FFMpegWriter
def generate_animation(
    bit_frequencies_history_epochs,
    positive_frequencies_history_epochs,
    negative_frequencies_history_epochs,
    positive_weights_history_epochs,
    negative_weights_history_epochs,
    accuracy_history,
    param_name,
    param_value
):
    epochs = len(bit_frequencies_history_epochs)
    num_bits = len(bit_frequencies_history_epochs[0])

    # Prepare combined data for differences
    difference_history_epochs = [
        pos - neg for pos, neg in zip(positive_frequencies_history_epochs, negative_frequencies_history_epochs)
    ]
    difference_weights_epochs = [
        pos - neg for pos, neg in zip(positive_weights_history_epochs, negative_weights_history_epochs)
    ]
    total_frequencies_epochs = [
        pos + neg for pos, neg in zip(positive_frequencies_history_epochs, negative_frequencies_history_epochs)
    ]
    total_weights_epochs = [
        pos + neg for pos, neg in zip(positive_weights_history_epochs, negative_weights_history_epochs)
    ]

    # Create the figure with 2 rows and 4 graphs
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(
        f"Testing {param_name}: {param_value} | Accuracy: Final {round(accuracy_history[-1], 2)}%",
        fontsize=16
    )  # Spanning title with accuracy description

    # First Column: Total Frequencies (Top) and Weights (Bottom)
    ax1, ax5 = axes[:, 0]
    bars1 = ax1.bar(range(num_bits), total_frequencies_epochs[0], color='blue')
    ax1.set_title('Total Frequencies')
    ax1.set_xlabel('Bit Index')
    ax1.set_ylabel('Frequency')
    ax1.set_ylim(0, max(max(row) for row in total_frequencies_epochs) + 10)
    ax1_2 = ax1.twinx()  # Secondary y-axis for accuracy
    accuracy_line, = ax1_2.plot([], [], color='red', marker='o', label='Accuracy')
    ax1_2.set_ylabel('Accuracy (%)')
    ax1_2.set_ylim(0, 100)
    ax1_2.legend(loc='upper right')

    bars5 = ax5.bar(range(num_bits), total_weights_epochs[0], color='blue')
    ax5.set_title('Total Weights')
    ax5.set_xlabel('Bit Index')
    ax5.set_ylabel('Weight')
    ax5.set_ylim(
        min(min(row) for row in total_weights_epochs) - 10,
        max(max(row) for row in total_weights_epochs) + 10
    )

    # Second Column: Positive Frequencies (Top) and Weights (Bottom)
    ax2, ax6 = axes[:, 1]
    bars2 = ax2.bar(range(num_bits), positive_frequencies_history_epochs[0], color='green')
    ax2.set_title('Positive Frequencies')
    ax2.set_xlabel('Bit Index')
    ax2.set_ylabel('Positive Frequency')
    ax2.set_ylim(0, max(max(row) for row in positive_frequencies_history_epochs) + 10)

    bars6 = ax6.bar(range(num_bits), positive_weights_history_epochs[0], color='green')
    ax6.set_title('Positive Weights')
    ax6.set_xlabel('Bit Index')
    ax6.set_ylabel('Positive Weight')
    ax6.set_ylim(
        min(min(row) for row in positive_weights_history_epochs) - 10,
        max(max(row) for row in positive_weights_history_epochs) + 10
    )

    # Third Column: Negative Frequencies (Top) and Weights (Bottom)
    ax3, ax7 = axes[:, 2]
    bars3 = ax3.bar(range(num_bits), negative_frequencies_history_epochs[0], color='orange')
    ax3.set_title('Negative Frequencies')
    ax3.set_xlabel('Bit Index')
    ax3.set_ylabel('Negative Frequency')
    ax3.set_ylim(0, max(max(row) for row in negative_frequencies_history_epochs) + 10)

    bars7 = ax7.bar(range(num_bits), negative_weights_history_epochs[0], color='orange')
    ax7.set_title('Negative Weights')
    ax7.set_xlabel('Bit Index')
    ax7.set_ylabel('Negative Weight')
    ax7.set_ylim(
        min(min(row) for row in negative_weights_history_epochs) - 10,
        max(max(row) for row in negative_weights_history_epochs) + 10
    )

    # Fourth Column: Differences (Frequencies and Weights)
    ax4, ax8 = axes[:, 3]
    bars4 = ax4.bar(range(num_bits), difference_history_epochs[0], color='purple')
    ax4.set_title('Frequency Differences (Positive - Negative)')
    ax4.set_xlabel('Bit Index')
    ax4.set_ylabel('Difference')
    ax4.set_ylim(
        min(min(row) for row in difference_history_epochs) - 10,
        max(max(row) for row in difference_history_epochs) + 10
    )

    bars8 = ax8.bar(range(num_bits), difference_weights_epochs[0], color='purple')
    ax8.set_title('Weight Differences (Positive - Negative)')
    ax8.set_xlabel('Bit Index')
    ax8.set_ylabel('Difference')
    ax8.set_ylim(
        min(min(row) for row in difference_weights_epochs) - 10,
        max(max(row) for row in difference_weights_epochs) + 10
    )

    # Update function for animation
    def update(epoch):
        # Update Total Frequencies and Accuracy
        for bar, freq in zip(bars1, total_frequencies_epochs[epoch]):
            bar.set_height(freq)
        accuracy_x = np.linspace(0, num_bits - 1, epochs)
        accuracy_line.set_data(accuracy_x[:epoch + 1], accuracy_history[:epoch + 1])

        # Update Positive and Negative Frequencies
        for bar, freq in zip(bars2, positive_frequencies_history_epochs[epoch]):
            bar.set_height(freq)
        for bar, freq in zip(bars3, negative_frequencies_history_epochs[epoch]):
            bar.set_height(freq)

        # Update Differences
        for bar, diff in zip(bars4, difference_history_epochs[epoch]):
            bar.set_height(diff)

        # Update Total, Positive, and Negative Weights
        for bar, weight in zip(bars5, total_weights_epochs[epoch]):
            bar.set_height(weight)
        for bar, weight in zip(bars6, positive_weights_history_epochs[epoch]):
            bar.set_height(weight)
        for bar, weight in zip(bars7, negative_weights_history_epochs[epoch]):
            bar.set_height(weight)

        # Update Weight Differences
        for bar, diff in zip(bars8, difference_weights_epochs[epoch]):
            bar.set_height(diff)

    ani = FuncAnimation(fig, update, frames=epochs, interval=100)

    # Save animation using FFmpeg
    filename = f"{output_dir}/{param_name}_{param_value}.mp4"
    writer = FFMpegWriter(fps=30, metadata=dict(artist="Tsetlin Experiment"), bitrate=1800)
    ani.save(filename, writer=writer)
    plt.close(fig)



# Run experiment function with animations
def run_experiment(varied_param, varied_value, params):
    params = params.copy()
    params[varied_param] = varied_value

    # Adjust std_dev based on num_values if applicable
    if varied_param == "std_dev":
        params["std_dev"] = params["num_values"] / varied_value

    # Adjust num_bits and num_replace based on ratios
    params["num_bits"] = int(params["num_values"] / params["num_bits_ratio"])
    params["num_replace"] = max(1, int(params["num_bits"] / params["num_replace_ratio"]))

    # Create dataset
    X_train, noise_matrix = create_samples_with_centered_reliability(
        num_samples=params["num_samples"],
        num_values=params["num_values"],
        std_dev=params["std_dev"],
        min_reliability=params["min_reliability"],
        max_reliability=params["max_reliability"],
        noise_function=params["noise_function"]
    )
    Y = (np.repeat([0, 1], len(X_train) // 2)).astype(np.uint32)

    # Initialize Tsetlin Machine
    tm = TMClassifier(params["clauses"], params["T"], params["s"], max_included_literals=32, platform='CPU')

    # Histories
    bit_frequencies_history = []
    positive_frequencies_history = []
    negative_frequencies_history = []
    positive_weights_history = []
    negative_weights_history = []
    accuracy_history = []

    # Training loop
    for epoch in range(100):  # Shorter epochs for quick testing
        X_train_bits = X_train
        tm.fit(X_train_bits.astype(np.uint32), Y)

        # Accuracy
        accuracy = 100 * (tm.predict(X_train_bits.astype(np.uint32)) == Y).mean()
        accuracy_history.append(accuracy)

        # Record literals and weights
        ta_states = count_ta_states(tm)
        positive_literals, negative_literals = calculate_bit_frequencies_with_polarity(ta_states)
        weights_positive, weights_negative = count_clause_weights(tm, ta_states)

        bit_frequencies_history.append(calculate_bit_frequencies(ta_states))
        positive_frequencies_history.append(positive_literals)
        negative_frequencies_history.append(negative_literals)
        positive_weights_history.append(weights_positive)
        negative_weights_history.append(weights_negative)

    # Generate and save animation
    generate_animation(
        bit_frequencies_history,
        positive_frequencies_history,
        negative_frequencies_history,
        positive_weights_history,
        negative_weights_history,
        accuracy_history,
        varied_param,
        varied_value
    )

    # Return result
    return {
        "parameter": varied_param,
        "value": varied_value,
        "accuracy": max(accuracy_history)
    }



# Run experiments
results = []
for param, variations in tqdm(parameter_variations.items(), desc="Parameters"):
    for value in tqdm(variations, desc=f"Testing {param}", leave=False):
        result = run_experiment(param, value, default_params)
        results.append(result)

# Save results to JSON
with open(f"{output_dir}/results.json", "w") as f:
    json.dump(results, f)

print("Experiments complete. Results saved.")
