import numpy as np
import os
import matplotlib.pyplot as plt

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



# Define the function for plotting and saving the distribution
def plot_distribution(noise_distribution, noise_function, output_dir="noise_distributions"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot noise distribution
    plt.figure(figsize=(12, 6))
    plt.plot(noise_distribution, label='Noise Distribution', marker='x', linestyle='--', color='r')
    plt.title(f"Noise Distribution Across Positions - {noise_function.capitalize()}")
    plt.xlabel("Position")
    plt.ylabel("Sum of 1's (Noise Intensity)")
    plt.legend()
    
    # Save plot to file
    output_file = os.path.join(output_dir, f"{noise_function}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved noise distribution plot for {noise_function} to {output_file}")

# Default parameters
default_params = {
    "num_samples": 500,
    "num_values": 500,
    "std_dev": 500 / 5,
    "min_reliability": 0,
    "max_reliability": 1,
    "noise_function": "linear",
}

# Noise functions to test
noise_functions = ["sinusoidal", "exponential", "linear", "quadratic", "uniform"]

# Generate and save plots for each noise function
for noise_function in noise_functions:
    print(f"Processing noise function: {noise_function}")
    params = default_params.copy()
    params["noise_function"] = noise_function
    
    # Generate samples
    _, noise_matrix = create_samples_with_centered_reliability(**params)
    
    # Calculate noise distribution
    noise_distribution = noise_matrix.sum(axis=0)
    
    # Plot and save the distribution
    plot_distribution(noise_distribution, noise_function)
