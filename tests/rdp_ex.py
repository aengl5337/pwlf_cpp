import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp

def RDP_find_kinks_adaptive(points, target_count=4):
    """
    Finds exactly `target_count` points (Start, Kink1, Kink2, ... End)
    by adaptively sweeping epsilon from coarse to fine.
    """
    
    # 1. Determine the scale of the data
    # We use the Y-axis span (max - min) to set our search bounds.
    # This removes the need for magic numbers like "2.0".
    y_span = np.ptp(points[:, 1])  # Peak-to-peak (max - min)
    
    # 2. Define search parameters
    # Start searching at 20% of the total height (very coarse)
    start_eps = y_span * 0.2
    # Stop if we get too fine (e.g., 0.1% of height - likely noise)
    min_eps = y_span * 0.001
    # Number of steps in our sweep
    steps = 100
    
    best_trajectory = None
    found_epsilon = None

    # 3. The Sweep (Logarithmic is usually better for scale)
    # We sweep downwards: from "Loose fit" -> "Tight fit"
    search_space = np.logspace(np.log10(start_eps), np.log10(min_eps), steps)

    for eps in search_space:
        simplified = rdp(points, epsilon=eps)
        
        # We want exactly the target count (Start + 2 Kinks + End = 4)
        # If we find MORE than 4, we've gone too deep and hit noise.
        # We return the last valid result that had <= 4 points.
        if len(simplified) == target_count:
            return np.array(simplified), eps
        
        elif len(simplified) > target_count:
            # We missed the exact target and jumped to too many points.
            # This happens if the kinks are small or noise is high.
            # In this case, the previous iteration was likely the best approximation.
            print(f"Warning: Jumped from {len(best_trajectory)} to {len(simplified)} points.")
            break
            
        # Store the current best attempt
        best_trajectory = simplified
        found_epsilon = eps

    # If we exit the loop without hitting the target, return what we have
    return np.array(best_trajectory) if best_trajectory is not None else points, found_epsilon


# --- 1. Generate Dummy Data (Matching your Joystick Profile) ---
# Simulating the steep "detent" in the center and linear springs on sides
x = np.linspace(-10, 10, 1000)
# Create a shape: Linear slope + a steep tanh step in the middle + noise
y = 0.5 * x + 5 * np.tanh(2 * x) + np.random.normal(0, 0.1, len(x))

# Combine into an (N, 2) array of points
points = np.column_stack([x, y])

# --- 2. The RDP Algorithm ---

# We want exactly 4 points: Start, Kink 1, Kink 2, End.
# We can iterate to find the perfect epsilon (threshold) that gives us this structure.
target_points = 4

# Search for the optimal epsilon
simplified_trajectory, optimal_eps = find_kinks_adaptive(points, target_count=target_points)

print(f"Optimization Complete.")
print(f"Optimal Epsilon: {optimal_eps:.4f}")
print(f"Points Found: {len(simplified_trajectory)}")

# Access the kinks (excluding start/end)
if len(simplified_trajectory) >= 3:
    kinks = simplified_trajectory[1:-1]
    print("Kinks found at:\n", kinks)
else:
    print("Could not isolate distinct kinks. Data may be too linear or too noisy.")

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))

# Plot Raw Data
plt.plot(points[:, 0], points[:, 1], color='lightgray', label='Raw Sensor Data', zorder=1)

# Plot RDP Result
plt.plot(simplified_trajectory[:, 0], simplified_trajectory[:, 1], 
         color='blue', linewidth=2, linestyle='--', label='RDP Simplification', zorder=2)

# Plot Detected Kinks
plt.scatter(kinks[:, 0], kinks[:, 1], color='red', s=100, zorder=3, label='Detected Breakpoints')

# Formatting
plt.title(f"RDP Detection (Epsilon: {optimal_eps:.2f})")
plt.xlabel("Deflection")
plt.ylabel("Loading (Force)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Output Coordinates
print(f"Kink 1 detected at: {kinks[0]}")
print(f"Kink 2 detected at: {kinks[1]}")