import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# 1. Load your data here
# x = deflection_data
# y = loading_data

# (Generating dummy data that looks like your image for this demo)
x = np.linspace(-100, 100, 500)
# Create a shape with a steep center and linear sides
y = 0.5 * x + 30 * np.tanh(0.2 * x) + np.random.normal(0, 0.2, 500)

# 2. SMOOTHING (Crucial step)
# Use Savitzky-Golay filter to smooth while preserving edges
# window_length must be odd; adjust based on your data density
y_smooth = savgol_filter(y, window_length=51, polyorder=3)

# 3. COMPUTE DERIVATIVES
# First derivative (dy/dx) -> Stiffness
dy = np.gradient(y_smooth, x)
# Second derivative (d2y/dx2) -> Change in Stiffness (Curvature)
d2y = np.gradient(dy, x)

# 4. FIND THE KINKS
# We look for peaks in the absolute value of the 2nd derivative
# 'distance' ensures we don't pick two points right next to each other
peaks, _ = find_peaks(np.abs(d2y), height=0.1, distance=20)

# Sort peaks by height and take the top 2 (the two kinks)
top_peaks = peaks[np.argsort(np.abs(d2y)[peaks])[-2:]]
# Sort by x-position so we get Left Kink then Right Kink
kink_indices = np.sort(top_peaks)

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Raw Data', alpha=0.5)
plt.plot(x, y_smooth, 'k--', label='Smoothed')
plt.plot(x[kink_indices], y[kink_indices], "rx", markersize=12, markeredgewidth=3, label='Detected Kinks')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Automated Breakpoint Detection")
plt.show()

print(f"Breakpoint 1: x={x[kink_indices[0]]:.2f}, y={y[kink_indices[0]]:.2f}")
print(f"Breakpoint 2: x={x[kink_indices[1]]:.2f}, y={y[kink_indices[1]]:.2f}")