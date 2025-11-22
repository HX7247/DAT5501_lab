import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Load stock price data
file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\HistoricalData_NVDA.csv"
df = pd.read_csv(file_path)

# Clean the 'Close/Last' column by removing '$' and converting to float
df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)

# Compute daily price changes
daily_changes = df['Close/Last'].diff().dropna()

# Range of array sizes (n) to test, log-spaced
n_values = np.logspace(1, 5, 50, dtype=int)

# Number of repeated timing trials per n
n_repeats = 50
timing_values = []

# Benchmark sorting time for different array sizes
for n in n_values:
    times = []
    # Randomly sample 'n' daily changes
    data_subset = np.random.choice(daily_changes, size=n)
    
    for _ in range(n_repeats):
        data = data_subset.copy()  # fresh copy for each run
        start_time = time.perf_counter()
        np.sort(data)              # sort and measure time
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    timing_values.append(np.median(times))  # store median time

# Plot results
plt.figure(figsize=(12, 8))

# Scatter plot of measured sorting times
plt.scatter(n_values, timing_values, c='blue', s=30, alpha=0.5, label='Measured time')

# Compute scaled O(n log n) curve for comparison
scale_factor = timing_values[-1] / (n_values[-1] * np.log(n_values[-1]))
nlogn = [n * np.log(n) * scale_factor for n in n_values]

# Plot O(n log n) reference curve
plt.plot(n_values, nlogn, 'r-', label='O(n log n) fit', linewidth=2)

# Use log–log axes
plt.xscale('log')
plt.yscale('log')

# Labels and styling
plt.xlabel('Array Size (n)')
plt.ylabel('Sorting Time (seconds)')
plt.title('Sorting Time Complexity Analysis')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

# Correlation between measured times and n log n fit (log-log space)
correlation = np.corrcoef(np.log(timing_values), np.log(nlogn))[0, 1]
print(f"Correlation coefficient between measured times and O(n log n) in log space: {correlation:.4f}")
