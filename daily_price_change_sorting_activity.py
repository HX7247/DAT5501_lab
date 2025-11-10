import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\HistoricalData_NVDA.csv"
df = pd.read_csv(file_path)
df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)
daily_changes = df['Close/Last'].diff().dropna()

n_values = np.logspace(1, 5, 50, dtype=int) 
n_repeats = 50 
timing_values = []

for n in n_values:
    times = []
    data_subset = np.random.choice(daily_changes, size=n)
    
    for _ in range(n_repeats):
        data = data_subset.copy()  
        start_time = time.perf_counter()
        np.sort(data)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    timing_values.append(np.median(times))

plt.figure(figsize=(12, 8))

plt.scatter(n_values, timing_values, c='blue', s=30, alpha=0.5, label='Measured time')

scale_factor = timing_values[-1] / (n_values[-1] * np.log(n_values[-1]))
nlogn = [n * np.log(n) * scale_factor for n in n_values]
plt.plot(n_values, nlogn, 'r-', label='O(n log n) fit', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size (n)')
plt.ylabel('Sorting Time (seconds)')
plt.title('Sorting Time Complexity Analysis')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()

correlation = np.corrcoef(np.log(timing_values), np.log(nlogn))[0, 1]
print(f"Correlation coefficient between measured times and O(n log n) in log space: {correlation:.4f}")