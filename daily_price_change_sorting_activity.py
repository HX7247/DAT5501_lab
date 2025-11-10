import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

file_path = r"C:\Users\test\DAT5501_lab\DAT5501_lab-2\data\HistoricalData_NVDA.csv"
df = pd.read_csv(file_path)

df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)
daily_changes = df['Close/Last'].diff().dropna()
n_values = range(7, 366)
timing_values = []

for n in n_values:
    data_subset = daily_changes[:n].values
    
    start_time = time.perf_counter()
    np.sort(data_subset)
    end_time = time.perf_counter()
    
    timing_values.append(end_time - start_time)

plt.figure(figsize=(10, 6))
plt.plot(n_values, timing_values, 'b-', label='Measured time')

nlogn = [n * np.log(n) * (timing_values[-1] / (n_values[-1] * np.log(n_values[-1]))) for n in n_values]
plt.plot(n_values, nlogn, 'r--', label='O(n log n)')

plt.xlabel('Array Size (n)')
plt.ylabel('Sorting Time (seconds)')
plt.title('Sorting Time vs Array Size')
plt.legend()
plt.grid(True)
plt.show()

correlation = np.corrcoef(timing_values, nlogn)[0, 1]
print(f"Correlation coefficient between measured times and O(n log n): {correlation:.4f}")