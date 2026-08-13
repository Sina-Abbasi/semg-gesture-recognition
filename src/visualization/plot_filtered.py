import os 
import pandas as pd 
import matplotlib.pyplot as plt 

# Load the processed data 
data_path = os.path.join("..","..","data", "processed", "synthetic_filtered_semg.csv")
df= pd.read_csv(data_path)

# Plot the very first 4 second(4000 samples)
sample_limit= 4000 
time = df["time"][:sample_limit]
raw_signal = df["raw_emg"][:sample_limit]
filtered_signal = df["filtered_emg"][:sample_limit]

plt.figure(figsize=(12, 6))

# Subplot 1: Raw Signal with Noise
plt.subplot(2, 1, 1)
plt.plot(time, raw_signal, color="red", alpha=0.7, linewidth=0.8)
plt.title("1. Raw sEMG Signal (With 50Hz & White Noise)")
plt.ylabel("Voltage (mV)")
plt.grid(True)

# Subplot 2: Clean Filtered Signal
plt.subplot(2, 1, 2)
plt.plot(time, filtered_signal, color="green", linewidth=0.8)
plt.title("2. Filtered sEMG Signal (Bandpass 20-450Hz + Notch 50Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.tight_layout()
plt.show()