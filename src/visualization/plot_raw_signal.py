import pandas as pd
import matplotlib.pyplot as plt

# Load the raw sEMG data from CSV file
df = pd.read_csv("../../data/raw/synthetic_semg.csv")

# Print the first 10 rows of data in terminal
print("First 10 rows of the dataset:")
print(df.head(10))

# Plot the signal (first 4 seconds for clear visualization)
# 4 seconds * 1000 Hz = 4000 data points
plt.figure(figsize=(12, 4))
plt.plot(df["time"][:4000], df["raw_emg"][:4000], color="blue", linewidth=0.8)

plt.title("Raw sEMG Signal with Noise")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (mV)")
plt.grid(True)

# Show the plot window
plt.show()