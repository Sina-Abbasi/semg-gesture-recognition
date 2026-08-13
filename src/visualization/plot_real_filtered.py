import os 
import pandas as pd 
import matplotlib.pyplot as plt 

data_path = os.path.join("..", "..", "data", "processed", "real_filtered_semg.csv")
df = pd.read_csv(data_path)

sample_limit = 500 
time = df["time"][:sample_limit]

# Raw vs Filtered for channel 1 & 2 
raw_ch1 = df["channel1"][:sample_limit]
filtered_ch1 = df["filtered_channel1"][:sample_limit]

raw_ch2 = df["channel2"][:sample_limit]
filtered_ch2 = df["filtered_channel2"][:sample_limit]

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.plot(time, raw_ch1, color="red", alpha=0.5, label="Raw Ch1")
plt.plot(time, filtered_ch1, color="blue", label="Filtered Ch1")
plt.title("Real sEMG - Channel 1 (Raw vs Filtered)")
plt.ylabel("Voltage")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, raw_ch2, color="orange", alpha=0.5, label="Raw Ch2")
plt.plot(time, filtered_ch2, color="green", label="Filtered Ch2")
plt.title("Real sEMG - Channel 2 (Raw vs Filtered)")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage")
plt.legend()
plt.grid(True)

plt.tight_layout()
figures_dir = os.path.join("..", "..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "real_raw_vs_filtered.png"))
plt.show()