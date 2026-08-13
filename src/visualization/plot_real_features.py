import os 
import matplotlib.pyplot as plt 
import pandas as pd

features_path = os.path.join("..","..","data", "processed", "real_features_semg.csv")
filtered_path = os.path.join("..", "..", "data", "processed", "real_filtered_semg.csv")

df_feat = pd.read_csv(features_path)
df_filt = pd.read_csv(filtered_path)


sample_limit = 500
time_filt = df_filt["time"][:sample_limit]
filt_ch1 = df_filt["filtered_channel1"][:sample_limit]

# Limit features data to the same time range
df_feat_limit = df_feat[df_feat["time"] <= time_filt.iloc[-1]]
time_feat = df_feat_limit["time"]
rms_ch1 = df_feat_limit["RMS_ch1"]
mav_ch1 = df_feat_limit["MAV_ch1"]

plt.figure(figsize=(12, 7))

# Subplot 1: Filtered Signal
plt.subplot(2, 1, 1)
plt.plot(time_filt, filt_ch1, color="blue", label="Filtered Signal (Ch1)")
plt.title("Filtered sEMG Signal (Channel 1)")
plt.ylabel("Voltage")
plt.legend()
plt.grid(True)

# Subplot 2: RMS vs MAV Features
plt.subplot(2, 1, 2)
plt.plot(time_feat, rms_ch1, color="red", linewidth=2, label="RMS Feature (Ch1)")
plt.plot(time_feat, mav_ch1,color="green",linewidth=2,linestyle="--",label="MAV Feature (Ch1)")
plt.title("Extracted Features (RMS vs MAV) from Filtered Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()


figures_dir = os.path.join("..", "..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "filtered_vs_features_ch1.png"))

plt.show()
