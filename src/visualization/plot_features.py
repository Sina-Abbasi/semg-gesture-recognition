import os 
import matplotlib.pyplot as plt 
import pandas as pd 

# Load extracted features 
features_path = os.path.join("..","..","data","processed","synthetic_features.csv")
df = pd.read_csv(features_path)

plt.figure(figsize=(12,6))

# Plot RMS (ROOT MEAN VAL)
plt.subplot(2,1,1)
plt.plot(df["time"], df["RMS"], color="purple", linewidth=1.2)
plt.title("1. Extracted RMS Feature ( Signal Power)")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Value")
plt.grid(True)

# Plot MAV (MEAN ABSOLUTE VAL)
plt.subplot(2,1,2)
plt.plot(df["time"], df["MAV"], color="Red", linewidth=1.2)
plt.title("2. Extracted MAV Feature (Amplitude Average)")
plt.xlabel("Time (Seconds)")
plt.ylabel("MAV Value")
plt.grid(True)

plt.tight_layout()
plt.show()