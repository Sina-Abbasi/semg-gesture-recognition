import os
import numpy as np 
import pandas as pd 

def generate_semg_signal(duration=10, sampling_rate=1000):
    # Create time steps from 0 to duration
    t = np.linspace(0, duration , sampling_rate * duration, endpoint=False)

    # Simple muscle signal (combining two frequencies eg: 35Hz and 90Hz)
    muscle_signal = np.sin(2 * np.pi * 35 * t) + 0.5 * np.sin(2 * np.pi * 90 * t)

    # Muscle activation: muscle is active for 2 seconds, then rests for 2 seconds
    activation = (np.sin(2 * np.pi * 0.25 * t) > 0).astype(int)
    clean_signal = muscle_signal * activation

    # Powerline noise (50Hz frequency)
    noise_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)

    # Random noise (white noise)
    random_noise = np.random.normal(0, 0.2, len(t))

    # Final raw EMG signal (signal + noise)
    raw_signal = clean_signal + noise_50hz + random_noise

    return t, raw_signal

# Run the script
if __name__ == "__main__":
    time, signal = generate_semg_signal(duration=10, sampling_rate=1000)

    # Store time and signal in a DataFrame
    df = pd.DataFrame({"time": time, "raw_emg": signal})

    # Save to data/raw directory
    os.makedirs("../../data/raw", exist_ok=True)
    df.to_csv("../../data/raw/synthetic_semg.csv", index=False)

    print("Data successfully generated and saved to data/raw/synthetic_semg.csv")
    