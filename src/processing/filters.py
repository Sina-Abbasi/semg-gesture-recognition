import os
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

RAW_PATH = os.path.join("..", "..", "data", "raw", "real_semg.txt")
PROCESSED_DIR = os.path.join("..", "..", "data", "processed")

# Bandpass filter: Keeps signal between 20Hz and 450Hz
def apply_bandpass(signal, low=20, high=450, fs=1000):
    nyquist = fs / 2  # Max frequency we can process (500Hz)
    b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)


# Notch filter: Removes exact 50Hz powerline noise
def apply_notch(signal, noise_freq=50, fs=1000):
    nyquist = fs / 2
    b, a = iirnotch(noise_freq / nyquist, 30)
    return filtfilt(b, a, signal)



if __name__ == "__main__":
    
    df = pd.read_csv(RAW_PATH, sep=r"\s+")

    channel_cols = [f"channel{i}" for i in range(1, 9)]

    # Applying both filters 
    for col in channel_cols:
        step1= apply_bandpass(df[col].values)
        clean_signal = apply_bandpass(step1)
        df[f"filtered_{col}"] = clean_signal

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join( PROCESSED_DIR, "real_filtered_semg.csv")
    df.to_csv(output_path, index = False)

    print("Signal successfully filtered and saved to : {output_path}!")