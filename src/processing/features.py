import os 
import numpy as np 
import pandas as pd 


INPUT_PATH = os.path.join("..", "..", "data", "processed", "real_filtered_semg.csv")
PROCESSED_DIR = os.path.join("..", "..", "data", "processed")

# 1. Calculate RMS (Root Mean Square) for a segment of signal
def calculate_rms(window):
    return np.sqrt(np.mean(window**2))

# 2. Calculate MAV (Mean Absolute Value) for a segment of signal
def calculate_mav(window):
    return np.mean(np.abs(window))

# 3. Extract features using a sliding window
def extract_multi_channel_features(df, window_size=100, step_size=50):
    feature_rows = []
    channel_cols= [f"filtered_channel{i}" for i in range (1,9)]

    # Slide the window across the signal
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_df = df.iloc[start:end]

        # Use majority voting for the class label in this window
        window_class = window_df["class"].mode()[0]

        row_data = {
            "time": window_df["time"].iloc[int(window_size / 2)],
            "class": window_class,
        }

        # Calculate RMS and MAV for each of the 8 channels
        for i, col in enumerate(channel_cols, start=1):
            window_signal = window_df[col].values
            row_data[f"RMS_ch{i}"] = calculate_rms(window_signal)
            row_data[f"MAV_ch{i}"] = calculate_mav(window_signal)

        feature_rows.append(row_data)

    return pd.DataFrame(feature_rows)

if __name__ == "__main__":
    
    df = pd.read_csv(INPUT_PATH)
    
    features_df = extract_multi_channel_features(
        df, window_size=100, step_size=50
    )

    # Save features to CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "real_features_semg.csv")
    features_df.to_csv(output_path, index=False)
    print(f"Real 8-channel features saved to: {output_path}")