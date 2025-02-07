# src/preprocessing.py
import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import logging
from config import CONFIG

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def preprocess_rf_data():
    raw_file = os.path.join(CONFIG["DATA_RAW_DIR"], "rf_data.npy")
    processed_dir = CONFIG["DATA_PROCESSED_DIR"]
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        rf_data = np.load(raw_file)
    except Exception as e:
        logging.error(f"Failed to load raw RF data: {e}")
        return

    lowcut = CONFIG["CENTER_FREQ"] - 0.05e9
    highcut = CONFIG["CENTER_FREQ"] + 0.05e9
    filtered_data = bandpass_filter(rf_data, lowcut, highcut, CONFIG["SAMPLE_RATE"])
    normalized_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
    
    processed_file = os.path.join(processed_dir, "processed_rf_data.npy")
    np.save(processed_file, normalized_data)
    logging.info(f"Processed data saved to {processed_file}")

    # Optional: Plot the first 1000 samples (real and imaginary parts)
    plt.figure()
    plt.plot(normalized_data[:1000].real, label="Real")
    plt.plot(normalized_data[:1000].imag, label="Imag")
    plt.legend()
    plt.title("Preprocessed RF Signal (First 1000 Samples)")
    plt.show()

if __name__ == "__main__":
    preprocess_rf_data()
