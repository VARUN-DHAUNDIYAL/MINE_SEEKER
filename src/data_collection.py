# src/data_collection.py
import os
import time
import numpy as np
import logging
import uhd
from config import CONFIG
import yaml

# Set up logging
logging.basicConfig(
    filename="logs/system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def setup_usrp():
    try:
        usrp = uhd.usrp.MultiUSRP()
        usrp.set_rx_rate(CONFIG["SAMPLE_RATE"])
        # Note: Use the proper UHD API for tuning; this is a placeholder.
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CONFIG["CENTER_FREQ"]))
        usrp.set_rx_gain(CONFIG["GAIN"])
        logging.info("USRP device initialized successfully.")
        return usrp
    except Exception as e:
        logging.error(f"USRP Setup Failed: {e}")
        return None

def collect_rf_data():
    raw_dir = CONFIG["DATA_RAW_DIR"]
    os.makedirs(raw_dir, exist_ok=True)
    usrp = setup_usrp()
    if usrp is None:
        logging.error("USRP device unavailable. Exiting data collection.")
        return None

    num_samples = CONFIG["NUM_SAMPLES"]
    samples = np.zeros(num_samples, dtype=np.complex64)

    try:
        # Acquire data using the USRP API. Adjust based on your UHD version.
        recv_streamer = usrp.get_rx_stream()
        recv_streamer.recv(samples)
        output_file = os.path.join(raw_dir, "rf_data.npy")
        np.save(output_file, samples)
        logging.info(f"RF data saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"RF Data Collection Failed: {e}")
        return None

def real_time_inference():
    """
    Continuously collects RF data and pushes it to a thread-safe queue.
    Multi-threaded inference is handled in a separate module.
    """
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import torch
    from scripts.inference import load_model

    # Set up device and load model for inference.
    device = torch.device(CONFIG["DEVICE"])
    model = load_model(device)
    
    rf_data_queue = queue.Queue()

    def inference_worker():
        while True:
            if not rf_data_queue.empty():
                rf_signal = rf_data_queue.get()
                try:
                    # Preprocess: Convert to 2-channel (real, imag), pad/trim to fixed length (e.g., 256)
                    if rf_signal.dtype in [np.complex64, np.complex128]:
                        X = np.stack([rf_signal.real, rf_signal.imag], axis=0)
                    else:
                        X = rf_signal
                    if X.shape[1] < 256:
                        pad_width = 256 - X.shape[1]
                        X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        X = X[:, :256]
                    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(X_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                    logging.info(f"Real-Time Prediction: {prediction}")
                except Exception as e:
                    logging.error(f"Inference error: {e}")

    # Start inference threads
    executor = ThreadPoolExecutor(max_workers=4)
    for _ in range(4):
        executor.submit(inference_worker)

    # Main loop: continuously collect RF data and add to queue
    while True:
        data_file = collect_rf_data()
        if data_file:
            rf_signal = np.load(data_file)
            rf_data_queue.put(rf_signal)
        time.sleep(1)

if __name__ == "__main__":
    # For one-time collection; for real-time, call real_time_inference()
    collect_rf_data()
