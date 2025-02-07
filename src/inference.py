import torch
import threading
import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from model import LandmineDetector

# Initialize logging
logging.basicConfig(filename="logs/system.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load model
model = LandmineDetector().to("cuda")
model.eval()

# Thread-safe queue for RF data
rf_data_queue = queue.Queue()

def process_rf_data():
    """ Multi-threaded inference for real-time RF signal classification. """
    while True:
        if not rf_data_queue.empty():
            rf_signal = rf_data_queue.get()
            rf_tensor = torch.tensor(rf_signal).unsqueeze(0).to("cuda")
            with torch.no_grad():
                prediction = model(rf_tensor)
            logging.info(f"Processed RF Data: Prediction={prediction.argmax().item()}")

# Run multiple inference threads
num_threads = 4  # Adjust based on hardware
executor = ThreadPoolExecutor(max_workers=num_threads)
for _ in range(num_threads):
    executor.submit(process_rf_data)

def add_rf_data(rf_signal):
    """ Adds RF data to queue for processing. """
    rf_data_queue.put(rf_signal)
    logging.info("New RF signal added to queue.")
