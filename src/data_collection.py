import logging
import time

# Configure logging
logging.basicConfig(
    filename="logs/real_time_inference.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def real_time_inference():
    while True:
        start_time = time.time()
        try:
            samples = np.zeros(CONFIG["NUM_SAMPLES"], dtype=np.complex64)
            recv_streamer.recv(samples)
            data = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(data)
                prediction = torch.argmax(output, dim=1).item()

            latency = time.time() - start_time
            logging.info(f"Prediction: {prediction} | Latency: {latency:.4f} sec")
            print(f"[REAL-TIME] Prediction: {prediction} | Latency: {latency:.4f} sec")

        except Exception as e:
            logging.error(f"Data streaming failed: {e}")
