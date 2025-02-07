# src/main.py
import argparse
import yaml
import os

from scripts.train import train_model
from scripts.inference import run_inference

def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="RF Landmine Detection System")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "realtime"],
                        default="inference", help="Mode to run the system")
    parser.add_argument("--data_path", type=str, default="data/raw/rf_data.npy", help="Path to RF data file")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    elif args.mode == "inference":
        run_inference(args.data_path)
    elif args.mode == "realtime":
        # For real-time inference via multi-threading (see src/data_collection.py for details)
        from src.data_collection import real_time_inference
        real_time_inference()
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
