#!/bin/bash

# 1. Activate Virtual Environment
source ~/envs/landmine/bin/activate

# 2. Start Flask Monitoring
nohup python monitor/app.py &

# 3. Run Real-Time Inference
python src/data_collection.py
