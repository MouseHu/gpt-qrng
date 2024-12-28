import numpy as np

data_dir = "./logs/thermal_gpt-mini_seed_42_batch_size_1024_958.log"

with open(data_dir,"r") as f:
    data = f.readlines()
    data = [line.strip("\n").split("Accuracy: ")[-1] for line in data if "Accuracy" in line]
    data = [float(line) for line in data]
    print(len(data), np.mean(data),np.max(data))
    print(data)
    print(np.std(data))