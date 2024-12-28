import numpy as np

# data_dir = "./logs/thermal_gpt-mini_seed_42_batch_size_1024_958.log"
# data_dir = "./logs/prenoise5G_photon_gpt-mini_seed_42_batch_size_1024_2911.log"
data_dir = "./logs/prenoise1G_photon_gpt-mini_seed_42_batch_size_1024_2893.log"
# data_dir = "../logs/prenoise500M_photon_gpt-mini_seed_42_batch_size_1024_1437.log"
# data_dir = "../logs/prenoise500M_empty_gpt-mini_seed_42_batch_size_1024_1424.log"
# data_dir = "../logs/prenoise10G_empty_gpt-mini_seed_42_batch_size_1024_4949.log"
# data_dir = "./logs/prenoise10G_homodyneon_gpt-mini_seed_42_batch_size_1024_3998.log"
# data_dir = "./logs/prenoise500M_homodyneon_gpt-mini_seed_42_batch_size_1024_1756.log"
# data_dir = "./logs/prenoise500M_homodyneoff_gpt-mini_seed_42_batch_size_1024_1752.log"
# data_dir = "./logs/prenoise1G_homodyneon_gpt-mini_seed_42_batch_size_1024_9557.log"
# data_dir = "./logs/prenoise5G_empty_gpt-mini_seed_42_batch_size_1024_9715.log"
# data_dir = "../logs/prenoise10G_photon_gpt-mini_seed_42_batch_size_1024_4964.log"

with open(data_dir,"r") as f:
    data = f.readlines()
    data = [line.strip("\n").split("Accuracy: ")[-1] for line in data if "Accuracy" in line]
    data = [float(line) for line in data]
    print(len(data), np.mean(data),np.max(data))
    print(data)
    print(np.std(data))