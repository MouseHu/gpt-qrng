import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

filename = "./correlation.pkl"
with open(filename, "rb") as f:
    dict = pkl.load(f)

types = ["empty", "homodyne_off", "homodyne_on", "photodetector"]
# freqs = ["500M", "1G", "5G", "10G"]
freqs = {"500M": 0, "1G": 1, "5G": 2, "10G": 3}
freq_label = [0.5, 1, 5, 10]
# lags = [1, 2, 3, 5, 10, 20]
lags = [1, 2, 3, 5, 10, 20]

for type in types:
    print(type)
    data = [[] for _ in range(len(freqs))]
    for key, value in dict.items():
        if type not in key:
            continue
        # print(freqs[key.split("/")[0]],list(value.values()))
        print(freqs[key.split("/")[0]], key.split("/")[1], value)
        data[freqs[key.split("/")[0]]] = list(value.values())

    data = np.array(data).transpose()
    for i, lag in enumerate(lags):
        plt.plot(freq_label, np.abs(data[i]))
    plt.title(type)
    plt.xlabel("Frequency (G)")
    plt.ylabel("Absolute Correlation")
    legend = [f"lag={lag}" for lag in lags]
    plt.legend(legend)
    plt.savefig(f"{type}_abs.png")
    plt.show()
