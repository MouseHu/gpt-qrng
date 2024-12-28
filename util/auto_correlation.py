import numpy as np
import pickle as pkl
from scipy.stats import entropy
from dataset.dataset import read_data

prefix = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/"
filename = "output.dat"
dictname = "output.dict"

subdirs = ["500M/1.empty", "500M/2.homodyne_off", "500M/3.homodyne_on", "500M/4.photodetector",
           "1G/empty", "1G/homodyne_off", "1G/homodyne_on", "1G/photodetector",
           "5G/empty", "5G/homodyne_off", "5G/homodyne_on", "5G/photodetector",
           "10G/1.empty", "10G/2.homodyne_off", "10G/3.homodyne_on", "10G/4.photodetector"]


# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/1.empty/output.dat"
# dictdir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/1.empty/output.dict"
def get_data(datadir, dictdir):
    seqlen = 20
    split = (0.97, 1)
    data = read_data(datadir, nbits=16)
    with open(dictdir, "rb") as f:
        dict = pkl.load(f)
    values = np.array(list(dict.keys()))
    # print(values)
    # data = read_data(datadir, nbits=8)
    scaled_split = [int(len(data) * p) for p in split]
    data = data[scaled_split[0]:scaled_split[1]]
    data = np.array(data[seqlen:])

    bins = np.bincount(data)
    probs = bins / np.sum(bins)
    maxp = np.max(bins)

    data = values[data]
    return data


lags = [1, 2, 3, 5, 10, 20]
data_dict = {}
for subdir in subdirs:
    corrs = {}
    datadir = f"{prefix}/{subdir}/{filename}"
    dictdir = f"{prefix}/{subdir}/{dictname}"
    data = get_data(datadir, dictdir)
    print(subdir)
    for lag in lags:
        print(lag)
        std1, std2 = np.std(data[lag:]), np.std(data[:-lag])
        corr = np.mean(data[lag:] * data[:-lag]) / (std1 * std2)
        print(corr)
        corrs[lag] = corr
    # print(np.correlate(data[lag:], data[:lag]))
    data_dict[subdir] = corrs

print(data_dict)

with open("correlation.pkl", "wb") as f:
    pkl.dump(data_dict, f)
