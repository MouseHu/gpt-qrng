import numpy as np
import pickle
from scipy.stats import entropy
from dataset.dataset import read_data

# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/1.empty/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/10G/1.empty/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/10G/4.photodetector/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/10G/2.homodyne_off/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/10G/3.homodyne_on/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/1G/homodyne_off/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/4.photodetector/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA3.413/rng-data/prenoise/500M/2.homodyne_off/output.dat"
datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/1G/photodetector/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/5G/photodetector/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/1G/empty/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/3.homodyne_on/output.dat"
# datadir = "/data/2023-10-23-QRNG-DATA/rng-data/Thermal noice/mergeddata1_500M.dat"

seqlen = 20
split = (0.97,1)
data = read_data(datadir, nbits=16)
# data = read_data(datadir, nbits=8)
scaled_split = [int(len(data) * p) for p in split]
data = data[scaled_split[0]:scaled_split[1]]
y = data[seqlen:]

x = np.bincount(y)
x = np.array(x, dtype=float)
x /= np.sum(x)
p = np.max(x)
std = np.sqrt(p * (1 - p) / len(y))
# print(np.max(x),np.argmax(x),x[50])
print(x)
print("Max Guessing Probability: ", p)
print("Num Bins: ",len(x))
print("Test Dataset Size: ",len(y))
print("Entropy: ",entropy(x))
# print(std)
print("Significant Prob (5 sigma): ", p + 5 * std)
#
# pp = 0.004091776978954442
# import math
#
# print((pp-p)/std)
# print(1-math.erf((pp-p)/std))

# nbins = np.max(y)+1
# accs = []
# print(nbins)
# for i in range(339):
#     test_y = np.random.randint(0, nbins, size=len(y))
#     correct = np.sum(test_y == y)
#     acc = correct / len(y)
#     print(acc)
#     accs.append(acc)
# #
# print(np.std(accs), np.max(accs),np.mean(accs))
