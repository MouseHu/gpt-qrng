import numpy as np
import pickle


def read_data(data_chunk, nbits=12):
    if nbits == 8:
        data = np.fromfile(data_chunk, dtype=np.uint8)
    elif nbits == 12:
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    elif nbits == 16:
        data = np.fromfile(data_chunk, dtype=np.uint16)
    elif nbits == 32:
        data = np.fromfile(data_chunk, dtype=np.uint32)
    elif nbits == 64:
        data = np.fromfile(data_chunk, dtype=np.uint64)
    elif nbits == 128:
        # currently numpy does not support it, so we use pickle
        with open(data_chunk, "rb") as f:
            data = pickle.load(f)

    else:
        assert 0, "nbits must be 8, 12, 16, 32 or 64"
    return data


def entropy(probs):
    return -1 * np.sum(probs * np.log(probs))


datadir = "/data/2023-10-23-QRNG-DATA/rng-data/Thermal noice/mergeddata1_500M.dat"

seqlen = 20
split = (0.7,1)
data = read_data(datadir, nbits=8)
scaled_split = [int(len(data) * p) for p in split]
data = data[scaled_split[0]:scaled_split[1]]
y = data[seqlen:]

x = np.bincount(y)
x = np.array(x, dtype=float)
x /= np.sum(x)
p = np.max(x)
# std = np.sqrt(p * (1 - p) / len(y))
print(np.max(x),np.argmax(x),x[50])
# # print(x)
# print(len(y))
# print(entropy(x))
# print(std)
# print(p + 5 * std)
#
# pp = 0.004091776978954442
# import math
#
# print((pp-p)/std)
# print(1-math.erf((pp-p)/std))

# accs = []
# for i in range(359):
#     test_y = np.random.randint(0, 255, size=len(y))
#     correct = np.sum(test_y == y)
#     acc = correct / len(y)
#     print(acc)
#     accs.append(acc)
#
# print(np.std(accs), np.max(accs))
