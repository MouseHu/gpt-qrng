import numpy as np

datadir = "/data/qrng_new/Vacuum_Fluctuation/rawdata-5-16-combine1G_50m.dat"


def read_data(data_chunk, nbits=12):
    if nbits == 8:
        data = np.fromfile(data_chunk, dtype=np.uint8)
    elif nbits == 12:
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        data = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    else:
        return None
    return data


def entropy(dist):
    dist = dist / np.sum(dist)
    ent = -1 * np.sum(dist * np.log(dist))
    min_ent = - np.log(np.max(dist))
    return ent, min_ent


data = read_data(datadir)
print(len(data))

bincount = np.bincount(data)

print(entropy(bincount))

# 7.690878215471744
# 7.695720941033684
# 7.7011982845961455 7.1397298952617385
