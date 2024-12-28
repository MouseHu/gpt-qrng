import numpy as np
import matplotlib
import pickle as pkl

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os

rootdir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/"
subdirs = os.listdir(rootdir)
print(subdirs)


def process(inputdir, outputdir):
    print(inputdir)
    with open(inputdir, "r") as f:
        data = f.readlines()
        raw_data = [float(line.strip("\n")) for line in data]
        print(len(raw_data))
        print(raw_data[:10])
        values, counts = np.unique(raw_data, return_counts=True)
        value_dict = {value: i for i, value in enumerate(values)}

        # print(values)
        print(value_dict)
        print(counts)
        discrete_value = np.array([value_dict[v] for v in raw_data], dtype=np.uint16)
        print(len(discrete_value))
        # discrete_value.tofile(outputdir)
        with open(outputdir, "wb") as f:
            pkl.dump(value_dict, f)


for subdir in subdirs:
    subsubdirs = os.listdir(os.path.join(rootdir, subdir))
    print(subsubdirs)
    for subsubdir in subsubdirs:
        output = [f for f in os.listdir(os.path.join(rootdir, subdir, subsubdir)) if "output.dict" in f]
        if len(output) > 0:
            print(f"Skipping {os.path.join(rootdir, subdir, subsubdir)}")
            continue
        filenames = [f for f in os.listdir(os.path.join(rootdir, subdir, subsubdir)) if
                     ".dat" in f and "output" not in f]
        # print(filenames)
        if len(filenames) < 1:
            continue
        inputdir = os.path.join(rootdir, subdir, subsubdir, filenames[0])
        # outputdir = os.path.join(rootdir, subdir, subsubdir, "output.dat")
        outputdir = os.path.join(rootdir, subdir, subsubdir, "output.dict")
        process(inputdir, outputdir)

# datadir = ["/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/4.photodetector/500M-4-06.dat"]
# # datadir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/1.empty/500M-1-03.dat"
# savedir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/500M/4.photodetector/output.dat"
# nbins = 128
# with open(datadir, "r") as f:
#     data = f.readlines()
#     raw_data = [float(line.strip("\n")) for line in data]
#     print(len(raw_data))
#     print(raw_data[:10])
#     values, counts = np.unique(raw_data, return_counts=True)
#     value_dict = {value: i for i, value in enumerate(values)}
#
#     print(values)
#     print(value_dict)
#     discrete_value = np.array([value_dict[v] for v in raw_data], dtype=np.uint16)
#     # print(discrete_value[:100])
#     # bin_data = [int(min(max(0, nbins // 2 - 1 + np.round((datapoint - mean) / binsize)), nbins - 1)) for datapoint in
#     #             raw_data]
#     # print(bin_data[:100])
#     #
#     # print(np.mean(raw_data), np.std(raw_data), np.min(raw_data), np.max(raw_data))
#     # print(np.max(bin_data), np.min(bin_data))
#     # bincount = np.bincount(bin_data)
#     # print(bincount)
#     # print(len(counts))
#     # plt.plot(counts)
#     # plt.show()
#     print(len(discrete_value))
#     discrete_value.tofile(savedir)
