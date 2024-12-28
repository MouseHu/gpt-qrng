rootdir = "/data/2023-10-23-QRNG-DATA/rng-data/prenoise/1G/empty/"

import os

files = [os.path.join(rootdir,f) for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir,f))]
print(files)


data = []
for file in files:
    print(file)
    with open(file,"r") as f:
        data+=f.readlines()

outputdir = os.path.join(rootdir,"combine.dat")
with open(outputdir,"w") as f:
    f.writelines(data)

