# %%
from src.PARAMOUNT import DMD
import os
import shutil
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import re
from src.utils import utils
from tqdm import tqdm
import matplotlib.pyplot as plt


path_parquet = f".data/UTB/Unforced/2D/raw_pq"
variables = DMD.get_folderlist(path_parquet, boolPrint=True)
var = variables[1]

tr=80
path_var = Path.cwd() / path_parquet / f"{var}"
df = pd.read_parquet(path_var, engine="pyarrow")
cutoff = int(df.shape[1] * tr / 100)
df0 = df.iloc[:, :cutoff]
df1 = df0.iloc[:, :-1]
df2 = df0.iloc[:, 1:]

# load in u, s and v from path_pod
# path_u = Path.cwd() / path_pod / f"{var}" / "u"
# path_s = Path.cwd() / path_pod / f"{var}" / "s.pkl"
# path_v = Path.cwd() / path_pod / f"{var}" / "v"
# u = pd.read_parquet(path_u, engine="pyarrow")
# v = pd.read_parquet(path_v, engine="pyarrow")
# s = utils.loadit(path_s)
u, s, v = np.linalg.svd(df1, full_matrices=False)
# r= 2000
# u = u[:, :r]
# s = s[:r]
# v = v[:r, :]
#%%
Atilde = u.T @ df2 @ v.T @ np.diag(s**-1)
Lambda, eigvecs = np.linalg.eig(Atilde)

phi = df2 @ v.T @ np.diag(s**-1) @ eigvecs
# phi = u @ eigvecs
dt = 4e-5
omega = np.log(Lambda) / dt
init = df.iloc[:, 0]
b = np.linalg.lstsq(phi, init, rcond=None)[0]
# b = np.linalg.pinv(phi) @ init
start = 0
end = cutoff
frame_skip = 1
time = np.arange(start, end, frame_skip) * dt
dynamics = np.zeros((len(b), len(time)), dtype=complex)
# for frame in range(start, end, frame_skip):
#     dynamics.append(b * np.exp(omega * (start + dt * frame)))
# p.append(np.real(np.dot(phi, b * da.exp(omega * (start + dt * frame))))
for i, t in enumerate(time):
    dynamics[:, i] = b * np.exp(omega * t)


prediction = phi @ dynamics
# prediction2 = phi2 @ dynamics
# dynamics = np.array(
#     [
#         b @ np.exp(omega * (start + dt * frame)
#         for frame in range(start, end, frame_skip)
#     ],
# )

# %%
location = 8565
plt.plot(df.iloc[location, :])
plt.plot(df0.iloc[location, :])
plt.plot(prediction.iloc[location, :])
# plt.plot(prediction2[location, :])
# %%
