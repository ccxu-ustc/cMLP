import sys
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../Neural-GC/')
from synthetic import simulate_var
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def standardise(X, axis=0, keepdims=True, copy=False):
    if copy:
        X = np.copy(X)
    X -= X.mean(axis=axis, keepdims=keepdims)
    X /= X.std(axis=axis, keepdims=keepdims)
    return X

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')
parser.add_argument('--event', type=int, default=0,
                    help="""an index of the event to create the animation to
                            (the indexing start with zero, if you index goes beyond out
                            the total number of events (plays), it will show you the last
                            one of the game)""")
args = parser.parse_args()

device = torch.device('cpu')
event = args.event

data = np.load("../SCDNNT/simulation/simulated data final/P_" + str(event) + ".npy")
data = standardise(data)
X_np = data

T,p = data.shape

X = torch.tensor(data[np.newaxis], dtype=torch.float32, device=device)

# Plot data
fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
axarr[0].plot(X_np)
axarr[0].set_xlabel('T')
axarr[0].set_title('Entire time series')
axarr[1].plot(X_np[:50])
axarr[1].set_xlabel('T')
axarr[1].set_title('First 50 time points')
plt.tight_layout()
plt.show()

cmlp = cMLP(X.shape[-1], lag=1, hidden=[100])

# Train with ISTA
train_loss_list = train_model_ista(
    cmlp, X, lam=0.0005, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=50000,
    check_every=100)

GC = cmlp.GC().cpu().data.numpy()
GC = np.expand_dims(GC,axis=0)
GC = GC.repeat(T-1,axis=0)

np.save("./GC_cMLP_"+str(event)+".npy",GC)
