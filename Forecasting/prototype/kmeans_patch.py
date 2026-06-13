import os
import sys

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from data_preparetion import data_provider
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='kmeans')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--num_workers', type=int, default=10)

args = parser.parse_args([
    '--data', 'custom',
    '--root_path', 'exchange',
    '--data_path', 'exchange_rate.csv',
    '--percent', '100'
])
if args.freq == 0:
        args.freq = 'h'
train_data, train_loader = data_provider(args, 'train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def extract_patches(x, patch_size=16, stride=16):
    B, T = x.shape
    patches = []

    for t in range(0, T - patch_size + 1, stride):
        patches.append(x[:, t:t+patch_size])

    return torch.cat(patches, dim=0)


all_patches = []
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    batch_x = batch_x.float().to(device)
    means = batch_x.mean(1, keepdim=True).detach()
    batch_x = batch_x - means
    stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
    batch_x /= stdev
    batch_x = batch_x.squeeze(-1)
    patches = extract_patches(
        batch_x,
        patch_size=8,
        stride=8
    )
    all_patches.append(patches.cpu())
all_patches = torch.cat(all_patches, dim=0)
all_patches = all_patches.cpu()
print("Total patches:", all_patches.shape)
torch.save(all_patches, "prototype/exchange/all_patches.pt")
all_patches = torch.load("prototype/exchange/all_patches.pt", map_location="cpu")
X = all_patches.numpy()

kmeans = KMeans(
    n_clusters=100,
    random_state=42,
    n_init=10
)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
np.save("prototype/exchange/100/prototype_centers.npy", centers)