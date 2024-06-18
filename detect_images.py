import argparse
import lpips
from sklearn.metrics import roc_auc_score
import pickle
from mask_utils import *
from recon_utils import *
import glob
import torch.nn.functional as F
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--result_path', default='', type=str, help='')
parser.add_argument('--metric', default='LPIPS', type=str, help='MSE, SSIM, LPIPS, SIMCLR')
parser.add_argument('--reps', default=1, type=int, help='')
parser.add_argument('--resize_type', default='bicubic', type=str, help='simclr resize type supports none, nearest, bicubic')
args = parser.parse_args()
print(vars(args))

reps = args.reps
root = args.result_path
metric = args.metric

pos_files = glob.glob(root + 'pos/*.pth')
neg_files = glob.glob(root + 'neg/*.pth')

print('Number of positive images: {0}, negative images: {1}'.format(len(pos_files), len(neg_files)))

all_pos_eval = np.zeros((len(pos_files), reps))
all_neg_eval = np.zeros((len(neg_files), reps))

if metric == 'SIMCLR':
    simclr_criterion = SimCLRv2Loss("pretrained/r50_1x_sk1.pth")
elif metric == 'LPIPS':
    lpips_criterion = lpips.LPIPS(net='alex', pretrained=True, lpips=True).cuda()

def simclr_resize(x):
    if args.resize_type == 'none':
        return x
    elif args.resize_type in ['nearest', 'bicubic']:
        result = F.interpolate(x, 224, mode=args.resize_type.split('_')[0])
        return result
    else:
        raise NotImplementedError

def lpips_scaler(x):
    return x * 2. - 1.

def calculate_metric(orig, recon, metric):
    if metric == 'MSE':
        return F.mse_loss(recon, orig, reduction="none").mean(dim=(1, 2, 3)).detach().cpu().numpy()
    elif metric == 'SSIM':
        return ssim_criterion(orig, recon)
    elif metric == 'LPIPS':
        return lpips_criterion(lpips_scaler(orig.cuda()), lpips_scaler(recon.cuda())).flatten().detach().cpu().numpy()
    elif metric == 'SIMCLR':
        return simclr_criterion(simclr_resize(orig).cuda(), simclr_resize(recon).cuda())
    else:
        raise NotImplementedError

for idx, file in enumerate(tqdm.tqdm(pos_files, desc="Processing positive images")):
    data = torch.load(file)
    orig = data['orig']
    for r in range(reps):
        recon = data['recon'][r]
        all_pos_eval[idx, r] = calculate_metric(orig, recon, metric)

for idx, file in enumerate(tqdm.tqdm(neg_files, desc="Processing negative images")):
    data = torch.load(file)
    orig = data['orig']
    for r in range(reps):
        recon = data['recon'][r]
        all_neg_eval[idx, r] = calculate_metric(orig, recon, metric)

agg_fn = np.median
all_pos_s = agg_fn(all_pos_eval, axis=1)
all_neg_s = agg_fn(all_neg_eval, axis=1)

results = np.append(all_pos_s, all_neg_s)
labels = np.append(np.ones_like(all_pos_s), np.zeros_like(all_neg_s))
if metric == 'LPIPS' or metric == 'MSE':
    n1 = roc_auc_score(labels, results * (-1))
else:
    n1 = roc_auc_score(labels, results)

print("%s ROC AUC: %.4f" % (metric, n1))
