# -*- coding: utf-8 -*-
# Modified from gopt: https://github.com/YuanGongND/gopt

# train and test the models
import argparse
import numpy as np
import sys
import os
import os.path as osp
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-dir", type=str, default="../data/als/with_score/whisper_large-v2/feat_mean+concat", help="directory containing extracted features and labels")
parser.add_argument("--layers", type=str, default="31")
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument("--model", type=str, default='svc')
parser.add_argument("--am", type=str, default='whisper_large-v2')

def gen_result_header():
    header = ['epoch', 'train_loss', 'train_precision', 'train_recall', 'train_micro_f1', 'train_macro_f1', 'test_loss', 'test_precision', 'test_recall', 'test_micro_f1', 'test_macro_f1', 'best_macro_f1']
    return header 

args = parser.parse_args()

print(f'Layer {args.layers}')
data_dirs = [osp.join(args.data_dir, f'layer{l}') for l in args.layers.split(',')]
am = args.am
n_weights = len(data_dirs)

tr_paths = [osp.join(data_dir, 'train') for data_dir in data_dirs]
te_paths = [osp.join(data_dir, 'test') for data_dir in data_dirs]

X_tr_list = [] 
for tr_path in tr_paths:
    X_tr_list.append(np.load(tr_path + '.npy'))
X_tr = np.concatenate(X_tr_list, axis=-1)

X_te_list = [] 
for te_path in te_paths:
    X_te_list.append(np.load(te_path + '.npy'))
X_te = np.concatenate(X_te_list, axis=-1)

with open(tr_paths[0] + '.score', 'r') as f:
    lines = f.read().strip().split('\n')
    Y_tr = [y for l in lines for y in list(map(int, l.strip().split()))]
    Y_tr = np.asarray(Y_tr)

with open(te_paths[0] + '.score', 'r') as f:
    lines = f.read().strip().split('\n')
    Y_te = [y for l in lines for y in list(map(int, l.strip().split()))]
    Y_te = np.asarray(Y_te)

print(X_tr.shape, X_te.shape)
assert (X_tr.shape[0] == Y_tr.shape[0]) and (X_te.shape[0] == Y_te.shape[0])

if args.model == 'svc':
    svm = SVC()
elif args.model == 'linear_svc':
    svm = LinearSVC()
else:
    svm = SGDClassifier()
svm.fit(X_tr[:100], Y_tr[:100])
Y_te_pred = svm.predict(X_te)
_, _, micro_f1, _ = precision_recall_fscore_support(Y_te, Y_te_pred, average='micro') 
precision, recall, macro_f1, _ = precision_recall_fscore_support(Y_te, Y_te_pred, average='macro')

exp_dir = Path(args.exp_dir)
exp_dir.mkdir(parents=True, exist_ok=True)
result = np.zeros([1, 12])
result[0, 7:11] = [precision, recall, micro_f1, macro_f1]
header = ','.join(gen_result_header())
np.savetxt(exp_dir / 'result.csv', result, fmt='%.4f', delimiter=',', header=header, comments='')
print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f}')
