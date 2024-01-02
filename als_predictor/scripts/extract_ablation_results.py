import argparse
import numpy as np
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--exp-name')
parser.add_argument('--hp-name')
parser.add_argument('--hps')
parser.add_argument('--out_file')

args = parser.parse_args()
hps = list(map(float, args.hps.split(',')))

data_dict = {
    args.hp_name: hps,
    'Precision': [],
    'Recall': [],
    'Accuracy': [],
    'F1': [],
    r'Spearman $\rho$': [],
    r'Kendall $\tau$': [],
    'Pairwise Accuracy': [],
    'Test MSE': [],
    'Precision (MSE)': [],
    'Recall (MSE)': [],
    'Accuracy (MSE)': [],
    'F1 (MSE)': [],
    r'Spearman $\rho$ (MSE)': [],
    r'Kendall $\tau$ (MSE)': [],
    'Pairwise Accuracy (MSE)': [],
}
for l in hps:
    exp_dir = Path(args.exp_name.format(l))
    res = pd.read_csv(exp_dir / 'result.csv')
    nrow = len(res['test_macro_f1'])
    p = res['test_precision'][nrow-1]
    r = res['test_recall'][nrow-1]
    acc = res['test_micro_f1'][nrow-1]
    f1 = res['test_macro_f1'][nrow-1]
    spearmanr = res['test_spearmanr'][nrow-1]
    kendalltau = res['test_kendalltau'][nrow-1]
    pairwise_acc = res['test_pairwise_acc'][nrow-1]
    test_loss = res['test_loss'][nrow-1]

    p_mse, r_mse, acc_mse, f1_mse, spearmanr_mse, kendalltau_mse, pairwise_acc_mse = 0, 0, 0, 0, 0, 0, 0
    if 'test_precision_mse' in res:
        p_mse = res['test_precision_mse'][nrow-1]
        r_mse = res['test_recall_mse'][nrow-1]
        acc_mse = res['test_micro_f1_mse'][nrow-1]
        f1_mse = res['test_macro_f1_mse'][nrow-1]
        spearmanr_mse = res['test_spearmanr_mse'][nrow-1]
        kendalltau_mse = res['test_kendalltau_mse'][nrow-1]
        pairwise_acc_mse = res['test_pairwise_acc_mse'][nrow-1]

    data_dict['Precision'].append(p)
    data_dict['Recall'].append(r)
    data_dict['Accuracy'].append(acc)
    data_dict['F1'].append(f1)
    data_dict[r'Spearman $\rho$'].append(spearmanr)
    data_dict[r'Kendall $\tau$'].append(kendalltau)
    data_dict['Pairwise Accuracy'].append(pairwise_acc)
    data_dict['Test MSE'].append(test_loss)
    data_dict['Precision (MSE)'].append(p_mse)
    data_dict['Recall (MSE)'].append(r_mse)
    data_dict['Accuracy (MSE)'].append(acc_mse)
    data_dict['F1 (MSE)'].append(f1_mse)
    data_dict[r'Spearman $\rho$ (MSE)'].append(spearmanr_mse)
    data_dict[r'Kendall $\tau$ (MSE)'].append(kendalltau_mse)
    data_dict['Pairwise Accuracy (MSE)'].append(pairwise_acc_mse)

df = pd.DataFrame(data_dict)
df.to_csv(args.out_file)
