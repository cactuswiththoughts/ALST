import argparse
import numpy as np
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--exp-name')
parser.add_argument('--layers')
parser.add_argument('--out_file')

args = parser.parse_args()
layers = list(map(int, args.layers.split(',')))

data_dict = {
    'Layer': layers,
    'Precision': [],
    'Recall': [],
    'Accuracy': [],
    'F1': []
}
for l in layers:
    exp_dir = Path(args.exp_name.format(l))
    res = pd.read_csv(exp_dir / 'result.csv')
    nrow = len(res['test_macro_f1'])
    p = res['test_precision'][nrow-1]
    r = res['test_recall'][nrow-1]
    acc = res['test_micro_f1'][nrow-1]
    f1 = res['test_macro_f1'][nrow-1]

    data_dict['Precision'].append(p)
    data_dict['Recall'].append(r)
    data_dict['Accuracy'].append(acc)
    data_dict['F1'].append(f1)

df = pd.DataFrame(data_dict)
df.to_csv(args.out_file)
