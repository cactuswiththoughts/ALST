import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
LABELS = ['0', '1', '2', '3', '4']


class ALSFeatureDataset(Dataset):
    '''Load ALS feature frames for the same speaker as a longitudinal sequence'''
    def __init__(self, paths, longitudinal=True):
        self.longitudinal = longitudinal
        feats_list = []
        for path in paths:
            feats = np.load(path + '.npy')    
            if feats.ndim == 3:
                feats = feats.squeeze(1)
            feats_list.append(feats)
        self.feats = np.stack(feats_list, axis=-1)

        self.offsets = []
        self.sizes = []
        with open(paths[0] + '.lengths', 'r') as f:
            lines = f.read().strip().split('\n')
            offset = 0    
            for l in lines:
                size = list(map(int, l.split()))
                self.sizes.append(size)
                self.offsets.append(offset)
                offset += sum(size)

        with open(paths[0] + '.score', 'r') as f:
            lines = f.read().strip().split('\n')
            self.labels = [
                list(map(LABELS.index, l.strip().split())) for l in lines
            ]

        # Extract transcripts
        if Path(paths[0] + '.ltr').exists():
            with open(paths[0] + '.ltr', 'r') as f:
                lines = f.read().strip().split('\n')
                self.phn_labels = [ 
                    [ord(c)-ord('A')+1 for c in l.strip().split()]
                    for l in lines
                ]
        else:
            self.phn_labels = [[0]*sum(size) for size in self.sizes]

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        size = self.sizes[idx]
        feat = torch.tensor(self.feats[offset:offset+sum(size)])
        label = torch.tensor(self.labels[idx])
        phn_label = torch.tensor(self.phn_labels[idx])
        return feat, label, size, phn_label

    def __len__(self):
        return len(self.sizes)

    @property
    def max_size(self):
        return max(sum(s) for s in self.sizes)

    def collater(self, batch):
        feats = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        sizes = [b[2] for b in batch]
        phn_labels = [b[3] for b in batch]

        bsz = len(feats)        
        d = feats[0].size(1)
        n_layers = feats[0].size(2)
        sent_sizes = torch.tensor([sum(s) for s in sizes])
        max_size = max(sent_sizes)

        label_sizes = torch.tensor([label.size(0) for label in labels])
        max_label_size = max(label_sizes)

        phn_sizes = [len(p) for p in phn_labels]
        max_phn_size = max(phn_sizes)

        padded_feats = feats[0].new_zeros(bsz, max_size, d, n_layers)
        padded_labels = - labels[0].new_ones(bsz, max_label_size)
        padded_phn_labels = phn_labels[0].new_zeros(bsz, max_phn_size)
        pool_mask = feats[0].new_zeros(bsz, max_label_size, max_size)
        for i, (feat, label, phn_label, size, label_size, phn_size) in enumerate(
                zip(feats, labels, phn_labels, sizes, label_sizes, phn_sizes)
            ):
            padded_feats[i, :sum(size)] = feat
            padded_labels[i, :label_size] = label
            padded_phn_labels[i, :phn_size] = phn_label
            offset = 0
            for j, s in enumerate(size):
                pool_mask[i, j, offset:offset+s] = 1. / s
                offset += s

        if not self.longitudinal:
            padded_feats = padded_feats.view(-1, 1, d)
            padded_labels = padded_labels.view(-1, 1)
            padded_phn_labels = padded_labels.view(-1, 1)
        return padded_feats, padded_labels, sent_sizes, label_sizes,\
            pool_mask, padded_phn_labels
