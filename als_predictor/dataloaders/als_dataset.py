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
                size = int(l)
                self.sizes.append(size)
                self.offsets.append(offset)
                offset += size

        with open(paths[0] + '.score', 'r') as f:
            lines = f.read().strip().split('\n')
            self.labels = [
                list(map(LABELS.index, l.strip().split())) for l in lines
            ]

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        size = self.sizes[idx]
        feat = torch.tensor(self.feats[offset:offset+size])
        label = torch.tensor(self.labels[idx]) 
        return feat, label

    def __len__(self):
        return len(self.sizes)

    @property
    def max_size(self):
        return max(self.sizes)

    def collater(self, batch):
        feats = [b[0] for b in batch]
        labels = [b[1] for b in batch]

        bsz = len(feats)
        sizes = torch.tensor([feat.size(0) for feat in feats])
        d = feats[0].size(1)
        n_layers = feats[0].size(2)
        max_size = max(sizes)

        label_sizes = torch.tensor([label.size(0) for label in labels])
        max_label_size = max(label_sizes)

        padded_feats = feats[0].new_zeros(bsz, max_size, d, n_layers)
        padded_labels = - labels[0].new_ones(bsz, max_label_size)
        for i, (feat, label, size, label_size) in enumerate(
                zip(feats, labels, sizes, label_sizes)
            ):
            padded_feats[i, :size] = feat
            padded_labels[i, :label_size] = label 
        
        if not self.longitudinal:
            padded_feats = padded_feats.view(-1, 1, d)
            padded_labels = padded_labels.view(-1, 1)
        return padded_feats, padded_labels, sizes, label_sizes
