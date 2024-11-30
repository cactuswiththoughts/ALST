import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
PHNS = ['<SIL>', '<UNK>', 'AH0', 'AY1', 'D', 'EY1', 'OW1', 'T', 'UW0', 'UW1', 'Y']
TRANS = [['AY1'], ['OW1'], ['Y'], ['UW1'], ['EY1'], ['Y'], ['OW1'], ['Y'], ['OW1'], ['T'], ['AH0', 'UW0'], ['D'], ['EY1']]


class ALSFeatureDataset(Dataset):
    '''Load ALS feature frames for the same speaker as a longitudinal sequence'''
    def __init__(
            self, data_paths, 
            longitudinal=True,
            use_phn_label=False,
            mask_except=None,
            split='train'
        ):
        self.split = split
        self.longitudinal = longitudinal
        self.mask_except = mask_except
        self.feats = []
        self.offsets = []
        self.sizes = []
        self.days = []
        self.labels = []
        self.phn_labels = []
        feats_list = [] 
        for path in data_paths:
            feats = np.load(path + '.npy')    
            if feats.ndim == 3:
                feats = feats.squeeze(1)
            feats_list.append(feats)
        self.feats = np.stack(feats_list, axis=-1)

        offset = 0
        with open(data_paths[0] + '.lengths', 'r') as f:
            lines = f.read().strip().split('\n')
            for l in lines:
                size = list(map(int, l.split()))
                self.sizes.append(size)
                self.offsets.append(offset)
                offset += sum(size)

        with open(data_paths[0] + '.score', 'r') as f:
            lines = f.read().strip().split('\n')
            labels = [
                list(map(int, l.strip().split())) for l in lines
            ]
            self.labels.extend(labels)

        # Extract number of days since diagnosis
        if Path(data_paths[0] + '.days').exists(): 
            with open(data_paths[0] + '.days', 'r') as f:
                lines = f.read().strip().split('\n')
                for l in lines:
                    days = list(map(int, l.split()))
                    self.days.append(days)

                max_days = max(max(d) for d in self.days)
                mean_start_days = np.mean([d[0] for d in self.days])
                mean_period = np.mean([max(d)-min(d) for d in self.days])
                intervals = [
                    [d_i - d_i_1 for d_i_1, d_i in zip(d[:-1], d[1:])]
                    for d in self.days if len(d)
                ]
                mean_interval = np.mean(
                    [interval for item in intervals for interval in item]
                )
                print(f'Maximum days since diagnosis: {max_days}')
                print(f'Mean days since diagnosis: {mean_start_days:.1f}')
                print(f'Mean period: {mean_period:.1f}')
                print(f'Mean interval: {mean_interval:.1f}')

        # Extract phoneme transcripts
        self.use_phn_label = use_phn_label
        if Path(data_paths[0] + '.phn').exists() and use_phn_label:
            with open(data_paths[0] + '.phn', 'r') as f:
                sents = f.read().strip().split('\n')
                phn_labels = [
                    [PHNS.index(phn) for phn in sent.strip().split()]
                    for sent in sents
                ]
        else:
            phn_labels = [[0]*sum(size) for size in self.sizes]
        self.phn_labels.extend(phn_labels)

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        size = self.sizes[idx]
        feat = torch.tensor(self.feats[offset:offset+sum(size)])
        label = torch.tensor(self.labels[idx])
        phn_label = torch.tensor(self.phn_labels[idx])
        if self.longitudinal and len(self.days):
            pos_ids = torch.tensor(
                [day for day, s in zip(self.days[idx], size) for _ in range(s)]
            )
        else:
            pos_ids = None

        if self.use_phn_label:
            phn_label = torch.tensor(self.phn_labels[idx])
            phn_pos_ids = torch.cat([torch.arange(s) for s in size])
        else:
            phn_label = None
            phn_pos_ids = None 
 
        return feat, label, size, phn_label, pos_ids, phn_pos_ids

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
        pos_ids = [b[4] for b in batch]
        phn_pos_ids = [b[5] for b in batch]

        bsz = len(feats)        
        d = feats[0].size(1)
        n_layers = feats[0].size(2)
        sent_sizes = torch.tensor([sum(s) for s in sizes])
        max_size = max(sent_sizes)
        max_seg_size = max([max(s) for s in sizes])

        label_sizes = torch.tensor([label.size(0) for label in labels])
        max_label_size = max(label_sizes)

        if self.use_phn_label:
            phn_sizes = [len(p) for p in phn_labels]
            max_phn_size = max(phn_sizes)
        else:
            phn_sizes = [1]*len(labels)
            max_phn_size = 1

        if self.longitudinal:
            padded_feats = feats[0].new_zeros(bsz, max_size, d, n_layers)
            padded_labels = - labels[0].new_ones(bsz, max_label_size)
            padded_phn_labels = None
            padded_phn_pos_ids = None
            if self.use_phn_label:
                padded_phn_labels = None
                padded_phn_pos_ids = phn_pos_ids[0].new_zeros(bsz, max_phn_size)

            pool_mask = feats[0].new_zeros(bsz, max_label_size, max_size)
            if pos_ids[0] is not None:
                padded_pos_ids = pos_ids[0].new_zeros(bsz, max_size)
            else:
                padded_pos_ids = None
            for i, (feat, label, phn_label, pos_id, phn_pos_id, size, label_size, phn_size) in enumerate(
                zip(feats, labels, phn_labels, pos_ids, phn_pos_ids, sizes, label_sizes, phn_sizes)
            ):
                padded_labels[i, :label_size] = label
                if self.longitudinal and (pos_ids[0] is not None):
                    padded_pos_ids[i, :len(pos_id)] = pos_id

                if self.use_phn_label:
                    padded_phn_pos_ids[i, :len(phn_pos_id)] = phn_pos_id

                padded_feats[i, :sum(size)] = feat
            
                offset = 0
                for j, s in enumerate(size):
                    pool_mask[i, j, offset:offset+s] = 1. / s
                    offset += s
        else:
            new_bsz = bsz*max_label_size
            padded_feats = feats[0].new_zeros(new_bsz, max_seg_size, d, n_layers)
            padded_labels = - labels[0].new_ones(new_bsz, 1)
            pool_mask = feats[0].new_zeros(new_bsz, 1, max_seg_size)
            padded_pos_ids = None
            padded_phn_labels = None
            padded_phn_pos_ids = None
            if self.use_phn_label:
                padded_phn_pos_ids = phn_pos_ids[0].new_zeros(new_bsz, max_phn_size) 
            sent_sizes = labels[0].new_zeros(new_bsz)
            for i, (feat, label, phn_label, phn_pos_id, size) in enumerate(
                    zip(feats, labels, phn_labels, phn_pos_ids, sizes)
                ):
                offset = 0
                for j, s in enumerate(size):
                    if self.mask_except is not None:
                        padded_feats[
                            i*max_label_size+j, self.mask_except
                        ] = feat[offset:offset+s][self.mask_except]
                    else:
                        padded_feats[i*max_label_size+j, :s] = feat[offset:offset+s]
                    padded_labels[i*max_label_size+j, :] = label[j]
                    if self.use_phn_label:
                        padded_phn_pos_ids[i*max_label_size+j, :s] = phn_pos_id[:s]

                    pool_mask[i*max_label_size+j, :, :s] = 1. / s
                    sent_sizes[i*max_label_size+j] = s
                    offset += s
        return padded_feats, padded_labels, sent_sizes, label_sizes,\
            pool_mask, padded_phn_labels, padded_pos_ids, padded_phn_pos_ids
