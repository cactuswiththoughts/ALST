# Extract GOP features from wav2vec 2.0
import argparse
import json
import numpy as np
from npy_append_array import NpyAppendArray
import os
import os.path as osp
from pathlib import Path
from shutil import copyfile
import torch
from prepare_data import convert_path_to_id

def encode(tgt, vocab):
     tgt_ids = list(map(vocab.index, tgt))
     tgt_ids = torch.tensor(tgt_ids)
     return tgt_ids

def read_segment(path, size):
    segs = json.load(open(path))
    
    clusts = []
    if segs[0][1] > 0:
        clusts.extend(['|']*segs[0][1])

    for s in segs:
        clusts.extend([s[0]]*(s[2]-s[1]))

    # assert size >= s[2]
    if size > s[2]:
        clusts.extend(['|']*(size-s[2]))
    elif size < s[2]:
        clusts = clusts[:size]
    return clusts

def merge_logits(logits, clusters, blank_idx=0):
    '''
    Args:
        logits: float tensor of size (seq_len, n_class)
        clusters: long tensor of size (seq_len,)
    '''
    tsz, csz = logits.shape
    new_clusters, idx, c = clusters.unique_consecutive(
        return_inverse=True, return_counts=True,
    )
    new_tsz = len(new_clusters)

    new_logits = logits.new_zeros(new_tsz, csz)
    new_logits.index_add_(dim=0, index=idx, source=logits)
    new_logits /= c.unsqueeze(-1)
    new_logits = new_logits[new_clusters != blank_idx]
    new_clusters = new_clusters[new_clusters != blank_idx]
    return new_logits, new_clusters
 
def extract_gop_feature(logits, clusters):
    new_logits, new_clusters = merge_logits(logits, clusters)
    true_logits = new_logits.gather(1, new_clusters.unsqueeze(1))
    gop_feat = torch.cat((new_logits, new_logits - true_logits), dim=1)
    return gop_feat.detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='location of .npy files')
    parser.add_argument('segment_dir', help='location of the phoneme boundary files')
    parser.add_argument('--vocab-file', required=True, help='location of the target dictionary')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)

    args = parser.parse_args()
    data = Path(args.data)
    seg_dir = Path(args.segment_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def create_file(dest):
        if osp.exists(dest):
            os.remove(dest)
        npaa = NpyAppendArray(dest)
        return npaa

    feat = np.load(data / f'{args.split}.npy')
    with open(data / f'{args.split}.lengths', 'r') as len_f:
        sizes = len_f.read().strip().split('\n')
        sizes = list(map(int, sizes))

    with open(data / f'{args.split}.tsv', 'r') as tsv_f:
        lines = tsv_f.read().strip().split('\n')
        _ = lines.pop(0)
        # Convert to utt name to wav ids
        wav_ids = [convert_path_to_id(Path(l.split('\t')[0])) for l in lines]

    with open(args.vocab_file, 'r') as f:
        vocab = [l.split()[0] for l in f.read().strip().split('\n')]
        blank = vocab[0]  # Assume the first symbol to be blank

    # Create NpyAppendArray 
    npaa = create_file(save_dir / f'{args.split}.npy')
    with open(save_dir / f'{args.split}.tsv', 'w') as tsv_f,\
        open(save_dir / f'{args.split}.seg', 'w') as seg_f,\
        open(save_dir / f'{args.split}.lengths', 'w') as len_f:
        print('./', file=tsv_f)
        clusters = []
        offset = 0
        for line, wav_id, size in zip(lines, wav_ids, sizes): 
            seg_path = seg_dir / wav_id.replace('.wav', '_char_seg.json')            
            if seg_path.exists():
                # Load the phoneme labels and boundaries
                print(seg_path, size)
                clusts = read_segment(seg_path, size)
                clusts = encode(clusts, vocab)
                clusters.append(clusts)
                print(line, file=tsv_f)
                print(' '.join(map(str, clusts.detach().cpu().tolist())), file=seg_f)

                logits = torch.from_numpy(feat[offset:offset+size]) 

                # Compute GOP
                gop_feat = extract_gop_feature(logits, clusts)
                print(gop_feat.shape)

                # Save .npy and .lengths
                npaa.append(gop_feat)
                print(len(gop_feat), file=len_f)
            offset += size

if __name__ == '__main__':
    main()
