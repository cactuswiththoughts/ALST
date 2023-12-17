## Extract forced alignments from wav2vec 2.0
import argparse
import json
import numpy as np
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

def decode(tgt_ids, vocab):
    tgt = [vocab[tgt_id] for tgt_id in tgt_ids]
    return ' '.join(tgt)
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='location of the metadata directory')
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

    with open(data / f'{args.split}.tsv', 'r') as tsv_f,\
        open(data / f'{args.split}.lengths', 'r') as len_f:
        paths = tsv_f.read().strip().split('\n')
        _ = paths.pop(0)
        # Convert to utt name to wav ids
        wav_ids = [convert_path_to_id(Path(l.split('\t')[0])) for l in paths] 
        lines = len_f.read().strip().split('\n')
        sizes = list(map(int, lines))

    with open(args.vocab_file, 'r') as f:
        vocab = [l.split()[0] for l in f.read().strip().split('\n')]
        blank = vocab[0]  # Assume the first symbol to be blank

    with open(save_dir / f'{args.split}.tsv', 'w') as tsv_f,\
        open(save_dir / f'{args.split}.ltr', 'w') as ltr_f,\
        open(save_dir / f'{args.split}.seg', 'w') as seg_f:
        print('./', file=tsv_f)
        for wavpath, wav_id, size in zip(paths, wav_ids, sizes): 
            seg_path = seg_dir / wav_id.replace('.wav', '_char_seg.json')            
            if seg_path.exists():
                # Load the phoneme labels and boundaries
                print(seg_path, size)
                clusts = read_segment(seg_path, size)
                clusts = encode(clusts, vocab)
                text = decode(torch.tensor(clusts).unique_consecutive().detach().tolist(), vocab)
                print(wavpath, file=tsv_f)
                print(text, file=ltr_f)
                print(' '.join(map(str, clusts.detach().cpu().tolist())), file=seg_f)

if __name__ == '__main__':
    main() 
