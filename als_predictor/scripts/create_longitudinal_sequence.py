import argparse
from collections import defaultdict
import numpy as np
from pathlib import Path

from npy_append_array import NpyAppendArray


parser = argparse.ArgumentParser(
    description="create longitudinal sequence by combining utterance across different times for the same speaker",
)
parser.add_argument('data', help='location of tsv and input feature files')
parser.add_argument('--split', help='which split to read')
parser.add_argument('--save-dir', help='where to save the output', required=True)
parser.add_argument('--pooling', default='mean+concat')
args = parser.parse_args()
data = Path(args.data)
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# Read input speech features at each recording time
feats = np.load(data / f'{args.split}.npy')
with open(data / f'{args.split}.lengths', 'r') as f:
    sizes = f.read().strip().split('\n')
    sizes = list(map(int, sizes))
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size

# Extract longitudinal information for each speaker;
# Map each speaker to a list of tuples of (recording time, utterance idx, score)
spk_info = defaultdict(list)
with open(data / f'{args.split}.tsv', 'r') as f:
    lines = f.read().strip().split('\n')
    _ = lines.pop(0)
    keep = 0
    for idx, line in enumerate(lines):
        wav_path = Path(line.strip().split('\t')[0])
        date, label = wav_path.stem.split('_')
        # Filter out labels outside 0-4
        if 0 <= int(label) <= 4:
            spk = wav_path.parent.name
            spk_info[spk].append((date, idx, label))
            keep += 1
    print(f'{keep} out of {len(lines)} wav files have ALS progression scores')

# Combine features across recording times
if (save_dir / f'{args.split}.npy').exists():
    (save_dir / f'{args.split}.npy').unlink()
npaa = NpyAppendArray(save_dir / f'{args.split}.npy')

with open(save_dir / f'{args.split}.ids', 'w') as id_f,\
    open(save_dir / f'{args.split}.lengths', 'w') as len_f,\
    open(save_dir / f'{args.split}.score', 'w') as score_f:
    for spk in sorted(spk_info):
        id_seq = []
        feat_seq = []
        label_seq = []
        for date, idx, label in sorted(spk_info[spk]):
            offset = offsets[idx]
            size = sizes[idx]
            feat = feats[offset:offset+size]
            if args.pooling == 'mean+concat':
                feat = feat.mean(0, keepdims=True)
            id_seq.append(f'{spk}_{date}_{label}')
            feat_seq.append(feat)
            label_seq.append(label)
        npaa.append(np.concatenate(feat_seq))

        print(len(feat_seq), file=len_f)
        print(' '.join(id_seq), file=id_f)
        print(' '.join(label_seq), file=score_f)       
