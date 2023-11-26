import argparse
import soundfile as sf
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--save-path')
    args = parser.parse_args()

    data = Path(args.data)

    dur_dict = {
        'Split': [],
        'Duration': [],
    }
    for split in ['train', 'test']:
        tsv_path = data / f'{split}.tsv'
        with open(tsv_path, 'r') as tsv_f:
            lines = tsv_f.read().strip().split('\n')
            root = Path(lines.pop(0))
            for l in lines:
                fname = l.strip().split('\t')[0]
                wav, sr = sf.read(root / fname)
                dur = len(wav) / sr
                dur_dict['Split'].append(split)
                dur_dict['Duration'].append(dur)
    print('#training hours: {}'.format(sum([l for s, l in zip(dur_dict['Split'], dur_dict['Duration']) if s == 'train']) / 3600))
    print('#test hours: {}'.format(sum([l for s, l in zip(dur_dict['Split'], dur_dict['Duration']) if s == 'test']) / 3600))
    df = pd.DataFrame(dur_dict)
    df.to_csv(args.save_path)

if __name__ == '__main__':
    main()    
