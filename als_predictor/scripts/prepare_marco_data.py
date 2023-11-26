import argparse
import pandas as pd
from pathlib import Path


def convert_date(date):
    month_day = date[0:4]
    year = date[4:8]
    hr_min_sec = date[8:]
    new_date = year+month_day+hr_min_sec
    return new_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', help='Path to the metadata directory')
    parser.add_argument('--out-dir', help='Path to the manifest directory')
    parser.add_argument('--wav-dir', help='Path to the wav file directory')
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir)

    df = pd.read_csv(in_dir / 'complet_meta_11_07_2023.csv').to_dict()
    utt_ids_all = sorted(df['ID'])

    for x in ['train', 'test']:
        with open(in_dir / f'{x}_indices.txt', 'r') as f:
            lines = f.read().strip().split('\n')
            utt_ids = list(map(float, lines))
            utt_ids = list(map(int, utt_ids))

        with open(out_dir / f'{x}.tsv', 'w') as f_tsv,\
            open(out_dir / f'{x}.wrd', 'w') as f_wrd:
            print('./', file=f_tsv)
            found = 0
            for utt_id in utt_ids:
                if utt_id not in utt_ids_all:
                    print(f'ID {utt_id} not found')
                    continue
                i = utt_ids_all.index(utt_id)
                patient_id, raw_date = df['filename'][i].split('_')[1:3]
                date = convert_date(raw_date) 
                score = df['Speech'][i]
                text = df['text'][i]
                wav_path = wav_dir / f'Patient_{patient_id}' / f'{date}_{score}.wav'

                if not wav_path.exists():
                    print(f'{wav_path} not found')
                    continue
                found += 1
                print(wav_path, file=f_tsv)
                print(text, file=f_wrd)

        print(f'Found {found} out of {len(utt_ids)} {x} wav files')

if __name__ == '__main__':
    main()
