import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path


def convert_path_to_id(wav_path):
    spk = wav_path.parent.name
    date = wav_path.stem.split('_')[0]
    year = date[:4]
    month_day = date[4:8]
    hr_min_sec = date[8:]
    new_date = month_day+year+hr_min_sec
    wav_id = f'{spk}_{new_date}_recording.wav'
    return wav_id

def normalize(text):
    return ''.join([c.upper() for c in text if c.isalpha() or (c == ' ')])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', help='Path to the metadata directory')
    parser.add_argument('--out-dir', help='Path to the manifest directory')
    parser.add_argument('--wav-dir', help='Path to the wav file directory')
    parser.add_argument('--label-dir', help='Path to the label directory')
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = Path(args.wav_dir)
    score_path = Path(args.label_dir) / 'all ALSFRS-R Data 09302022.xlsx' 
    text_path = Path(args.label_dir) / 'ALSTDI Voice Recording Phrases.xlsx'

    text_dict = pd.read_excel(text_path).to_dict()
    all_wav_ids = text_dict['File Wav']
    phrases = text_dict['Phrase Text']
    phrase_order = text_dict['Phrase Order']
    utt2txt = defaultdict(dict)
    utt2score = {}
    for idx, k in enumerate(sorted(all_wav_ids)):
        wav_id = all_wav_ids[k]
        if not isinstance(wav_id, str):
            continue
        j = phrase_order[k]
        label = phrases[k]
        label = normalize(label)
        utt2txt[wav_id][j] = label 

    # Read .wav file paths
    utt2wav = {}
    for spk_dir in wav_dir.iterdir():
        spk = spk_dir.name
        for fn in spk_dir.iterdir():
            wav_id = convert_path_to_id(fn)
            score = int(fn.stem.split('_')[-1])
            if 0 <= score <= 4:  # ALS scores are between 0-4
                utt2wav[wav_id] = fn
    print(f'{len(utt2wav)} .wav files in total')

    # Create .tsv and .wrd files
    for x in ['train', 'val', 'test']:
        with open(in_dir / f'voice_{x}_files.txt', 'r') as f_in,\
            open(out_dir / f'{x}.tsv', 'w') as f_tsv,\
            open(out_dir / f'{x}.wrd', 'w') as f_wrd:
            wav_ids = f_in.read().strip().split('\n')
            print('./', file=f_tsv)

            found = 0
            spks = []
            for wav_id in wav_ids:
                wav_id = wav_id.strip()
                if wav_id not in utt2wav:
                    # print(f'ID {wav_id} not found')
                    # print(wav_id)
                    continue
                wav_path = utt2wav[wav_id]
                text = ''
                if wav_id in utt2txt:
                    text = ' '.join(
                        [utt2txt[wav_id][k] for k in sorted(utt2txt[wav_id])]
                    )
                found += 1

                spk = wav_id.split('_')[1]
                if not spk in spks:
                    spks.append(spk)

                print(wav_path, file=f_tsv)
                print(text, file=f_wrd)

        print(f'Found {found} out of {len(wav_ids)} {x} wav files from {len(spks)} patients')

if __name__ == '__main__':
    main()
