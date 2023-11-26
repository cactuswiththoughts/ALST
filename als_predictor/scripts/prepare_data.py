import argparse
from collections import defaultdict
import numpy as np
import os
import pandas as pd
from pathlib import Path


def convert_path_to_id(wav_path):
    spk = wav_path.parent.name
    date = wav_path.stem.split('_')[0]
    year = date[0:4]
    month_day = date[4:8]
    hr_min_sec = date[8:]
    new_date = month_day+year+hr_min_sec
    wav_id = f'{spk}_{new_date}_recording.wav'
    return wav_id

def normalize(text):
    return ''.join([c.upper() for c in text if c.isalpha() or (c == ' ')])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_dir')
    parser.add_argument('label_dir')
    parser.add_argument('out_dir')
    parser.add_argument(
        '--with-labels', choices={'all', 'text', 'score', 'score_text'},
        help='keep only utterances with labels',
    )

    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    label_path = Path(args.label_dir) / 'ALSTDI Voice Recording Phrases.xlsx'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read speech files
    spk2utt = defaultdict(list)
    utt2score = {}
    count = 0
    for spk_dir in wav_dir.iterdir():
        spk = spk_dir.name
        for fn in spk_dir.iterdir():
            wav_id = convert_path_to_id(fn)
            score = int(fn.stem.split('_')[-1])
            if 0 <= score <= 4:  # ALS scores are between 0-4
                utt2score[wav_id] = score
            if count < 10:
                print(wav_id)  # XXX
            count += 1
            spk2utt[spk].append(fn)

    # Read speech transcripts
    label_dict = pd.read_excel(label_path).to_dict()
    all_wav_ids = label_dict['File Wav']
    phrases = label_dict['Phrase Text']
    phrase_order = label_dict['Phrase Order']
    utt2text = defaultdict(dict)
    for idx, k in enumerate(sorted(all_wav_ids)):
        wav_id = all_wav_ids[k]
        if not isinstance(wav_id, str):
            continue
        if idx < 10:
            print('utt2text key:', wav_id)  # XXX
        j = phrase_order[k]
        label = phrases[k]
        label = normalize(label)
        utt2text[wav_id][j] = label

    # Create wav.scp, text and utt2spk files
    with open(out_dir / 'wav.scp', 'w') as wavscp_f,\
        open(out_dir / 'text', 'w') as txt_f,\
        open(out_dir / 'utt2spk', 'w') as utt2spk_f:
        idx = 0
        for spk in sorted(spk2utt):
            for wav_path in spk2utt[spk]:
                wav_id = convert_path_to_id(wav_path)            
                if ('text' in args.with_labels) and (wav_id not in utt2text):
                    continue

                if ('score' in args.with_labels) and (wav_id not in utt2score):
                    continue

                trans = utt2text[wav_id]
                trans = [trans[order] for order in sorted(trans)]
                tran = ' '.join(trans)
                if idx < 10:
                    print(tran)

                utt_id = f'utt{idx:010d}'
                print(' '.join([utt_id, str(wav_path)]), file=wavscp_f)
                print(' '.join([utt_id, tran]), file=txt_f)
                print(' '.join([utt_id, spk]), file=utt2spk_f)
                idx += 1
        print(f'Found {idx} out of {count} wav files with {args.with_labels}')

if __name__ == '__main__':
    main()
