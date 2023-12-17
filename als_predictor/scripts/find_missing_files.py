import json
import os
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

wav_dir = '/data/sls/scratch/yuangong/dataset/ALS/Voice_Recordings_16k'
reord_wav_dir = '/data/sls/scratch/yuangong/dataset/ALS/Voice_Recordings_16k_reorder'
wav_ids = list(os.listdir(wav_dir))
reord_wav_ids = []
for root, dirs, files in os.walk(reord_wav_dir):
    for fn in files:
        if fn.endswith('.wav'):
            fpath = Path(root) / fn
            wav_id = convert_path_to_id(fpath)
            reord_wav_ids.append(wav_id)

missing_all = []
with open('missing.txt', 'w') as fp:
    for wav_id in wav_ids:
        if not wav_id in reord_wav_ids:
            missing_all.append(wav_id)

    print('\n'.join(sorted(missing_all)), file=fp)

print(f'Found {len(reord_wav_ids)} out of {len(wav_ids)} files')

'''
split_dir = Path('/data/sls/scratch/limingw/workplace/ALS/data/als')
for x in ['train', 'val', 'test']:
    with open(split_dir / f'voice_{x}_files.txt', 'r') as f_in,\
        open(split_dir / f'{x}_missing_files.txt', 'w') as f_out,\
        open(split_dir / f'{x}_missing_files_reord.txt', 'w') as f_out_reord:
        missing = []
        missing_reord = []
        for wav_id in f_in:
            wav_id = wav_id.strip()
            if not wav_id in wav_ids:
                missing.append(wav_id)
            if not wav_id in reord_wav_ids: 
                missing_reord.append(wav_id)

        print('\n'.join(sorted(missing)), file=f_out)
        print('\n'.join(sorted(missing_reord)), file=f_out_reord)
'''

label_dir = Path('/data/sls/scratch/yuangong/dataset/ALS/Metadata')
score_path = label_dir / 'all ALSFRS-R Data 09302022.xlsx'
score_dict = pd.read_excel(score_path)
wav_ids_with_score = ['-'.join([str(ide), str(date).split()[0]]) for ide, date in zip(score_dict['ID'], score_dict['Entry Date'])]
print(wav_ids_with_score[:10])
missing_score = []
with open('missing_score.txt', 'w') as fp:
    for wav_id in missing_all:
        parts = wav_id.split('_')
        if len(parts) < 4:
            missing_score.append(wav_id)
            continue
        _, patient_id, date, _ = parts
        month = date[:2]
        day = date[2:4]
        year = date[4:8]
        new_wav_id = '-'.join([patient_id, year, month, day])
        if not new_wav_id in wav_ids_with_score:
            missing_score.append(wav_id)
    
    print('\n'.join(sorted(missing_score)), file=fp)

