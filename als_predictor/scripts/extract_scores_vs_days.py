import argparse
from datetime import datetime
import numpy as np
import pandas as pd

# Given a sequence of wav_ids, extract the ALS progression
# score and the number of days from its date to the first recording
def date_str_to_ymd(date):
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    return year, month, day

def date_distance(date1, date2):
    y1, m1, d1 = date_str_to_ymd(date1)
    y2, m2, d2 = date_str_to_ymd(date2)
    delta = datetime(y2, m2, d2) - datetime(y1, m1, d1)
    return delta.days

parser = argparse.ArgumentParser()
parser.add_argument('--in-path')
parser.add_argument('--score-path', default='')
parser.add_argument('--out-path')
args = parser.parse_args()

data_dict = {
    'Patient ID': [], 'Days': [], 'Score': [],
}

scores = None
if args.score_path:
    scores = np.load(args.score_path)
    print(scores)

with open(args.in_path, 'r') as f_in:
    offset = 0
    for l in f_in:
        wav_ids = l.split()
        start_date = ''
        for i, wav_id in enumerate(wav_ids):
            _, patient, date, score = wav_id.split('_')
            if scores is not None:
                score = scores[offset+i]

            if not start_date:
                start_date = date
                n_days = 0
            else:
                n_days = date_distance(start_date, date)
            
            data_dict['Patient ID'].append(patient)
            data_dict['Days'].append(n_days)
            data_dict['Score'].append(int(score))
        offset += len(wav_ids)

if scores is not None:
    assert len(scores) == len(data_dict['Score'])

df = pd.DataFrame(data_dict)
df.to_csv(args.out_path)
