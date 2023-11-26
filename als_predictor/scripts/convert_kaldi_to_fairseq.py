import argparse
from pathlib import Path
import random
random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', help='Path to the Kaldi directory')
    parser.add_argument('--out-dir', help='Path to the manifest directory')
    parser.add_argument('--test-speaker-ratio', type=float, help='ratio of test speakers')
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    with open(in_dir / 'wav.scp', 'r') as wavscp,\
        open(in_dir / 'text', 'r') as txt_f,\
        open(in_dir / 'utt2spk', 'r') as utt2spk_f:
        lines = wavscp.read().strip().split('\n')
        utt2wav = {l.split()[0]:l.split()[1] for l in lines}

        lines = txt_f.read().strip().split('\n') 
        utt2txt = {l.split()[0]:' '.join(l.split()[1:]) for l in lines}

        lines = utt2spk_f.read().strip().split('\n')
        utt2spk = {l.split()[0]:l.split()[1] for l in lines}
        spks = list(set(utt2spk.values()))

        test_spks = []
        if args.test_speaker_ratio:
            n_spk = len(spks)
            n_test_spk = int(n_spk * args.test_speaker_ratio)
            test_spks = random.sample(spks, n_test_spk)
            print(f'Number of test/total speakers: {len(test_spks)}/{len(spks)}')

    i_tr = 0
    i_te = 0
    with open(out_dir / 'train.tsv', 'w') as tr_tsv_f,\
        open(out_dir / 'train.wrd', 'w') as tr_wrd_f,\
        open(out_dir / 'test.tsv', 'w') as te_tsv_f,\
        open(out_dir / 'test.wrd', 'w') as te_wrd_f:
        print('./', file=tr_tsv_f)
        print('./', file=te_tsv_f)
        for utt_id in sorted(utt2wav):
            if utt2spk[utt_id] in test_spks:
                print(utt2wav[utt_id], file=te_tsv_f)
                print(utt2txt[utt_id], file=te_wrd_f)
                i_te += 1
            else:
                print(utt2wav[utt_id], file=tr_tsv_f)
                print(utt2txt[utt_id], file=tr_wrd_f)
                i_tr += 1
    print(f'Number of train/test utterances: {i_tr}/{i_te}')

if __name__ == '__main__':
    main()
