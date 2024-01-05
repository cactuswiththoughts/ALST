import argparse
from pathlib import Path
from praatio.textgrid import openTextgrid

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir")
    parser.add_argument("--align_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--split")
    parser.add_argument("--sr", type=int, default=16e3)
    parser.add_argument("--ds_factor", type=int, default=2)
    return parser

def convert_path_to_id(wav_path):
    spk = wav_path.parent.name
    date = wav_path.stem.split('_')[0]
    year = date[:4]
    month_day = date[4:8]
    hr_min_sec = date[8:]
    new_date = month_day+year+hr_min_sec
    wav_id = f'{spk}_{new_date}_recording'
    return wav_id

def read_alignment(align_path):
    starts = []
    ends = []
    labels = []
    try:
        tg = openTextgrid(align_path, includeEmptyIntervals=True)
    except:
        raise RuntimeError(f"Failed to open {align_path}")
    wrds = tg._tierDict["words"].entries
    phns = tg._tierDict["phones"].entries
    return wrds, phns

def main():
    parser = get_parser()
    args = parser.parse_args()
    manifest_dir = Path(args.manifest_dir)
    align_dir = Path(args.align_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(manifest_dir / f"{args.split}.tsv", "r") as f_tsv,\
        open(manifest_dir / f"{args.split}.lengths", "r") as f_len,\
        open(out_dir / f"{args.split}.phn", "a") as f_phn,\
        open(out_dir / f"{args.split}.seg", "w") as f_seg:
        sizes = f_len.read().strip().split('\n')
        sizes = list(map(int, sizes))

        lines = f_tsv.read().strip().split("\n")
        _ = lines.pop(0)
        for l, size in zip(lines, sizes):
            wav_path = Path(l.split("\t")[0])
            wav_id = convert_path_to_id(wav_path)
            align_path = align_dir / f"{wav_id}.TextGrid"
            if not align_path.exists():
                print(f'{align_path} not found')
                phn_labels = ["<UNK>"]
                phn_segs = ["0"]*size
            else:
                count += 1
                words, phones = read_alignment(align_path)
                phn_labels = []
                phn_segs = []
                for i, (start, end, label) in enumerate(phones):
                    s = int(start * 100)
                    e = int(end * 100)
                    if not label:
                        label = "<SIL>"
                    phn_labels.append(label)
                    phn_segs.extend([str(i)]*(e-s))
                phn_segs = phn_segs[::args.ds_factor]
            print(" ".join(phn_labels), file=f_phn)
            print(" ".join(phn_segs), file=f_seg)
        print(f"Found {count} of {len(sizes)} files with phoneme alignments")

if __name__ == "__main__":
    main()
