# -*- coding: utf-8 -*-
import argparse
import whisperx
from npy_append_array import NpyAppendArray
import os
import os.path as osp
from pathlib import Path
import torch
import tqdm
UNK = '<unk>'


def get_parser():
    parser = argparse.ArgumentParser()  
    parser.add_argument('data', help='location of .npy files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--model-name', help='which whisper model to use')
    parser.add_argument('--download-root', help='where to download the whisper model', default='/data/sls/scratch/limingw/workplace/models')
    return parser


class WhisperReader(object):
    def __init__(self, model_name, download_root):
        self.model = whisperx.load_model(model_name, device='cuda', download_root=download_root) 
        self.model_a, self.metadata = whisperx.load_align_model(language_code='en', device='cuda')
        self.fs = 16e3

    def read_audio(self, fname, sr=16e3):    
        audio = whisperx.load_audio(fname)
        audio_len = len(audio) // 320
        audio = whisperx.pad_or_trim(audio)
        mel = whisperx.log_mel_spectrogram(audio).to(self.model.device)
        return mel, audio_len

    def normalize(self, w):
        w = w.strip()
        w = ''.join([c for c in w if c.isalpha()])
        return w

    def get_feats(self, audio, audio_len):
        '''Return encoder features'''
        with torch.no_grad():
            print(audio_len)
            embed = self.model.encoder(audio.unsqueeze(0)).squeeze(0)
            embed = embed[:audio_len]
            print(embed.size())
        return embed, tran, times

    def align(self, audio, text):
        '''Align whisper output'''
        segments = [
            {
                'start': 0.0,
                'end': len(audio) / self.fs,
                'text': text,
                'words': [],
                'clean_char': [c for w in text for c in w],
            } for w in text
        ] 
        result = whisperx.align(
            segments, 
            self.model_a, 
            self.metadata, 
            audio, 'cuda', 
            return_char_alignments=True,
        )
        new_segments = result['segments']
        print(new_segments)

        # TODO Extract segments
        return new_segments

def get_iterator(args):
    # TODO Create .ltr files
    with open(osp.join(args.data, args.split) + '.tsv', 'r') as tsv_f,\
        open(osp.join(args.data, args.split) + '.ltr', 'r') as wrd_f:
        lines = tsv_f.read().split('\n')
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split('\t')[0]) for line in lines if len(line) > 0]

        lines = wrd_f.read().split('\n')
        texts = [line.split() for line in lines]

        num = len(files)
        reader = WhisperReader(args.model_name, args.download_root)

        def iterate():
            for fname, text in zip(files, texts):
                mel, audio_len = reader.read_audio(fname)
                embed = reader.get_feats(mel, audio_len)
                segments = reader.align(mel, text)
                yield embed, segments

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    def create_file(dest):
        if osp.exists(dest + '.npy'):
            os.remove(dest + '.npy')
        npaa = NpyAppendArray(dest + '.npy')
        return npaa

    npaa = create_file(save_path)
    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + '.lengths', 'w') as l_f,\
        open(save_path + '_whisper.wrd', 'w') as l_w,\
        open(save_path + '_whisper.phn', 'w') as l_p,\
        open(save_path + '_whisper.wrdseg', 'w') as l_ws:
        for embed, words, times, phns in tqdm.tqdm(iterator, total=num):
            print(len(embed), file=l_f)

            if len(embed) > 0:
                npaa.append(embed.detach().cpu().numpy())

            if not len(words):
                words = ['']
                phns = ['']
                times = [(0, 0.02)]

            print(' '.join(words), file=l_w)
            clusters = []
            for i, (start, end) in enumerate(times):
                start = int(start * 50)
                end = int(end * 50)
                dur = end - start
                clusters.extend([str(i)]*dur)
            print(len(clusters), len(embed))
            assert len(embed) >= len(clusters)
            if len(embed) > len(clusters):
                gap = len(embed) - len(clusters)
                clusters.extend([str(i+1)]*gap)
            print(' '.join(clusters), file=l_ws)
           
if __name__ == '__main__':
    main()
