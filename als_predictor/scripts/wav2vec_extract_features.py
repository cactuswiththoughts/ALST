#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from pathlib import Path
from shutil import copyfile

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layers', type=str, default='14', help='which layers to use')
    # fmt: on

    return parser


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layers):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file], arg_overrides={'data': Path(cp_file).parent, 'task': 'audio_finetuning'},
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layers = layers

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        if len(x) > 480000:
#            print(loc, x.shape)
            x = x[:480000]  # Trim the audio at 30 seconds

        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, padding_mask=None, mask=False, features_only=True)
#            return m_res["x"].squeeze(0).cpu()
            
            feats = []
            for layer in self.layers:
                if layer == -1:
                    feats.append(m_res["encoder_out"].squeeze(1).cpu())
                else:
                    feats.append(m_res["layer_results"][layer][0].squeeze(0).cpu())
            return feats


def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        num = len(files)
        layers = list(map(int, args.layers.split(',')))
        reader = Wav2VecFeatureReader(args.checkpoint, layers)

        def iterate():
            for fname in files:
                w2v_feats = reader.get_feats(fname)
                yield w2v_feats

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    layers = list(map(int, args.layers.split(',')))
    print(layers)

    for l in layers:
        os.makedirs(osp.join(args.save_dir, f'layer{l}'), exist_ok=True)

    def create_files(dest):
        
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if osp.exists(osp.join(args.data, args.split) + ".wrd"):
            copyfile(osp.join(args.data, args.split) + ".wrd", dest + ".wrd")
        if osp.exists(osp.join(args.data, args.split) + ".phn"):
            copyfile(osp.join(args.data, args.split) + ".phn", dest + ".phn")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_paths = [osp.join(args.save_dir, f'layer{l}', args.split) for l in layers]
    npaas = [create_files(save_path) for save_path in save_paths]
    l_fs = [open(save_path + ".lengths", "w") for save_path in save_paths]

    generator, num = get_iterator(args)
    iterator = generator()

    for w2v_feats_list in tqdm.tqdm(iterator, total=num):
        for w2v_feats, npaa, l_f in zip(w2v_feats_list, npaas, l_fs): 
            print(len(w2v_feats), file=l_f)

            if len(w2v_feats) > 0:
                npaa.append(w2v_feats.numpy())

    for l_f in l_fs:
        l_f.close()

if __name__ == "__main__":
    main()
