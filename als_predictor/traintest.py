# -*- coding: utf-8 -*-
# Modified from gopt: https://github.com/YuanGongND/gopt

# train and test the models
import argparse
import numpy as np
import sys
import os
import os.path as osp
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
from dataloaders import * 

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-dir", type=str, default="../../data/als/w2v2_big_960h_layer14", help="directory containing extracted features and labels")
parser.add_argument("--layers", type=str, default="14")
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--embed_dim", type=int, default=512, help="predictor embedding dimension")
parser.add_argument("--depth", type=int, default=1, help="depth of the predictor")
parser.add_argument("--am", type=str, default='wav2vec2_big_960h')
parser.add_argument("--model", type=str, default='alst')
parser.add_argument("--mse_weight", type=float, default=0.0)
parser.add_argument("--no-longitudinal", action='store_true', help='not use longitudinal information')
parser.add_argument("--use-phn-label", action='store_true', help='use phoneme labels as inputs')

# just to generate the header for the result.csv
def gen_result_header():
    header = ['epoch', 'train_loss', 'train_precision', 'train_recall', 'train_micro_f1', 'train_macro_f1', 'test_loss', 'test_precision', 'test_recall', 'test_micro_f1', 'test_macro_f1', 'best_macro_f1']
    return header 

def length_to_mask(length, max_len):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    device = length.device
    dtype = length.dtype 
    mask = torch.arange(
        max_len, device=device, dtype=dtype
    ).repeat(len(length), 1) < length.unsqueeze(1)
    return mask

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    best_epoch, best_loss, best_macro_f1 = 0, np.inf, 0
    global_step, epoch = 0, 0
    exp_dir = Path(args.exp_dir)
    (exp_dir / 'models').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'preds').mkdir(parents=True, exist_ok=True)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3), flush=True)
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3), flush=True)
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)
 
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss_fn = nn.MSELoss()
    mse_weight = args.mse_weight

    print("current #steps=%s, #epochs=%s" % (global_step, epoch), flush=True)
    print("start training...", flush=True)
    result = np.zeros([args.n_epochs, 12])
    
    for epoch in range(args.n_epochs):
        audio_model.train()
        for i, (audio_input, als_label, sizes, label_sizes, pool_mask, phn_label) in enumerate(train_loader):
            audio_input = audio_input.to(device, non_blocking=True)
            als_label = als_label.to(device, non_blocking=True)
            sizes = sizes.to(device, non_blocking=True)
            label_sizes = label_sizes.to(device, non_blocking=True)
            pool_mask = pool_mask.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            if not args.use_phn_label:
                phn_label = None

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']), flush=True)

            mask = length_to_mask(sizes, max_len=max(sizes))
            label_mask = length_to_mask(label_sizes, max_len=max(label_sizes))
            if args.model == 'alst_encdec':
                logits = audio_model(audio_input, als_label, mask=mask, pool_mask=pool_mask, phns=phn_label)
            else:
                logits = audio_model(audio_input, mask=mask, pool_mask=pool_mask, phns=phn_label)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), als_label.flatten())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('start validation', flush=True)
        tr_loss, tr_prec, tr_rec, tr_micro_f1, tr_macro_f1, tr_confusion = validate(audio_model, train_loader, args, -1)
        te_loss, te_prec, te_rec, te_micro_f1, te_macro_f1, te_confusion = validate(audio_model, test_loader, args, best_loss)
        if te_macro_f1 < best_macro_f1:
            best_loss = te_loss
            best_macro_f1 = te_macro_f1
            best_epoch = epoch
            torch.save(audio_model.state_dict(), exp_dir / 'models/best_audio_model.pth')

        print('Precision: {:.3f}, Recall: {:.3f}, Micro F1: {:.3f}, Macro F1: {:.3f}, Best Macro F1: {:.3f}'.format(te_prec, te_rec, te_micro_f1, te_macro_f1, best_macro_f1), flush=True)

        result[epoch, :6] = [epoch, tr_loss, tr_prec, tr_rec, tr_micro_f1, tr_macro_f1]
        result[epoch, 6:12] = [te_loss, te_prec, te_rec, te_micro_f1, te_macro_f1, best_macro_f1] 

        header = ','.join(gen_result_header())
        np.savetxt(exp_dir / 'result.csv', result, fmt='%.4f', delimiter=',', header=header, comments='')
        np.savetxt(exp_dir / 'confusion.csv', te_confusion, fmt='%d', delimiter=',')
        print('-------------------validation finished-------------------', flush=True)

        if global_step > warm_up_step:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']), flush=True)


def validate(audio_model, val_loader, args, best_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    confusion = np.zeros([5, 5])
    pred_labels = []
    gold_labels = []
    exp_dir = Path(args.exp_dir)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        loss = 0.0
        for i, (audio_input, als_label, sizes, label_sizes, pool_mask, phn_label) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            als_label = als_label.to(device)
            sizes = sizes.to(device, non_blocking=True)
            label_sizes = label_sizes.to(device, non_blocking=True)
            pool_mask = pool_mask.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            if not args.use_phn_label:
                phn_label = None

            mask = length_to_mask(sizes, max_len=max(sizes))
            label_mask = length_to_mask(label_sizes, max_len=max(label_sizes))

            if args.model == 'alst_encdec':
                pred_label = audio_model(
                    audio_input, max_len=als_label.size(1), mask=mask,
                    pool_mask=pool_mask, phns=phn_label,
                )
            else:
                logits = audio_model(audio_input, mask=mask, pool_mask=pool_mask, phns=phn_label)
                pred_label = logits.argmax(-1)
                # loss += loss_fn(logits.view(-1, logits.size(-1)), als_label).cpu().detach().item()

            pred_label = pred_label.flatten()
            gold_label = als_label.flatten()
            pred_labels.extend(pred_label.detach().cpu().tolist())
            gold_labels.extend(gold_label.detach().cpu().tolist())

        pred_labels = np.asarray(pred_labels)
        gold_labels = np.asarray(gold_labels) 
        pred_labels = pred_labels[gold_labels != -1]
        gold_labels = gold_labels[gold_labels != -1]

        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='micro',
        )
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='macro',
        )
        confusion = confusion_matrix(gold_labels, pred_labels)

        loss /= len(val_loader)        
        if not (exp_dir / 'preds' / 'gold_als_label.npy').exists():
            np.save(exp_dir / 'preds' / 'gold_als_label.npy', gold_labels)
        np.save(exp_dir / 'preds' / 'pred_als_label.npy', pred_labels)

    return loss, precision, recall, micro_f1, macro_f1, confusion

args = parser.parse_args()

print(f'Layer {args.layers}')
data_dirs = [osp.join(args.data_dir, f'layer{l}') for l in args.layers.split(',')]
am = args.am
feat_dim = {
    'w2v2_big_960h': 1024, 
    'w2v2_small_960h': 512,
    'whisper_large-v1': 1280,
    'whisper_large-v2': 1280,
    'whisper_medium': 1024, 
    'whisper_base': 512,
}
input_dim = feat_dim[am]
n_weights = len(data_dirs)

tr_dataset = ALSFeatureDataset([osp.join(data_dir, 'train') for data_dir in data_dirs], longitudinal=(not args.no_longitudinal))
tr_dataloader = DataLoader(tr_dataset, collate_fn=tr_dataset.collater, batch_size=args.batch_size, shuffle=True)
te_dataset = ALSFeatureDataset([osp.join(data_dir, 'test') for data_dir in data_dirs], longitudinal=(not args.no_longitudinal))
te_dataloader = DataLoader(te_dataset, collate_fn=te_dataset.collater, batch_size=50, shuffle=False)
max_len = max(tr_dataset.max_size, te_dataset.max_size)
print(f'maximal sequence length: {max_len}')

if args.model == 'alst_encdec':
    audio_model = ALSEncDecTransformer(embed_dim=args.embed_dim, depth=args.depth, input_dim=input_dim, max_len=max_len, n_weights=n_weights)
elif args.model == 'linear':
    audio_model = ALSLinear(input_dim, n_weights=n_weights)
else:
    audio_model = ALSTransformer(embed_dim=args.embed_dim, depth=args.depth, input_dim=input_dim, max_len=max_len, n_weights=n_weights)
print(args.model)
print(audio_model)

train(audio_model, tr_dataloader, te_dataloader, args)
