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
from sklearn.preprocessing import LabelBinarizer
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.compute_rank_scores import compute_rank_scores
from scripts.compute_auc import compute_auc
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
from dataloaders import * 

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", choices={'train', 'eval'}, default='train')
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
parser.add_argument("--ce_weight", type=float, default=1.0)
parser.add_argument("--mse_weight", type=float, default=0.0)
parser.add_argument("--no-longitudinal", action='store_true', help='not use longitudinal information')
parser.add_argument("--use-phn-label", action='store_true', help='use phoneme labels as inputs')

# just to generate the header for the result.csv
def gen_result_header():
    header = [
        'epoch', 'train_loss', 
        'train_precision', 
        'train_recall',
        'train_micro_f1',
        'train_macro_f1',
        'test_loss',
        'test_precision',
        'test_recall',
        'test_micro_f1',
        'test_macro_f1',
        'best_macro_f1',
        'test_spearmanr', 
        'test_kendalltau',
        'test_pairwise_acc',
        'test_auc',
        'test_precision_mse',
        'test_recall_mse',
        'test_micro_f1_mse', 
        'test_macro_f1_mse', 
        'best_macro_f1_mse', 
        'test_spearmanr_mse', 
        'test_kendalltau_mse',
        'test_pairwise_acc_mse',
    ]
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

    best_epoch, best_loss, best_macro_f1, best_macro_f1_mse = 0, np.inf, 0, 0
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

    print("current #steps=%s, #epochs=%s" % (global_step, epoch), flush=True)
    print("start training...", flush=True)
    result = np.zeros([args.n_epochs, 24])
    
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
            scores = None
            if args.model == 'alst_encdec':
                logits_and_scores = audio_model(audio_input, als_label, mask=mask, pool_mask=pool_mask, phns=phn_label)
                logits = logits_and_scores[:, :, :-1]
                scores = logits_and_scores[:, :, -1]
            else:
                logits_and_scores = audio_model(audio_input, mask=mask, pool_mask=pool_mask, phns=phn_label)
                logits = logits_and_scores[:, :, :-1]
                scores = logits_and_scores[:, :, -1]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), als_label.flatten())
            loss = args.ce_weight * loss
            if scores is not None:
                mse_loss = mse_loss_fn(scores, als_label.float())
                mse_loss = args.mse_weight * mse_loss
                loss += mse_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('start validation', flush=True)
        tr_loss,\
        (tr_prec, tr_prec_mse),\
        (tr_rec, tr_rec_mse),\
        (tr_micro_f1, tr_micro_f1_mse),\
        (tr_macro_f1, tr_macro_f1_mse),\
        _, _, _, _, tr_confusion = validate(audio_model, train_loader, args, -1)
        te_loss,\
        (te_prec, te_prec_mse),\
        (te_rec, te_rec_mse),\
        (te_micro_f1, te_micro_f1_mse),\
        (te_macro_f1, te_macro_f1_mse),\
        (te_spearmanr, te_spearmanr_mse),\
        (te_kendalltau, te_kendalltau_mse),\
        (te_pairwise_acc, te_pairwise_acc_mse),\
        te_auc, te_confusion = validate(audio_model, test_loader, args, best_loss)
        if te_macro_f1 > best_macro_f1:
            best_loss = te_loss
            best_macro_f1 = te_macro_f1
            best_epoch = epoch
            torch.save(audio_model.state_dict(), exp_dir / 'models/best_audio_model.pth')

        if te_macro_f1_mse > best_macro_f1_mse:
            best_macro_f1_mse = te_macro_f1_mse
            torch.save(audio_model.state_dict(), exp_dir / 'models/best_audio_model_mse.pth')

        print(f'Precision: {te_prec:.3f}, Recall: {te_rec:.3f}, Micro F1: {te_micro_f1:.3f}, Macro F1: {te_macro_f1:.3f}, AUC: {te_auc:.3f}, Spearman Corr.: {te_spearmanr:.3f}, Kendall Corr.: {te_kendalltau:.3f}, Pairwise Acc.: {te_pairwise_acc:.3f}, Best Macro F1: {best_macro_f1:.3f}', flush=True)
        print(f'(MSE) Precision: {te_prec_mse:.3f}, Recall: {te_rec_mse:.3f}, Micro F1: {te_micro_f1_mse:.3f}, Macro F1: {te_macro_f1_mse:.3f}, Spearman Corr.: {te_spearmanr_mse:.3f}, Kendall Corr.: {te_kendalltau_mse:.3f}, Pairwise Acc.: {te_pairwise_acc_mse:.3f}, Best Macro F1: {best_macro_f1_mse:.3f}', flush=True)

        result[epoch, :6] = [epoch, tr_loss, tr_prec, tr_rec, tr_micro_f1, tr_macro_f1]
        result[epoch, 6:16] = [te_loss, te_prec, te_rec, te_micro_f1, te_macro_f1, best_macro_f1, te_spearmanr, te_kendalltau, te_pairwise_acc, te_auc]
        result[epoch, 16:24] = [te_prec_mse, te_rec_mse, te_micro_f1_mse, te_macro_f1_mse, best_macro_f1_mse, te_spearmanr_mse, te_kendalltau_mse, te_pairwise_acc_mse]

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
    pred_probs = []
    pred_scores = []
    pred_labels_mse = []
    gold_labels = []
    label_size_list = []
    exp_dir = Path(args.exp_dir)

    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        mse_loss = 0.0
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
            
            logits = None
            scores = None
            if args.model == 'alst_encdec':
                pred_label, scores = audio_model(
                    audio_input, max_len=als_label.size(1), mask=mask,
                    pool_mask=pool_mask, phns=phn_label,
                )
                pred_label_mse = torch.maximum(
                    torch.minimum(scores.round(), torch.tensor(4)),
                    torch.tensor(0),
                )
            else:
                logits_and_scores = audio_model(audio_input, mask=mask, pool_mask=pool_mask, phns=phn_label)
                logits = logits_and_scores[:, :, :-1]
                scores = logits_and_scores[:, :, -1]
                pred_label = logits.argmax(-1)
                pred_label_mse = torch.maximum(
                    torch.minimum(scores.round(), torch.tensor(4)),
                    torch.tensor(0),
                )
                mse_loss += mse_loss_fn(scores, als_label.float()).detach().cpu().numpy()

            pred_label = pred_label.flatten()
            gold_label = als_label.flatten()
            pred_labels.extend(pred_label.detach().cpu().tolist())
            gold_labels.extend(gold_label.detach().cpu().tolist())
            label_size_list.extend(label_sizes.detach().cpu().tolist())
            if logits is not None: 
                pred_prob = torch.softmax(logits, dim=-1).reshape(-1, 5)
                pred_probs.extend(pred_prob.detach().cpu().tolist())

            if scores is not None:
                scores = scores.flatten()
                pred_scores.extend(scores.detach().cpu().tolist())
                pred_label_mse = pred_label_mse.flatten()
                pred_labels_mse.extend(pred_label_mse.detach().cpu().tolist())

        pred_labels = np.asarray(pred_labels)
        gold_labels = np.asarray(gold_labels)
        keep = gold_labels != -1
        pred_labels = pred_labels[keep]
        gold_labels = gold_labels[keep]
        if len(pred_scores):
            pred_scores = np.asarray(pred_scores)
            pred_scores = pred_scores[keep]

        auc = 0
        if len(pred_probs):
            pred_probs = np.asarray(pred_probs)
            pred_probs = pred_probs[keep]
            auc = compute_auc(gold_labels, pred_probs)

        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='micro',
        )
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            gold_labels, pred_labels, average='macro',
        )
        spearmanr, kendalltau, pairwise_acc = compute_rank_scores(
            gold_labels, pred_labels, label_size_list)
        confusion = confusion_matrix(gold_labels, pred_labels)

        mse_loss /= len(val_loader)
        #if not (exp_dir / 'preds' / 'gold_als_label.npy').exists():
        np.save(exp_dir / 'preds' / 'gold_als_label.npy', gold_labels)
        np.save(exp_dir / 'preds' / 'pred_als_label.npy', pred_labels)
        if len(pred_scores):
            np.save(exp_dir / 'preds' / 'pred_als_scores.npy', pred_scores)

        micro_f1_mse, precision_mse, recall_mse, macro_f1_mse, spearmanr_mse, kendalltau_mse, pairwise_acc_mse = 0., 0., 0., 0., 0., 0., 0.
        if len(pred_labels_mse):
            pred_labels_mse = np.asarray(pred_labels_mse)
            pred_labels_mse = pred_labels_mse[keep]
            _, _, micro_f1_mse, _ = precision_recall_fscore_support(
                gold_labels, pred_labels_mse, average='micro',
            )
            precision_mse, recall_mse, macro_f1_mse, _ = precision_recall_fscore_support(
                gold_labels, pred_labels_mse, average='macro',
            )
            spearmanr_mse, kendalltau_mse, pairwise_acc_mse = compute_rank_scores(
                gold_labels, pred_labels_mse, label_size_list)
            if macro_f1_mse > macro_f1:
                confusion = confusion_matrix(gold_labels, pred_labels_mse)

            np.save(exp_dir / 'preds' / 'pred_als_label_mse.npy', pred_labels)
    np.savetxt(exp_dir / 'confusion.csv', confusion, fmt='%d', delimiter=',')
    return mse_loss, (precision, precision_mse), (recall, recall_mse), (micro_f1, micro_f1_mse), (macro_f1, macro_f1_mse), (spearmanr, spearmanr_mse), (kendalltau, kendalltau_mse), (pairwise_acc, pairwise_acc_mse), auc, confusion

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
    audio_model = ALSEncDecTransformer(embed_dim=args.embed_dim, depth=args.depth, input_dim=input_dim, max_len=max_len, n_weights=n_weights, output_type='both')
elif args.model == 'linear':
    audio_model = ALSLinear(input_dim, n_weights=n_weights)
else:
    audio_model = ALSTransformer(embed_dim=args.embed_dim, depth=args.depth, input_dim=input_dim, max_len=max_len, n_weights=n_weights, output_type='both')
print(args.model)
print(audio_model)

if args.mode == 'train':
    train(audio_model, tr_dataloader, te_dataloader, args)
else:
    exp_dir = Path(args.exp_dir) 
    audio_model = nn.DataParallel(audio_model)
    audio_model.load_state_dict(torch.load(exp_dir / 'models/best_audio_model_mse.pth'))
    te_loss,\
    (te_prec, te_prec_mse),\
    (te_rec, te_rec_mse),\
    (te_micro_f1, te_micro_f1_mse),\
    (te_macro_f1, te_macro_f1_mse),\
    (te_spearmanr, te_spearmanr_mse),\
    (te_kendalltau, te_kendalltau_mse),\
    (te_pairwise_acc, te_pairwise_acc_mse),\
    te_auc, te_confusion = validate(audio_model, te_dataloader, args, -1)

    result = np.zeros([1, 24])
    result[0, 6:16] = [te_loss, te_prec, te_rec, te_micro_f1, te_macro_f1, te_macro_f1, te_spearmanr, te_kendalltau, te_pairwise_acc, te_auc]
    result[0, 16:24] = [te_prec_mse, te_rec_mse, te_micro_f1_mse, te_macro_f1_mse, te_macro_f1_mse, te_spearmanr_mse, te_kendalltau_mse, te_pairwise_acc_mse]
    header = ','.join(gen_result_header())
    np.savetxt(exp_dir / 'result.csv', result, fmt='%.4f', delimiter=',', header=header, comments='')

    print(f'Precision: {te_prec:.3f}, Recall: {te_rec:.3f}, Micro F1: {te_micro_f1:.3f}, Macro F1: {te_macro_f1:.3f}, AUC: {te_auc:.3f}, Spearman Corr.: {te_spearmanr:.3f}, Kendall Corr.: {te_kendalltau:.3f}, Pairwise Acc.: {te_pairwise_acc:.3f}', flush=True)
    print(f'(MSE) Precision: {te_prec_mse:.3f}, Recall: {te_rec_mse:.3f}, Micro F1: {te_micro_f1_mse:.3f}, Macro F1: {te_macro_f1_mse:.3f}, Spearman Corr.: {te_spearmanr_mse:.3f}, Kendall Corr.: {te_kendalltau_mse:.3f}, Pairwise Acc.: {te_pairwise_acc_mse:.3f}', flush=True)
    print('-------------------validation finished-------------------', flush=True)
