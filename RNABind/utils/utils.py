import os
import random
import pathlib
import numpy as np
import pandas as pd
from loguru import logger

import torch
from sklearn.metrics import auc, precision_recall_curve


# -----------------------------------------------------------------------------
# Set seed for random, numpy, torch, cuda.
# -----------------------------------------------------------------------------
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Checkpoint loading and saving.
# -----------------------------------------------------------------------------
def save_best_checkpoint(args, epoch, model, best_score, best_epoch, optimizer, lr_scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'best_score': best_score,
                  'best_epoch': best_epoch,
                  'epoch': epoch,
                  'config': args}

    ckpt_dir = pathlib.Path(args.output_dir) / args.task / f'number_{args.dataset_number}' / f'seed_{args.seed}' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f'{args.task}_best_model.pt')
    torch.save(save_state, save_path)
    logger.info(f"---------- checkpoint saved at epoch {epoch} ----------")


def load_best_result(args, model):
    ckpt_dir = pathlib.Path(args.output_dir) / args.task / f'number_{args.dataset_number}' / f'seed_{args.seed}' / 'checkpoints'
    best_ckpt_path = os.path.join(ckpt_dir, f'{args.task}_best_model.pt')
    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt['best_epoch']

    return model, best_epoch


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
def build_optimizer(args, model):
    params = model.parameters()
    optimizer = torch.optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.l2,
    )
    return optimizer


# -----------------------------------------------------------------------------
# Lr_scheduler
# -----------------------------------------------------------------------------
def build_scheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.factor,
        patience=args.patience,
        min_lr=args.min_lr
    )
    return scheduler


# -----------------------------------------------------------------------------
# Metric utils
# -----------------------------------------------------------------------------
def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def save_results(args, train_results_dict, val_results_dict, test_results_dict, overwrite=False):
    csv_name = f'results_{args.tag}_seed_{args.seed}.csv'
    save_path = pathlib.Path(args.output_dir) / args.task / f'number_{args.dataset_number}' / f'seed_{args.seed}' / csv_name

    # Check if the output directory exists, if not, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train_results = pd.DataFrame([{f'train_{k}': v.item() for k, v in train_results_dict.items()}])
    val_results = pd.DataFrame([{f'val_{k}': v.item() for k, v in val_results_dict.items()}])
    test_results = pd.DataFrame([{f'test_{k}': v.item() for k, v in test_results_dict.items()}])
    results = pd.concat([train_results, val_results, test_results], axis=1).round(4)

    try:
        results.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {save_path}. Error: {e}")


# -----------------------------------------------------------------------------
# other utils
# -----------------------------------------------------------------------------
def remove_keys_from_dict(d, keys):
    r"""
    Remove keys from a dictionary.
    """
    for key in keys:
        if key in d:
            if key == 'coords' and d[key] is None:
                d.pop(key)
            elif key != 'coords':
                d.pop(key)


def keep_keys_in_dict(d, keys):
    r"""
    Keep keys in a dictionary.
    """
    if 'coords' in d and d.get('coords') is None:
        d.pop('coords')
    for key in list(d.keys()):
        if key not in keys:
            d.pop(key)