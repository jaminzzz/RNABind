import os
import datetime
import argparse
import pathlib
from loguru import logger
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


from RNABind.utils import seed_set
from RNABind.utils import build_scheduler, build_optimizer
from RNABind.utils import save_best_checkpoint, load_best_result, save_results, prc_auc
from RNABind.data import build_rna_bs_dataloader
from RNABind.models import build_model


def parse_args():
    r"""Parse arguments.
    """
    parser = argparse.ArgumentParser(description="Codes for RNABind")

    # environment arguments
    parser.add_argument('--seed', default=1, type=int, help="set seed")
    parser.add_argument('--dataset_number', default=1, type=int, help="set number")
    parser.add_argument('--task', default='rl_binding_site', type=str, help="task name, rl_binding_site or pl_binding_site (pre-training)")
    parser.add_argument('--transfer_learning', default=False, type=bool, help="whether to use transfer learning")

    # directory arguments
    parser.add_argument('--data_path', default='~/RNABind/bs_data/RNA', type=str, help="directory of pre-training data")
    parser.add_argument('--output_dir', default='~/RNABind/results', type=str, help="directory of pre-training results")
    parser.add_argument('--tag', default='ernierna_single', type=str, help="tag of the experiment")

    # network arguments
    parser.add_argument('--embedding_type', default='ernierna', type=str, help="embedding type")
    parser.add_argument('--in_node_nf', default=768, type=int, help="input node feature dimension")
    parser.add_argument('--hidden_nf', default=128, type=int, help="hidden node feature dimension")
    parser.add_argument('--out_node_nf', default=128, type=int, help="output node feature dimension")
    parser.add_argument('--in_edge_nf', default=16, type=int, help="input edge feature dimension")
    parser.add_argument('--n_layers', default=3, type=int, help="num of rna encoder layers")
                        
    # Training settings
    parser.add_argument('--tensorboard_enable', default=False, type=bool, help="enable tensorboard or not")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size, default 32 for rl_scoring, 128 for pl_scoring") 
    parser.add_argument('--lr', default=3e-4, type=float, help="initial (base) learning rate")
    parser.add_argument('--l2', default=1e-4, type=float, help="l2 regularization weight")
    parser.add_argument('--early_stop', default=30, type=int, help="early stop patience")
    parser.add_argument('--max_epochs', default=300, type=int, help="number of training epoch")
    parser.add_argument('--factor', default=0.75, type=float, help="factor for lr_scheduler")
    parser.add_argument('--patience', default=10, type=int, help="patience for lr_scheduler")
    parser.add_argument('--min_lr', default=1e-4, type=float, help="min lr for lr_scheduler")
    parser.add_argument('--show_freq', default=1, type=int, help="show frequency")
    parser.add_argument('--metric', type=str, help="Metric for best model", default='auroc')

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.tag)

    return args


def run_a_train_epoch(model, criterion, dataloader, optimizer, device):
    r"""Conducting one training epoch.
    """
    model.train()

    loss_all = 0
    batch_results_all = {}
    
    for data in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        model = model.to(device)
    
        pred = model(data)

        label = data.binding_site.unsqueeze(1).float()
        label = label.to(device)

        loss = criterion(pred, label)
        loss_all += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 转换为numpy
        pred = pred.detach().cpu().numpy()
        label = label.cpu().numpy()
        # per batch results 
        batch = data.batch.cpu().numpy()
        for i in range(batch.max() + 1): 
            # 根据index获取每个batch的元素
            batch_index = np.where(batch == i)[0]
            # 获取pdb_id
            pdb_id = data.pdb_id[i]
            # 获取每个batch的预测值和标签值
            preds = pred[batch_index]
            labels = label[batch_index]
            # 计算每个batch的元素的个数
            len_residues = len(preds)
            # 将每个batch的长度和auroc值存入batch_results
            batch_results_all[pdb_id] = (len_residues, roc_auc_score(labels, preds), prc_auc(labels, preds))

    # Record loss
    epoch_loss = loss_all / len(dataloader)
    # Compute metric
    auroc_sum, auprc_sum, count = 0, 0, 0
    for key in batch_results_all:
        auroc_sum += batch_results_all[key][1]
        auprc_sum += batch_results_all[key][2]
        count += 1

    auroc = auroc_sum / count
    auprc = auprc_sum / count
    results_dict =  {'auroc': auroc, 'auprc': auprc}

    return epoch_loss, results_dict
        

@torch.no_grad()
def run_an_eval_epoch(model, criterion, dataloader, device):
    r"""Conducting one validation epoch.
    """
    model.eval()

    loss_all = 0
    batch_results_all = {}

    for data in dataloader:
        data = data.to(device)
        model = model.to(device)

        pred = model(data)
        
        label = data.binding_site.unsqueeze(1).float()
        label = label.to(device)

        loss = criterion(pred, label)
        loss_all += loss.item()

        pred = pred.detach().cpu().numpy()
        label = label.cpu().numpy()

        # per batch results 
        batch = data.batch.cpu().numpy()
        for i in range(batch.max() + 1):
            # 根据index获取每个batch的元素
            batch_index = np.where(batch == i)[0]
            # 获取pdb_id
            pdb_id = data.pdb_id[i]
            # 获取每个batch的预测值和标签值
            preds = pred[batch_index]
            labels = label[batch_index]
            # 计算每个batch的元素的个数
            len_residues = len(preds)
            # 将每个batch的长度和auroc值存入batch_results
            batch_results_all[pdb_id] = (len_residues, roc_auc_score(labels, preds), prc_auc(labels, preds))

    # Record loss
    epoch_loss = loss_all / len(dataloader)
    # Compute metric
    auroc_sum, auprc_sum, count = 0, 0, 0
    for key in batch_results_all:
        auroc_sum += batch_results_all[key][1]
        auprc_sum += batch_results_all[key][2]
        count += 1

    auroc = auroc_sum / count
    auprc = auprc_sum / count
    results_dict =  {'auroc': auroc, 'auprc': auprc}

    return epoch_loss, results_dict, batch_results_all


def train(args):
    r"""The training process.
    """
    # step 0: set up args by tag
    if args.tag == 'lucaone':
        args.embedding_type = 'lucaone'
        args.in_node_nf = 2560
    elif args.tag == 'rinalmo':
        args.embedding_type = 'rinalmo'
        args.in_node_nf = 1280
    elif args.tag == 'protrna':
        args.embedding_type = 'protrna'
        args.in_node_nf = 1280
    elif args.tag == 'rnaernie':
        args.embedding_type = 'rnaernie'
        args.in_node_nf = 768
    elif args.tag == 'ernierna':
        args.embedding_type = 'ernierna'
        args.in_node_nf = 768
    elif args.tag == 'rnamsm': 
        args.embedding_type = 'rnamsm'
        args.in_node_nf = 768
    elif args.tag == 'rnafm':
        args.embedding_type = 'rnafm'
        args.in_node_nf = 640
    elif args.tag == 'rnabert':
        args.embedding_type = 'rnabert'
        args.in_node_nf = 120
    elif args.tag == 'onehot':
        args.embedding_type = 'onehot'
        args.in_node_nf = 128 # one-hot encoding by Embedding layer
    else:
        pass    
    
    # step 1: dataloder loading
    train_loader, val_loader, test_loader = build_rna_bs_dataloader(args)
    # step 2: model loading
    model = build_model(args)
    # device mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # step 3: optimizer loading
    optimizer = build_optimizer(args, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# of params: {n_parameters}")

    # step 4: lr_scheduler loading
    lr_scheduler = build_scheduler(args, optimizer)

    # step 5: loss function loading
    criterion = nn.BCELoss(reduction='mean')

    # step 6: tensorboard loading
    if args.tensorboard_enable:
        tensorboard_dir = pathlib.Path(args.output_dir) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # step 7: training loop
    best_epoch, best_score = 0, 0
    early_stop_count = 0
    logger.info(f'==================== Training starts at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ====================')
    for epoch in range(args.max_epochs):
        # (1) results after one epoch training.
        train_loss, train_results = run_a_train_epoch(model, criterion, train_loader, optimizer, device)
        val_loss, val_results, _ = run_an_eval_epoch(model, criterion, val_loader, device)
        # just for observing the test set results during training
        test_loss, test_results, _ = run_an_eval_epoch(model, criterion, test_loader, device)

        train_score, val_score, test_score = train_results[args.metric], val_results[args.metric], test_results[args.metric]

        # (2) upadate learning rate.
        lr_scheduler.step(val_score)

        # (3) log results.
        if epoch % args.show_freq == 0 or epoch == args.max_epochs - 1:
            lr_now = lr_scheduler.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch_{epoch:<2} ==> "
                        f"train_loss:{train_loss:.3f} val_loss:{val_loss:.3f} test_loss:{test_loss:.3f} | "
                        f"train_{args.metric}:{train_score:.3f} val_{args.metric}:{val_score:.3f} test_{args.metric}:{test_score:.3f} | "
                        f"lr:{lr_now:.2e}")

        # (4) tensorboard for training visualization.
        loss_dict, metric_dict = {"train_loss": train_loss}, {f"train_{args.metric}": train_score}
        loss_dict["val_loss"], metric_dict[f"val_{args.metric}"] = val_loss, val_score

        if args.tensorboard_enable:
            # we only select the args.metric to visualize.
            writer.add_scalars(f"scalar/{args.metric}", metric_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)

        # (5) save best results. 
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(args, epoch, model, best_score, best_epoch, optimizer, lr_scheduler)
            early_stop_count = 0
        else:
            early_stop_count += 1
        # (6) early stopping judgement.
        if early_stop_count > args.early_stop > 0:
            logger.info('Early stop hitted!')
            break

    if args.tensorboard_enable:
        writer.close()

    # step 8: record training time.
    logger.info(f'==================== Training ends at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ====================')

    # step 9: evaluation.
    model, best_epoch = load_best_result(args, model)
    train_loss, train_results_dict, _ = run_an_eval_epoch(model, criterion, train_loader, device)
    val_loss, val_results_dict, _ = run_an_eval_epoch(model, criterion, val_loader, device)
    test_loss, test_results_dict, batch_results = run_an_eval_epoch(model, criterion, test_loader, device)
    score = test_results_dict[args.metric]
    logger.info(f'Seed {args.seed}  ==> train_loss:{train_loss:.3f} val_loss:{val_loss:.3f} test_loss:{test_loss:.3f} | '
          f'train_{args.metric}:{train_results_dict[args.metric]:.3f} val_{args.metric}:{val_results_dict[args.metric]:.3f} '
          f'test_{args.metric}:{test_results_dict[args.metric]:.3f} | best epoch:{best_epoch}'
          )

    # step 10: save results.
    save_results(args, train_results_dict, val_results_dict, test_results_dict)

    # step 11: save batch results.
    batch_results_dir = pathlib.Path(args.output_dir) / args.task / f'number_{args.dataset_number}' / f'seed_{args.seed}' / "batch_results"
    batch_results_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_results_dir / 'batch_results.csv', 'w') as f:
        for key in batch_results:
            f.write(f'{key}, {batch_results[key][0]}, {batch_results[key][1]:.4f}, {batch_results[key][2]:.4f}\n')

    return score


if __name__ == "__main__":
    # set logger
    logger_tag = 'rl_binding_sites'
    log_file_name = '~/RNABind/logs/{1}/{0}_{1}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), logger_tag)
    logger.add(log_file_name)
    # parse args
    args = parse_args()
    # set random seed

    for number in range(1, 5):
        args.dataset_number = number
        for seed in range(1, 6):
            args.seed = seed    
            seed_set(args.seed)
            # print tag, number and seed
            logger.info(f'tag: {args.tag}, number: {args.dataset_number}, seed: {args.seed}')
            # print device mode
            logger.info('device: GPU' if torch.cuda.is_available() else 'device: CPU')
            # training
            train(args)
