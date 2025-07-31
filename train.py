# %%
import os
import csv
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_preprocessing import CustomData
from dataset import load_ddi_dataset
from train_logger import TrainLogger
from model import MPNN_DDI  
import argparse
from metrics import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

def val(model, criterion, dataloader, device, model_name):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

        with torch.no_grad():
            
            pred = model((head_pairs, tail_pairs, rel))

            loss = criterion(pred, label)

            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    np.save(f"y_true_{model_name}.npy", labels)
    np.save(f"y_scores_{model_name}.npy", pred_probs)
    np.save(f"y_pred_{model_name}.npy", (pred_probs > 0.5).astype(int))

    acc, auroc, f1_score, precision, recall, ap, aupr = do_compute_metrics(pred_probs, labels)

    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap, aupr

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kge_dim', type=int, default=64)
    parser.add_argument('--rel_total', type=int, default=86)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='dmpnn', choices=['dmpnn', 'gat', 'gin', 'transformer', 'graphsage'])
    args = parser.parse_args()

    params = dict(
        model=args.model,
        data_root='data/preprocessed/',
        save_dir='save',
        dataset='drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay,
        kge_dim=args.kge_dim,
        rel_total=args.rel_total,
        hidden_dim=args.hidden_dim
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    save_model_flag = args.save_model
    data_path = os.path.join(params['data_root'], params['dataset'])
    exp_path = f"result/{args.model}/"
    os.makedirs(exp_path, exist_ok=True)

    train_loader, val_loader, test_loader = load_ddi_dataset(
        root=data_path, batch_size=args.batch_size, fold=args.fold)

    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'dmpnn':
        model = MPNN_DDI(node_dim, edge_dim, hidden_dim=args.hidden_dim, n_iter=args.n_iter,
                         kge_dim=args.kge_dim, rel_total=args.rel_total)
    elif args.model == 'gat':
        from model import GAT_DDI
        model = GAT_DDI(node_dim, hidden_dim=args.hidden_dim, n_iter=args.num_layers,
                    kge_dim=args.kge_dim, rel_total=args.rel_total,
                    heads=args.num_heads, dropout=args.dropout)
    elif args.model == 'gin':
        from model import GIN_DDI
        model = GIN_DDI(node_dim, edge_dim, hidden_dim=args.hidden_dim, n_iter=args.num_layers,
                        kge_dim=args.kge_dim, rel_total=args.rel_total, dropout=args.dropout)
    elif args.model == 'transformer':
        from model import LocalTransformer_DDI
        model = LocalTransformer_DDI(node_dim=node_dim, hidden_dim=args.hidden_dim, n_iter=args.num_layers,
                                    num_heads=args.num_heads, kge_dim=args.kge_dim, rel_total=args.rel_total, dropout=args.dropout)
    elif args.model == 'graphsage':
        from model import GraphSAGE_DDI
        model = GraphSAGE_DDI(node_dim, edge_dim, hidden_dim=args.hidden_dim, n_iter=args.num_layers,  
                            kge_dim=args.kge_dim, rel_total=args.rel_total, dropout=args.dropout
    )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    running_loss = AverageMeter()
    running_acc = AverageMeter()

    output_csv_dir = "plot_data"
    os.makedirs(output_csv_dir, exist_ok=True)
    csv_file = os.path.join(output_csv_dir, f"training_log_{args.model}.csv")

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "test_loss", "test_acc",
            "val_auroc", "val_prec", "val_rec", "val_aupr", "val_f1_score",
            "test_auroc", "test_prec", "test_rec", "test_aupr", "test_f1_score"
        ])

    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            head_pairs, tail_pairs, rel, label = batch
            head_pairs = head_pairs.to(device)
            tail_pairs = tail_pairs.to(device)
            rel = rel.to(device)
            label = label.to(device)

            pred = model((head_pairs, tail_pairs, rel))
                
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_prec, val_rec, val_ap, val_aupr = val(
            model, criterion, val_loader, device, args.model)

        test_loss, test_acc, test_auroc, test_f1_score, test_prec, test_rec, test_ap, test_aupr = val(
            model, criterion, test_loader, device, args.model)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, epoch_loss, epoch_acc,
                val_loss, val_acc, test_loss, test_acc,
                val_auroc, val_prec, val_rec, val_aupr, val_f1_score,
                test_auroc, test_prec, test_rec, test_aupr, test_f1_score
            ])

        logger.info(
            f"Epoch {epoch} | Train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} auroc {val_auroc:.4f} f1 {val_f1_score:.4f} "
            f"| Test loss {test_loss:.4f} acc {test_acc:.4f} auroc {test_auroc:.4f} f1 {test_f1_score:.4f}"
        )

        scheduler.step()

        if save_model_flag:
            save_model_dict(model, logger.get_model_dir(), f"epoch-{epoch}_val_acc-{val_acc:.4f}")

# %%
if __name__ == "__main__":
    main()
