import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    sensitivity, specificity, = [], []
    epoch_num = args.epochs

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            for j in range(len(data.hg)):
                data.hg[j].to(device)
                # data.hg[j].v_weight = torch.tensor(data.hg[j].v_weight)
                # data.hg[j].v_weight.to(device)
            data.x = data.x.to(device)
            if args.mixup:
                data, y_a, y_b, lam = mixup(data)
            optimizer.zero_grad()
            out = model(data)

            if args.mixup:
                loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
            else:
                loss = F.nll_loss(out, data.y)
                # Debugging logs
                # print(f"Loss: {loss.item()}")
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient norm for {name}: {param.grad.norm().item()}")

            loss.backward()
            optimizer.step()

            loss_all += loss.item()
        epoch_loss = loss_all / len(train_loader.dataset)

        train_micro, train_auc, train_macro, train_sen, train_spe = evaluate(model, device, train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}, '
                     f'train_sen={(train_sen * 100):.2f}, '
                     f'train_spe={(train_spe * 100):.2f}')
        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro, test_sen, test_spe = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            sensitivity.append(test_sen)
            specificity.append(test_spe)
            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f},' \
                   f'test_sen={(test_sen * 100):.2f}, test_spe={(test_spe * 100):.2f}\n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    sensitivity, specificity = np.sort(np.array(sensitivity)), np.sort(np.array(specificity))
    return accs.mean(), aucs.mean(), macros.mean(), sensitivity.mean(), specificity.mean()


@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        for j in range(len(data.hg)):
            data.hg[j].to(device)
        c = model(data)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()
    # print("trues:", trues)
    # print("preds_prob:", preds_prob)
    # print("preds:", preds)
    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
    # 计算混淆矩阵
    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix.ravel()

    # 计算 Specificity (SPE) 和 Sensitivity (SEN)
    train_sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    train_spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    if test_loader is not None:
        test_micro, test_auc, test_macro, test_sen, test_spe = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, train_sen, train_spe, test_micro, test_auc, test_macro, test_sen, test_spe
    else:
        return train_micro, train_auc, train_macro, train_sen, train_spe,
