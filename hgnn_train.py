import numpy as np
# import nni
import torch
import time
import copy
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.metrics.classification import _format_inputs


def hgnn_train_and_evaluate(model, fts, G, lbls, device, idx_train, idx_test, criterion, optimizer, scheduler,
                            num_epochs, print_freq=500):
    since = time.time()
    accs, aucs, macros = [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # if epoch % print_freq == 0:
        #     print('-' * 10)
        #     print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            idx = idx_train if phase == 'train' else idx_test
            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            # if epoch % print_freq == 0:
            #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        # v2
        # for phase in ['train', 'val']:
        #     if phase == 'train':
        #         model.train()  # Set model to training mode
        #     else:
        #         model.eval()  # Set model to evaluate mode
        #
        #     running_loss = 0.0
        #     running_corrects = 0
        #     idx = idx_train
        #     # Iterate over data.
        #     optimizer.zero_grad()
        #     with torch.set_grad_enabled(phase == 'train'):
        #         outputs = model(fts, G)
        #         loss = criterion(outputs[idx], lbls[idx])
        #         _, preds = torch.max(outputs, 1)
        #
        #         # backward + optimize only if in training phase
        #         if phase == 'train':
        #             loss.backward()
        #             optimizer.step()
        #             scheduler.step()
        #
        #     # statistics
        #     running_loss += loss.item() * fts.size(0)
        #     running_corrects += torch.sum(preds[idx] == lbls.data[idx])
        #
        #     epoch_loss = running_loss / len(idx)
        #     epoch_acc = running_corrects.double() / len(idx)
        #
        #     # if epoch % print_freq == 0:
        #     #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #
        #     # deep copy the model
        #     if epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_wts = copy.deepcopy(model.state_dict())

        # if (epoch + 1) % 100 == 0:
        #     train_micro, train_auc, train_macro = hgnn_evaluate(model, fts, G, lbls, device, idx_train)
        #     logging.info(f'(--Train) | Epoch={epoch:03d}, loss={epoch_loss:.4f}, '
        #                  f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
        #                  f'train_auc={(train_auc * 100):.2f}')
        #
        # if (epoch + 1) % print_freq == 0:
        #     test_micro, test_auc, test_macro = hgnn_evaluate(model, fts, G, lbls, device, idx_test)
        #     accs.append(test_micro)
        #     aucs.append(test_auc)
        #     macros.append(test_macro)
        #     text = f'(Train Epoch {epoch}), test_micro={(test_micro * 100):.2f}, ' \
        #            f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
        #     logging.info(text)

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    # print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


@torch.no_grad()
def hgnn_evaluate(model, fts, G, lbls, device, test_index):
    evaluator = Evaluator(["accuracy", {"f1_score": {"average": "micro"}}, {"f1_score": {"average": "macro"}}])
    model.eval()
    outputs = model(fts, G)
    outs, lbls = outputs[test_index], lbls[test_index]
    res = evaluator.test(lbls, outs)
    train_micro, train_auc, train_macro = res['accuracy'], res['f1_score -> average@micro'], res[
        'f1_score -> average@macro']

    return train_micro, train_auc, train_macro


@torch.no_grad()
def hgnn_final_evaluate(model, fts, G, lbls, device, test_index, eva, tp_count, tn_count, fp_count, fn_count):
    evaluator = Evaluator(["accuracy", {"f1_score": {"average": "micro"}}, {"f1_score": {"average": "macro"}}])
    model.eval()
    outputs = model(fts, G)
    outs, lbls = outputs[test_index], lbls[test_index]
    if eva == 'loo':
        y_true, y_pred = _format_inputs(lbls, outs)
        y_true, y_pred = y_true.item(), y_pred.item()
        if y_true == y_pred and y_true == 0:
            tp_count += 1
        if y_true == y_pred and y_true == 1:
            tn_count += 1
        if y_true != y_pred and y_true == 1:
            fp_count += 1
        if y_true != y_pred and y_true == 0:
            fn_count += 1
        return tp_count, tn_count, fp_count, fn_count
    else:
        res = evaluator.test(lbls, outs)
        train_micro, train_auc, train_macro = res['accuracy'], res['f1_score -> average@micro'], res[
            'f1_score -> average@macro']

    return train_micro, train_auc, train_macro


def hgnng_train_and_evaluate(model, fts, G, age_graph, lbls, device, idx_train, idx_test, criterion, optimizer, scheduler,
                            num_epochs, print_freq=500):
    # since = time.time()
    accs, aucs, macros = [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        # V1
        # for phase in ['train', 'val']:
        #     if phase == 'train':
        #         model.train()  # Set model to training mode
        #     else:
        #         model.eval()  # Set model to evaluate mode
        #
        #     running_loss = 0.0
        #     running_corrects = 0
        #     idx = idx_train if phase == 'train' else idx_test
        #     # Iterate over data.
        #     optimizer.zero_grad()
        #     with torch.set_grad_enabled(phase == 'train'):
        #         outputs = model(fts, G, age_graph)
        #         loss = criterion(outputs[idx], lbls[idx])
        #         _, preds = torch.max(outputs, 1)
        #
        #         # backward + optimize only if in training phase
        #         if phase == 'train':
        #             loss.backward()
        #             optimizer.step()
        #             scheduler.step()
        #
        #     # statistics
        #     running_loss += loss.item() * fts.size(0)
        #     running_corrects += torch.sum(preds[idx] == lbls.data[idx])
        #
        #     epoch_loss = running_loss / len(idx)
        #     epoch_acc = running_corrects.double() / len(idx)
        #
        #     # if epoch % print_freq == 0:
        #     #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #
        #     # deep copy the model
        #     if phase == 'val' and epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_wts = copy.deepcopy(model.state_dict())

        # v2
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            idx = idx_train
            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G, age_graph)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            # if epoch % print_freq == 0:
            #     print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # 记录到logger文件里
        if (epoch + 1) % 20 == 0:
            train_micro, train_auc, train_macro = hgnng_evaluate(model, fts, G, age_graph, lbls, device, idx_train)
            logging.info(f'(--Train) | Epoch={epoch:03d}, loss={epoch_loss:.4f}, '
                         f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                         f'train_auc={(train_auc * 100):.2f}')
        # if (epoch + 1) % print_freq == 0:
        #     test_micro, test_auc, test_macro = hgnng_evaluate(model, fts, G, age_graph, lbls, device, idx_test)
        #     accs.append(test_micro)
        #     aucs.append(test_auc)
        #     macros.append(test_macro)
        #     text = f'(Train Epoch {epoch}), test_micro={(test_micro * 100):.2f}, ' \
        #            f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
        #     logging.info(text)

        if (epoch+1) % print_freq == 0:
            print("{0}:".format(epoch+1), '-' * 20)
            print(f'Best val Acc: {best_acc:4f}')
    # 打印每一轮时间
    # time_elapsed = time.time() - since
    # print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


@torch.no_grad()
def hgnng_evaluate(model, fts, G, age_graph, lbls, device, test_index):
    evaluator = Evaluator(["accuracy", {"f1_score": {"average": "micro"}}, {"f1_score": {"average": "macro"}}])
    model.eval()
    outputs = model(fts, G, age_graph)
    outs, lbls = outputs[test_index], lbls[test_index]
    res = evaluator.test(lbls, outs)
    train_micro, train_auc, train_macro = res['accuracy'], res['f1_score -> average@micro'], res[
        'f1_score -> average@macro']

    return train_micro, train_auc, train_macro


@torch.no_grad()
def hgnng_final_evaluate(model, fts, G, age_graph, lbls, device, test_index, eva, tp_count, tn_count, fp_count, fn_count):
    evaluator = Evaluator(["accuracy", {"f1_score": {"average": "micro"}}, {"f1_score": {"average": "macro"}}])
    model.eval()
    outputs = model(fts, G, age_graph)
    outs, lbls = outputs[test_index], lbls[test_index]
    if eva == 'loo':
        y_true, y_pred = _format_inputs(lbls, outs)
        y_true, y_pred = y_true.item(), y_pred.item()
        if y_true == y_pred and y_true == 0:
            tp_count += 1
        if y_true == y_pred and y_true == 1:
            tn_count += 1
        if y_true != y_pred and y_true == 1:
            fp_count += 1
        if y_true != y_pred and y_true == 0:
            fn_count += 1
        return tp_count, tn_count, fp_count, fn_count
    else:
        res = evaluator.test(lbls, outs)
        train_micro, train_auc, train_macro = res['accuracy'], res['f1_score -> average@micro'], res[
            'f1_score -> average@macro']

    return train_micro, train_auc, train_macro