import pickle
import itertools

import torch.optim.lr_scheduler
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader, ImbalancedSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from functools import reduce
from models import *


def save_to_disk(instance, filename: str):
    with open(filename, 'wb') as opt:
        pickle.dump(instance, opt, pickle.HIGHEST_PROTOCOL)


def load_from_disk(filename: str):
    with open(filename, 'rb') as inp:
        results = pickle.load(inp)
    return results


def k_fold(dataset, folds):
    kf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()
    loss_func = torch.nn.CrossEntropyLoss()
    total_loss = 0  # sum of loss calculated at each batch in loader

    for data in loader:
        optimizer.zero_grad()
        out = model(data)  # model output gives logits for each sample in the batch
        batch_loss = loss_func(out, data.y)  # compute average loss for the batch
        total_loss += batch_loss.item() * num_graphs(data)  # undo average to get sum. add to epoch_loss.
        batch_loss.backward()
        optimizer.step()
    epoch_loss = total_loss / len(loader.dataset)  # total epoch loss / num graphs in train_loader
    return epoch_loss


def eval_loss(model, loader):  # returns the loss for a single epoch

    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    loss = 0
    for data in loader:
        with torch.no_grad():
            out = model(data)
        batch_loss = loss_func(out, data.y)
        loss += batch_loss.item() * num_graphs(data)
    return loss / len(loader.dataset)


def eval_acc(model, loader):  # returns the accuracy for a single epoch
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
        correct += int((pred == data.y).sum())
    accuracy = correct / len(loader.dataset)
    return accuracy


def eval_all(model, loader, targets=None, preds=None):
    if targets is None and preds is None:
        targets = []
        preds = []

        model.eval()

        for data in loader:
            with torch.no_grad():
                batch_preds = model(data).argmax(dim=1)
                targets.append(data.y.tolist())
                preds.append(batch_preds.tolist())
        targets = reduce(lambda a, b: a + b, targets)
        preds = reduce(lambda a, b: a + b, preds)
        conf = confusion_matrix(targets, preds).tolist()
        tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
        p = tp + fn  # number of true 1's in the data
        n = tn + fp  # number of true 0's in the data
        tpr = tp / p  # sensitivity
        tnr = tn / n  # specificity
        acc = (tp + tn) / (p + n)  # accuracy

    return acc, conf, tpr, tnr


def split_data(dataset):
    length = len(dataset)
    generator = torch.Generator().manual_seed(12345)
    train_data, valid_data, test_data = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    return train_data, valid_data, test_data


def grid_search(dataset, train_data, valid_data, hps):

    hps_list = [_ for _ in hps.values()]
    combs = [_ for _ in itertools.product(*hps_list)]
    best_hps = {k: v for k, v in zip(hps, combs[0])}
    best_acc = 0
    best_results = None

    train_sampler = ImbalancedSampler(train_data)
    valid_sampler = ImbalancedSampler(valid_data)

    for i in range(len(combs)):
        print(f'Evaluating {i+1}th set of parameters out of {len(combs)}')

        model, hidden, batch_size, lr, epochs, conv_layers = combs[i]

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, shuffle=False)

        ith_model = model(dataset=dataset, num_layers=conv_layers, hidden_channels=hidden)
        ith_model.reset_parameters()
        optimizer = Adam(ith_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-10)

        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []

        progress= tqdm(range(1, epochs + 1), leave=False)
        for epoch in progress:

            loss = train(ith_model, optimizer, train_loader)
            train_loss.append(loss)
            train_acc.append(eval_acc(ith_model, train_loader))
            valid_loss.append(eval_loss(ith_model, valid_loader))
            valid_acc.append(eval_acc(ith_model, valid_loader))
            scheduler.step(valid_loss[-1])

            info = {
                'epoch': epoch,
                'train_loss': train_loss[-1],
                'valid_loss': valid_loss[-1],
                'train_acc': train_acc[-1],
                'valid_acc': valid_acc[-1],
                }

            log = f'Epoch: {info["epoch"]}, Training Loss: {info["train_loss"]:.3f},' \
                   f'Validation Loss: {info["valid_loss"]:.3f}, Validation Accuracy: {info["valid_acc"]:.3f}'
            progress.set_description(log)
        print(max(valid_acc))
        if max(valid_acc) > best_acc:
            best_acc = max(valid_acc)
            current_hps = combs[i]
            best_hps = {k: v for k, v in zip(hps, current_hps)}
            valid_accuracy, conf_matrix, sensitivity, specificity = eval_all(ith_model, valid_loader)
            best_results ={
                'hyperparameters': best_hps,
                'validation accuracy': valid_accuracy,
                'confusion matrix': conf_matrix,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'train losses': train_loss,
                'validation losses': valid_loss,
                'train accuracies': train_acc,
                'validation accuracies': valid_acc
            }
    return best_results







