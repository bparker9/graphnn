import numpy as np
import pickle
import torch
from functools import reduce
from torch.optim import Adam
from torch_geometric.loader import DenseDataLoader, DataLoader, ImbalancedSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from functools import reduce
from models import *

class SaveResults:
    def __init__(self, dataset, model, folds, epochs, batch_size, lr, include_test):
        self.dataset = f'Dataset: {dataset}'
        self.model = f'Model: {model}'
        self.folds = f'Number of folds: {folds}'
        self.epochs = f'Number of epochs: {epochs}'
        self.batch_size = f'Batch size: {batch_size}'
        self.lr = f'Learning rate: {lr}'
        self.include_test = f'Was the test set evaluated? {include_test}'

    def save_to_disk(self, filename: str):
        with open(filename, 'wb') as opt:
            pickle.dump(self, opt, pickle.HIGHEST_PROTOCOL)


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


# This will initiate training and evaluation of the model
# All data from the run will be stored in the object "results".
def cross_validation(dataset, model, folds, epochs, batch_size, lr, test=False, save=False):
    if save:
        global results
        results = SaveResults(str(dataset), str(model), str(folds), str(epochs), str(batch_size), str(lr), test)

    test_acc = []

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_data = dataset[train_idx]
        test_data = dataset[test_idx]
        valid_data = dataset[val_idx]

        train_sampler = ImbalancedSampler(train_data)
        valid_sampler = ImbalancedSampler(valid_data)
        test_sampler = ImbalancedSampler(test_data)

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, shuffle=False)

        model.reset_parameters()

        optimizer = Adam(model.parameters(), lr=lr)

        progress = tqdm(range(1, epochs + 1), ncols=100)

        train_loss = []  # losses and accuracies per epoch
        valid_loss = []
        train_acc = []
        valid_acc = []

        for epoch in progress:
            loss = train(model, optimizer, train_loader)
            train_loss.append(loss)
            train_acc.append(eval_acc(model, train_loader))
            valid_loss.append(eval_loss(model, valid_loader))
            valid_acc.append(eval_acc(model, valid_loader))
            info = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': train_loss[-1],
                'valid_loss': valid_loss[-1],
                'train_acc': train_acc[-1],
                'valid_acc': valid_acc[-1],
            }

            log = f'Fold: {info.get("fold")}, Epoch: {info.get("epoch")}, Training Loss: {info.get("train_loss"):.3f}, ' \
                  f'Validation Loss: {info.get("valid_loss"):.3f}, Validation Accuracy: {info.get("valid_acc"):.3f}'
            progress.set_description(log)

        if test:
            test_targets = []
            test_preds = []

            model.eval()

            for data in test_loader:
                with torch.no_grad():
                    pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                    test_targets.append(data.y.tolist())
                    test_preds.append(pred.tolist())

            # the lists are nested so we need to flatten them.
            test_targets = reduce(lambda a, b: a + b, test_targets)
            test_preds = reduce(lambda a, b: a + b, test_preds)
            conf = confusion_matrix(test_targets, test_preds).tolist()
            tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
            p = tp + fn  # number of true 1's in the data
            n = tn + fp  # number of true 0's in the data
            tpr = tp / p  # sensitivity
            tnr = tn / n  # specificity
            acc = (tp + tn) / (p + n)  # accuracy
            test_acc.append(acc)

            if save:
                setattr(results, f'fold{fold + 1}',
                        dict(
                            trainloss=train_loss,
                            validloss=valid_loss,
                            trainacc=train_acc,
                            validacc=valid_acc,
                            targets=test_targets,
                            preds=test_preds,
                            confusion=conf,
                            sensitivity=tpr,
                            specificity=tnr,
                            accuracy=acc,
                        ))

        elif save:
            setattr(results, f'fold{fold + 1}',
                    dict(
                        trainloss=train_loss,
                        validloss=valid_loss,
                        trainacc=train_acc,
                        validacc=valid_acc,
                    ))
    if test:
        test_acc = np.array(test_acc)
        accuracy_mean = test_acc.mean()
        accuracy_std = test_acc.std()

        if save:
            setattr(results, 'finalacc_mean', accuracy_mean)
            setattr(results, 'finalacc_std', accuracy_std)
            return results

        output = f'The mean accuracy over {folds} folds is {accuracy_mean}. The standard deviation is {accuracy_std}.'
        return output

    elif save:
        return results
    else:
        return 'Success!!!'





