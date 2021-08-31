import os
import os.path as osp
import networkx as nx
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sacred.run import Run
from logging import Logger
from sacred import Experiment
from sacred.observers import MongoObserver
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data import CATEGORY_IDS
from data import GraphDataset, TextGraphDataset, GloVeTokenizer
import models
import utils
import joblib

ex = Experiment()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_PATH = 'output/'


@ex.config
def config():
    dataset = 'entities'
    checkpoint = None
    use_cached_text = False
    lr_classifier_name = "lr-entities.model"
    result_name = "labeled_"


@ex.command
def node_classification(dataset, checkpoint, _run: Run, _log: Logger,
                        lr_classifier_name, result_name):
    '''
        This function takes BERT generated embedding as feature
        to train a classifier
    '''
    ent_emb = torch.load(f'output/ent_emb-{checkpoint}.pt', map_location='cpu')
    if isinstance(ent_emb, tuple):
        ent_emb = ent_emb[0]

    ent_emb = ent_emb.squeeze().numpy()
    num_embs, emb_dim = ent_emb.shape
    _log.info(f'Loaded {num_embs} embeddings with dim={emb_dim}')

    emb_ids = torch.load(f'output/ents-{checkpoint}.pt', map_location='cpu', encoding='iso8859-1')
    ent2idx = utils.make_ent2idx(emb_ids, max_ent_id=emb_ids.max()).numpy()
    maps = torch.load(f'data/{dataset}/maps.pt', encoding='iso8859-1')
    ent_ids = maps['ent_ids']

    class2label = defaultdict(lambda: len(class2label))
    entities = []
    splits = ['train', 'dev', 'test']
    split_2data = dict()
    for split in splits:
        with open(f'data/{dataset}/{split}-ents-class.txt', encoding='iso8859-1') as f:
            idx = []
            labels = []
            for line in f:
                entity, ent_class = line.strip().split()
                try:
                    entity_id = ent_ids[entity]
                    entity_idx = ent2idx[entity_id]
                    idx.append(entity_idx)
                    labels.append(class2label[ent_class])
                    if split == "test":
                        entities.append(entity)
                except:
                    _log.info(f'Entity extraction failed due to coding issue: {entity}')
                    pass

            x = ent_emb[idx]
            y = np.array(labels)
            split_2data[split] = (x, y)

    x_train, y_train = split_2data['train']
    x_dev, y_dev = split_2data['dev']
    x_test, y_test = split_2data['test']

    best_dev_metric = 0.0
    best_c = 0
    for k in range(-4, 2):
        c = 10 ** -k
        model = LogisticRegression(C=c, multi_class='multinomial',
                                   max_iter=1000)
        model.fit(x_train, y_train)

        dev_preds = model.predict(x_dev)
        dev_acc = accuracy_score(y_dev, dev_preds)
        _log.info(f'{c:.3f} - {dev_acc:.3f}')

        if dev_acc > best_dev_metric:
            best_dev_metric = dev_acc
            best_c = c

    _log.info(f'Best regularization coefficient: {best_c:.4f}')
    model = LogisticRegression(C=best_c, multi_class='multinomial',
                               max_iter=1000)
    x_train_all = np.concatenate((x_train, x_dev))
    y_train_all = np.concatenate((y_train, y_dev))
    model.fit(x_train_all, y_train_all)

    for metric_fn in (accuracy_score, balanced_accuracy_score):
        train_preds = model.predict(x_train_all)
        train_metric = metric_fn(y_train_all, train_preds)

        test_preds = model.predict(x_test)
        test_preds_proba = model.predict_proba(x_test)
        test_metric = metric_fn(y_test, test_preds)

        _log.info(f'Train {metric_fn.__name__}: {train_metric:.3f}')
        _log.info(f'Test {metric_fn.__name__}: {test_metric:.3f}')

    # save fitted best performed lr classifier
    joblib.dump(model, lr_classifier_name)

    # save classification result
    lable2class = ['location', 'person', 'art', 'organization', 'other', 'product', 'building', 'event']
    predictions = []
    probs = []
    n = 0
    for num in test_preds:
        predictions.append(lable2class[num])
        probs.append(test_preds_proba[n][num])
        n += 1
    res = np.vstack((np.array(entities), np.array(predictions), np.array(probs)))
    np.savetxt(result_name, res.T, delimiter=' ', fmt='%s')


@ex.command
def node_prediction(dataset, checkpoint, _run: Run, _log: Logger,result_name):
    '''
    This function use classifier to predict label for
    either question or entity description's embedding
    '''
    ent_emb = torch.load(f'output/ent_emb-{checkpoint}.pt', map_location='cpu')
    if isinstance(ent_emb, tuple):
        ent_emb = ent_emb[0]
    ent_emb = ent_emb.squeeze().numpy()
    num_embs, emb_dim = ent_emb.shape
    _log.info(f'Loaded {num_embs} embeddings with dim={emb_dim}')

    emb_ids = torch.load(f'output/ents-{checkpoint}.pt', map_location='cpu', encoding='iso8859-1')
    ent2idx = utils.make_ent2idx(emb_ids, max_ent_id=emb_ids.max()).numpy()
    maps = torch.load(f'data/{dataset}/maps.pt', encoding='iso8859-1')
    ent_ids = maps['ent_ids']
    class2label = defaultdict(lambda: len(class2label))

    with open(f'data/{dataset}/ents-class.txt', encoding='iso8859-1') as f:
        idx = []
        entities = []
        for line in f:
            entity, ent_class = line.strip().split()
            try:
                entity_id = ent_ids[entity]
                entities.append(entity)
                entity_idx = ent2idx[entity_id]
                idx.append(entity_idx)
                # labels.append(class2label[ent_class])
            except:
                _log.info(f'Entity extraction failed due to coding issue: {entity}')
                pass

        x = ent_emb[idx]

    tokens = str(dataset).split("_")
    if tokens[-1] == "questions":
        model = joblib.load('models/lr-questions.model')
        label2class = ['art', 'other', 'location', 'person', 'organization', 'event', 'product', 'building']
    else:
        model = joblib.load('models/lr-entities.model')
        label2class = ['location', 'person', 'art', 'organization', 'other', 'product', 'building', 'event']

    x_preds = model.predict(x)
    test_preds_proba = model.predict_proba(x)
    predictions = []
    probs = []
    n = 0

    for num in x_preds:
        predictions.append(label2class[num])
        probs.append(test_preds_proba[n][num])
        n += 1
    res = np.vstack((np.array(entities), np.array(predictions), np.array(probs)))
    if tokens[-1] == "questions":
        np.savetxt(result_name+"questions.txt", res.T, delimiter=' ', fmt='%s')
    else:
        np.savetxt(result_name+"entities.txt", res.T, delimiter=' ', fmt='%s')


ex.run_commandline()
