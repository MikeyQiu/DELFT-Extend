import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import time
import pickle
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data import load_data
from data import TriviaQADataset
from data import batcher

from model import Model, Encoder_rnn, LinearAttn, BilinearSeqAttn, RGCNLayer
from pytorch_transformers.tokenization_bert import BertTokenizer

# from transformers import AutoTokenizer, AutoModel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
#
# model = AutoModel.from_pretrained("studio-ousia/luke-base")

logger = logging.getLogger()
'''
1 Model Evaluation 
2 Model Parameterization 
3 Training/Testing 
'''
'''
Model Evaluation
'''


def evaluate(data_loader, model):
    total_count = 0
    model.eval()
    total_list = list()
    for j, dev_batch in enumerate(tqdm(data_loader)):
        logits = model(dev_batch)
        pos_idx = [i for i in range(dev_batch.label.size(0)) if dev_batch.label[i].item() != -1]
        if len(pos_idx) == 1:
            total_count += 1
            total_list.append(1)
            continue

        logits = logits[pos_idx].data.cpu().numpy().tolist()
        label = dev_batch.label[pos_idx]
        sorted_idx = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
        if sorted_idx[0] == 0 and logits[sorted_idx[0]] != logits[sorted_idx[1]]:
            total_count += 1
            total_list.append(1)
        else:
            total_list.append(0)

    logger.info("SCORE IS: %d" % total_count)
    print("SCORE IS: %d" % total_count)
    # file = open('total_list_luke_trivia_pruned.txt','w')
    # file.write(str(total_list))
    # file.close()
    return total_count, total_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)  # default false
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--train-file", type=str,
                        default='data/trivia_train_graph.json')
    parser.add_argument("--dev-file", type=str,
                        default='data/trivia_dev_graph.json')
    parser.add_argument('--test-file', type=str,
                        default='data/trivia_test_graph.json')
    parser.add_argument('--save-model', type=str,
                        default='experiments/trivia_delft_luke.pt')
    parser.add_argument('--load-model', type=str,
                        default='experiments/trivia_delft_luke.pt')
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5) #If training time is too long, please reduce the epoch
    parser.add_argument('--input-size', type=int, default=768)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--log-file', type=str,
                        default='experiments/trivia_luke.log')
    parser.add_argument('--test', action='store_true', default=False)  # false
    parser.add_argument("--self-attn", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available()
                                    and not args.no_cuda else "cpu")
    print(device)
    args.device = device

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(args.log_file, 'a')

    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    if args.test:
        test_graph = load_data(args.test_file)
        # model = torch.load(args.load_model)
        model = Model(args)
        load_model = torch.load(args.load_model)
        model.load_state_dict(load_model.state_dict())
        model.device = device
        # tokenizer = BertTokenizer.from_pretrained(args.bert_model) #1 pretrained model
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
        dev_dataset = TriviaQADataset(test_graph, args, tokenizer, False)  # 2 extract data
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=1,
                                collate_fn=batcher(device),
                                shuffle=False,
                                num_workers=0)
        model.to(device)
        score, total_list = evaluate(dev_loader, model)
        exit()
    ####1 load data to graph ####
    train_graph = load_data(args.train_file)
    dev_graph = load_data(args.dev_file)

    model = Model(args)  # Instance of DELFT
    model.device = device
    # If training on the basis of current model
    # load_model = torch.load(args.load_model,map_location='cpu')
    # model.load_state_dict(load_model.state_dict())
    optimizer = torch.optim.Adamax(model.parameters())


    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    train_dataset = TriviaQADataset(train_graph, args, tokenizer, True)  # 2 extract data
    dev_dataset = TriviaQADataset(dev_graph, args, tokenizer, False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)

    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=1,
                            collate_fn=batcher(device),
                            shuffle=False,
                            num_workers=0)

    bce_loss_logits = nn.BCEWithLogitsLoss()
    mini_check_point = 300
    check_point = 4000
    print_loss_total = 0
    best_accuracy = 0
    start = time.time()
    model = model.to(device)

    '''
    Training
    '''

    for epoch in range(args.epochs):  # epoches
        model.train()
        for idx, batch in enumerate(tqdm(train_loader)):  # batches
            logits = model(batch)  # model delft's output
            pos_idx = [i for i in range(batch.label.size(0)) if batch.label[i].item() != -1]
            loss = bce_loss_logits(logits[pos_idx],
                                   batch.label[pos_idx].type(torch.FloatTensor).to(device))  # binary cross entropy

            # backpropagation
            optimizer.zero_grad()
            print_loss_total += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

            # log output
            if idx % mini_check_point == 0 and idx > 0:
                logger.info('number of steps: %d, loss: %.5f time: %.5f' % (
                idx, print_loss_total / mini_check_point, time.time() - start))
                print_loss_total = 0
            if idx % check_point == 0 and idx > 0:
                score, total_list = evaluate(dev_loader, model)
                model.train()
                if score >= best_accuracy:
                    best_accuracy = score
                    torch.save(model, args.save_model)
