import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import register_data_args
from dgl.nn.pytorch.conv import SAGEConv
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))  # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    print('----loading dataset----')
    dataset = DglNodePropPredDataset(name='ogbn-products')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    print('>>> dataset loaded, name: {}, task: {}'.format(dataset_name, dataset_task))

    print('----processing data for training----')
    split_idx = dataset.get_idx_split()
    g = dataset.graph[0]
    features = g.ndata['feat']
    labels = dataset.labels.squeeze()
    train_mask = [0] * len(labels)
    for id in split_idx['train']:
        train_mask[id] = 1
    val_mask = [0] * len(labels)
    for id in split_idx['valid']:
        val_mask[id] = 1
    test_mask = [0] * len(labels)
    for id in split_idx['test']:
        test_mask[id] = 1
    full_mask = [1] * len(labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)
        full_mask = torch.BoolTensor(full_mask)
    else:
        train_mask = torch.ByteTensor(train_mask)
        val_mask = torch.ByteTensor(val_mask)
        test_mask = torch.ByteTensor(test_mask)
        full_mask = torch.ByteTensor(full_mask)
    in_feats = features.shape[1]
    n_classes = dataset.num_classes
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)



    # create GraphSAGE model
    print('----building train model----')
    model = GraphSAGE(g,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type
                      )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    print('----train start----')
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1 = time.time()
        dur.append(t1 - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    acc = evaluate(model, features, labels, full_mask)
    print("Full Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)
