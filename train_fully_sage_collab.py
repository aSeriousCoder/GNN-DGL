import argparse
import time
from logging import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from ogb.linkproppred import Evaluator
from ogb.linkproppred.dataset_dgl import DglLinkPropPredDataset
import dgl.function as fn
from dgl.utils import check_eq_shape


class SAGEConvLink(SAGEConv):
    def forward(self, graph, feat):
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        def message_func(edges):
            weights = edges.data['edge_weight']
            src_data = edges.src['h']
            return_data = src_data * weights
            return {'m': return_data}

        def reduce_func(nodes):
            pass

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            graph.update_all(message_func, fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst  # same as above if homogeneous
            graph.update_all(message_func, fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(message_func, fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(message_func, self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


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
        self.layers.append(SAGEConvLink(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConvLink(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))  # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, graph, split_edge, optimizer):
    model.train()
    predictor.train()

    x = graph.ndata['feat']

    pos_train_edge = split_edge['train']['edge']

    optimizer.zero_grad()

    h = model(x)

    edge = pos_train_edge.t()

    pos_out = predictor(h[edge[0]], h[edge[1]])
    pos_loss = -torch.log(pos_out + 1e-15).mean()

    # Just do some trivial random sampling.
    edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                         device=x.device)
    neg_out = predictor(h[edge[0]], h[edge[1]])
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, predictor, graph, split_edge, evaluator):
    model.eval()
    predictor.eval()

    x = graph.ndata['feat']

    h = model(x)

    pos_train_edge = split_edge['train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    edge = pos_train_edge.t()
    pos_train_preds = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

    edge = pos_valid_edge.t()
    pos_valid_preds = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

    edge = neg_valid_edge.t()
    neg_valid_preds = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

    edge = pos_test_edge.t()
    pos_test_preds = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

    edge = neg_test_edge.t()
    neg_test_preds = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_preds,
            'y_pred_neg': neg_valid_preds,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_preds,
            'y_pred_neg': neg_valid_preds,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_preds,
            'y_pred_neg': neg_test_preds,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main(args):
    # load and preprocess dataset
    print('----loading dataset----')
    dataset = DglLinkPropPredDataset(name='ogbl-collab')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    x = graph.ndata['feat']
    print('>>> dataset loaded, name: {}, task: {}'.format(dataset_name, dataset_task))

    print('----Building Models----')

    model = GraphSAGE(g=graph, in_feats=x.size(-1), n_hidden=args.hidden_channels, n_classes=args.hidden_channels,
                      n_layers=args.num_layers, activation=None, dropout=args.dropout, aggregator_type='gcn')

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout)

    evaluator = Evaluator(name='ogbl-collab')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)

    print('----Training----')
    for epoch in range(1, 1 + args.epochs):
        t0 = time.time()
        loss = train(model, predictor, graph, split_edge, optimizer)
        t1 = time.time()

        if epoch % args.eval_steps == 0 or epoch == args.epochs:
            results = test(model, predictor, graph, split_edge, evaluator)
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(key)
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_hits:.2f}%, '
                      f'Valid: {100 * valid_hits:.2f}%, '
                      f'Test: {100 * test_hits:.2f}%'
                      f'Epoch Time: {t1 - t0}')
            print('------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    args = parser.parse_args()
    print(args)

    main(args)
