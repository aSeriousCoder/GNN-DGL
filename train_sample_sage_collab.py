import argparse
import time
from random import randint
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from dgl.utils import check_eq_shape
from ogb.linkproppred import Evaluator
from ogb.linkproppred.dataset_dgl import DglLinkPropPredDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# 自定义采样器，但是很明显这个采样器并没有很多针对场景的配置，应该可以直接拿到别的地方用
class NeighborSampler(object):
    def __init__(self, g, fanouts):
        '''
        fanout : int or dict[etype, int]
        The number of sampled neighbors for each node on each edge type. Provide a dict
        to specify different fanout values for each edge type.
        定义了每一层的采样规模，比如[10, 25]就是原点采样10个邻居，然后这些邻居再采样他们的25个邻居，也就是说
        这个地方还定义了采样的层数
        '''
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        '''
        :param seeds:  采样的中心顶点
        :return: [由采样出的顶点们构成的子网（未合成版本）]
        '''
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(
                self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def load_subtensor(g, input_nodes_l1, input_nodes_l0):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs_l1 = g.ndata['feat'][input_nodes_l1]
    batch_inputs_l0 = g.ndata['feat'][input_nodes_l0]
    return batch_inputs_l1, batch_inputs_l0


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
            h_neigh = (graph.dstdata['neigh'] +
                       graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(message_func, fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(message_func, self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError(
                'Aggregator type {} not recognized.'.format(self._aggre_type))

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
        self.activation = activation
        self.dropout = dropout

        # input layer
        self.layers.append(SAGEConv(
            in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(
                n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))  # activation None

    def forward(self, blocks, x_l1, x_l0):
        h = x_l1
        '''
        显然一个block是一个单层网
        '''
        for l, (layer, block) in enumerate(zip(self.layers, blocks[:-1])):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                if self.activation:
                    h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h_neib = h

        h = x_l0
        '''
        显然一个block是一个单层网
        '''
        for l, (layer, block) in enumerate(zip(self.layers, blocks[1:])):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                if self.activation:
                    h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h_seed = h

        return h_seed, h_neib

    def inference(self, x, ):
        h = x
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


def train(model, predictor, graph, split_edge, optimizer, dataloader):
    model.train()
    predictor.train()

    # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
    for step, blocks in tqdm(enumerate(dataloader)):

        '''
        此处才真正组成子网，进行minibatch训练
        '''
        # The nodes for input lies at the LHS side of the first block.
        # The nodes for output lies at the RHS side of the last block.
        input_nodes_l1 = blocks[0].srcdata[dgl.NID]
        input_nodes_l0 = blocks[1].srcdata[dgl.NID]
        batch_inputs_l1, batch_inputs_l0 = load_subtensor(graph, input_nodes_l1, input_nodes_l0)

        optimizer.zero_grad()

        '''
        id可能还有一些问题
        edge的形式也有问题
        '''

        # Compute loss and prediction
        seed_pred, neib_pred = model(blocks, batch_inputs_l1, batch_inputs_l0)
        pred = neib_pred
        pred[:len(seed_pred)] = seed_pred

        ''''
        把种子们的邻居找出来，用这些边进入下一轮学习
        '''
        edges_from_seed = blocks[-1].edges()

        pos_out = predictor(pred[edges_from_seed[0]], pred[edges_from_seed[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        g = blocks[-3]
        num_of_node = blocks[-2].number_of_dst_nodes('_N')
        edge_neg = [[], []]
        while len(edge_neg[0]) < len(edges_from_seed[0]):
            node1 = randint(0, num_of_node - 1)
            node2 = randint(0, num_of_node - 1)
            while node2 == node1:
                node2 = randint(0, num_of_node)
            if not g.has_edge_between(node1, node2, '_E'):
                edge_neg[0].append(node1)
                edge_neg[1].append(node2)
        neg_out = predictor(pred[edge_neg[0]], pred[edge_neg[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, predictor, graph, split_edge, evaluator, dataloader):
    model.eval()
    predictor.eval()

    x = graph.ndata['feat']

    h = model.inference(x)

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
    print('>>> dataset loaded, name: {}, task: {}'.format(
        dataset_name, dataset_task))

    print('----Building Models----')

    # 转为herograph，否则和采样器的api对不起来
    g = dgl.graph(graph.all_edges())
    g.ndata['features'] = x
    prepare_mp(g)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout)
                                  for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=np.array(range(g.number_of_nodes())),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)  # 此配置了并行采样

    model = GraphSAGE(g=graph, in_feats=x.size(-1), n_hidden=args.hidden_channels, n_classes=args.hidden_channels,
                      n_layers=args.num_layers, activation=None, dropout=args.dropout, aggregator_type='gcn')

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout)

    evaluator = Evaluator(name='ogbl-collab')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)

    print('----Training----')
    for epoch in range(1, 1 + args.epochs):
        t0 = time.time()
        loss = train(model, predictor, graph, split_edge, optimizer, dataloader)
        t1 = time.time()

        if epoch % args.eval_steps == 0 or epoch == args.epochs:
            results = test(model, predictor, graph, split_edge, evaluator, dataloader)
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
    parser.add_argument('--fan-out', type=str, default='10,10,10,10,10')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    print(args)

    main(args)
