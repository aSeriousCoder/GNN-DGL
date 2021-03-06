import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset

from dgl.graph import DGLGraph
from dgl.contrib.sampling.sampler import NeighborSampler
from dgl.heterograph import DGLHeteroGraph


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
        :return: [由采样出的顶点们构成的子网（未合成版本)]，即[(二阶邻居+一阶邻居+seed，一阶邻居+seed), (一阶邻居+seed，seed)]
        '''
        seeds = th.LongTensor(np.asarray(seeds))
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


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # 此处仍然使用DGL封装好的层，聚合函数写死为“mean”，也可以像之前的版本进行变化
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            # 并没有实际构建一个局部网络，而是通过指定参与消息传递的结点和传递方向实现采样网络功能
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(
                    g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

                # 这个地方只用了一阶邻居，而且全部计算成本很大，未来需要改进一下

            x = y
        return y


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
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def main(args):
    # 配置执行器
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # Step.1 load and preprocess dataset，这部分和没有sample时无区别
    print('----loading dataset----')
    dataset = DglNodePropPredDataset(name='ogbn-products')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    print('>>> dataset loaded, name: {}, task: {}'.format(
        dataset_name, dataset_task))

    print('----processing data for training----')
    g = dataset.graph[0]
    features = g.ndata['feat']
    labels = dataset.labels.squeeze()
    in_feats = features.shape[1]
    n_classes = dataset.num_classes
    n_edges = g.number_of_edges()

    # 数据分割
    split_idx = dataset.get_idx_split()
    # nid：[id]，用于构造block（mini-batch）
    train_nid = split_idx['train']
    val_nid = split_idx['valid']
    test_nid = split_idx['test']
    # mask：[1,0]，用于分割图
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
    if hasattr(th, 'BoolTensor'):
        train_mask = th.BoolTensor(train_mask)
        val_mask = th.BoolTensor(val_mask)
        test_mask = th.BoolTensor(test_mask)
        full_mask = th.BoolTensor(full_mask)
    else:
        train_mask = th.ByteTensor(train_mask)
        val_mask = th.ByteTensor(val_mask)
        test_mask = th.ByteTensor(test_mask)
        full_mask = th.ByteTensor(full_mask)
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

    # Step.2 配置采样器

    # 将图转为herograph，否则和采样器的api对不起来
    g = dgl.graph(g.all_edges())
    g.ndata['features'] = features

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
    prepare_mp(g)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout)
                                  for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,  # 采样函数
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)  # 此处配置了并行采样

    # Step.3 配置模型、loss function和optimizer

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Step.4 批处理训练（batch）

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]  # 所有seed+邻居的ID
            seeds = blocks[-1].dstdata[dgl.NID]  # 采样seed的ID

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(
                model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    test_acc = evaluate(
        model, g, g.ndata['features'], labels, test_mask, args.batch_size, device)
    print('Test Acc {:.4f}'.format(test_acc))
    full_acc = evaluate(
        model, g, g.ndata['features'], labels, full_mask, args.batch_size, device)
    print('Full Acc {:.4f}'.format(full_acc))
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='10,25')  # 每一层采样的规模
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    # 此处可调整多线程采样
    args = argparser.parse_args()

    main(args)
