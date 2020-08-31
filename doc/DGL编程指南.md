# Deep Graph Learning（DGL）

GitHub：https://github.com/dmlc/dgl

开发文档：https://docs.dgl.ai/

实验数据集：Open Graph Benchmark（[OGB](https://ogb.stanford.edu/)）



## DGL设计原理

### 核心思路：图上的消息传递

#### massage passing：

定义一个node向其目标node传递什么消息

```python
def message_func(edges):
		weights = edges.data['edge_weight'] # 带权图使用link weight对消息进行加权
		src_data = edges.src['h']
		return_data = src_data * weights
		return {'m': return_data}
```

#### massage aggregating:

定义一个node如何对接收到的消息进行聚合

```python
def reduce_func(nodes):
		pass
```

#### node updating

```python
graph.update_all(message_func, fn.mean('m', 'neigh'))
# 此处使用了自定义的message_func和封装好的reduce_func
```

通过以上三个函数，便可以定义一个GNN层的操作（即从其邻居结点获得什么信息，如何对这些信息进行聚合，然后更新自己的状态）



### Node Prediction任务

此类任务相对而言比较简单，使用封装好的组件就可以轻松实现一个模型，大体工作流如下：

1. 加载数据集并构造DGL的Graph
2. 定义GNN层和GNN模型
3. 训练的相关配置
4. 训练和测试



### Link Prediction任务

此处采取了使用GNN生成Node Embedding，然后再用一个神经网络预测是否存在边的架构，与业界思路基本一致，其他实现思路尚未实践。实现上就是在Node Prediction的基础上加了一个Link Predictor。



### Sampling

Sampling通过加装基于自定义采样器DataLoader和修改模型为批处理模式实现，采样器代码如下

```python
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
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
```

在main函数中添加

```python
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
```

之后便可以使用以下代码获取采样

```python
   # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
    for step, blocks in tqdm(enumerate(dataloader)):
```





## 实验代码

GitHub：https://github.com/aSeriousCoder/GNN-DGL

### Node Prediction(without sampling)

#### main函数

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability") # 弃用率，模拟消息传递过程中的随机损失
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu") # 默认不适用GPU
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
                        help="Aggregator type: mean/gcn/pool/lstm") # aggregate funcion选用的算法
    args = parser.parse_args()
    print(args)

    main(args)


def main(args):
  
    # step.1 load and preprocess dataset
    
    # 此处为从OGB直接获取数据集
    # 数据集较大，需要下载一段时间
    print('----loading dataset----')
    dataset = DglNodePropPredDataset(name='ogbn-products')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    print('>>> dataset loaded, name: {}, task: {}'.format(dataset_name, dataset_task))

    print('----processing data for training----')
    # 在node prediction任务中，train-test-valid的划分使用mask来进行，此处便是获取OGB分好的数据划分
    split_idx = dataset.get_idx_split() 
    g = dataset.graph[0] # 由于是node prediction任务，整个数据集就只有一张图
    features = g.ndata['feat']
    
    # 构造mask
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
    
    # 一些配置
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
        
        
    # Step.2 create learning model, loss function and optimizer
    
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
    
    # 使用封装好的loss function，也可以自定义
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    # Step.3 Training
    
    print('----train start----')
    dur = []
    for epoch in range(args.n_epochs):
        model.train() # 开启训练模式
        t0 = time.time()
        logits = model(features) # forward
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

```



#### 模型

```python
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
        self.layers = nn.ModuleList() # 设置模组流水线
        self.g = g # 倒入数据集

        # input layer
        self.layers.append(
          SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
              SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))
        # 以上都使用了DGL封装好的SAGEConv层，也可以自定义

    # 手动定义forward函数
    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h
```



#### 评估

```python
def evaluate(model, features, labels, mask):
    model.eval() # 开启测试模式
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
```





### Node Prediction(with sampling)

#### main函数

```python
if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='10,25') # 每一层采样的规模
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
    print('>>> dataset loaded, name: {}, task: {}'.format(dataset_name, dataset_task))

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
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks, # 采样函数
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)  # 此处配置了并行采样

    
    # Step.3 配置模型、loss function和optimizer
    
    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
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
            input_nodes = blocks[0].srcdata[dgl.NID] # 所有seed+邻居的ID
            seeds = blocks[-1].dstdata[dgl.NID] # 采样seed的ID

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    test_acc = evaluate(model, g, g.ndata['features'], labels, test_mask, args.batch_size, device)
    print('Test Acc {:.4f}'.format(test_acc))
    full_acc = evaluate(model, g, g.ndata['features'], labels, full_mask, args.batch_size, device)
    print('Full Acc {:.4f}'.format(full_acc))
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

```



#### 采样器

```python
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
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
```



#### 获取采样出的局部网络的featrue和seed的label

```python
def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels
```



#### 模型

```python
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
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
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

```



#### 评估函数

```python
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

```





### Link Prediction(without sampling)

Link Prediction的思路还是基于Node，首先用GNN获取Node的embedding，然后再通过一个Link Predictor预测两个点之间是否有边，属于一个双层训练架构。 

#### main函数

```python
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
    
    
def main(args):
  
    # Step.1 load and preprocess dataset
    
    print('----loading dataset----')
    dataset = DglLinkPropPredDataset(name='ogbl-collab')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    x = graph.ndata['feat']
    print('>>> dataset loaded, name: {}, task: {}'.format(dataset_name, dataset_task))

    # Step.2 配置模型
    
    print('----Building Models----')

    model = GraphSAGE(g=graph, in_feats=x.size(-1), n_hidden=args.hidden_channels,  n_classes=args.hidden_channels, n_layers=args.num_layers, activation=None, dropout=args.dropout, aggregator_type='gcn')

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
```



#### 模型

```python
class SAGEConvLink(SAGEConv):
    def forward(self, graph, feat):
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        # 自定义的消息传播函数
        def message_func(edges):
            weights = edges.data['edge_weight'] # 带权图使用link weight对消息进行加权
            src_data = edges.src['h']
            return_data = src_data * weights
            return {'m': return_data}

        # 自定义的消息聚合函数，此处没有使用
        def reduce_func(nodes):
            pass

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            graph.update_all(message_func, fn.mean('m', 'neigh')) # 缓存的名字必须对应
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

# 一个简单的神经网络，用于预测两点之间是否有边
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
        x = x_i * x_j # 使用两个embedding的乘积作为单输入
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
```



#### 训练和测试

```python
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

```





### Link Predicton(with sampling)

#### main函数

除需要额外配置采样器外和无采样版本相同

```python
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
    
def main(args):
    
    # Step.1 load and preprocess dataset
    
    print('----loading dataset----')
    dataset = DglLinkPropPredDataset(name='ogbl-collab')
    dataset_name = dataset.name
    dataset_task = dataset.task_type
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    x = graph.ndata['feat']
    print('>>> dataset loaded, name: {}, task: {}'.format(
        dataset_name, dataset_task))
    
    # step.2 配置采样器（和Node Prediction相同）

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
    
    # step.3 配置模型

    model = GraphSAGE(g=graph, in_feats=x.size(-1), n_hidden=args.hidden_channels, n_classes=args.hidden_channels,
                      n_layers=args.num_layers, activation=None, dropout=args.dropout, aggregator_type='gcn')

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout)

    evaluator = Evaluator(name='ogbl-collab')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)
    
    # step.4 训练

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

```



#### 采样器（和Node Prediction相同）

##### 采样设计

首先给定一些seed node，然后对其一阶邻居进行采样，由此获得一批pos pair，然后再继续采样，获得其各自的接收域用于GNN生成embedding

```python
# 采样并不是以边为元素进行的，而是以node为元素进行的（从图中采样出的一对邻居便是一对pos pair），neg pair可以采用整图随机抽样生成
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
```



#### 模型

```python
# SAGEConvLink和无采样版本相同
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
    # 构建与无采样版本相同
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
      	# 同时对两个输入进行forward，单个输入的forward和node pred基本相同
        # 也可以完全照搬Node pred的模型，不过由于link pred的样例是含有两个node的，因此要进行两次forward
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

# 与无采样版本相同
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
```



#### 训练&测试

```python
# 一次性拿出两个，也可以使用Node Pred中的此函数，分两次取出
def load_subtensor(g, input_nodes_l1, input_nodes_l0):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs_l1 = g.ndata['feat'][input_nodes_l1]
    batch_inputs_l0 = g.ndata['feat'][input_nodes_l0]
    return batch_inputs_l1, batch_inputs_l0

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
        把种子们的邻居找出来，用这些边进入predictor的学习
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


```

