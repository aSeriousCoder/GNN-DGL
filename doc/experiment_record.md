



### FULL

```bash
Using backend: pytorch
Namespace(aggregator_type='gcn', dataset=None, dropout=0.5, gpu=-1, lr=0.01, n_epochs=100, n_hidden=16, n_layers=1, syn_gnp_n=1000, syn_gnp_p=0.0, syn_nclasses=10, syn_nfeats=500, syn_seed=42, syn_test_ratio=0.5, syn_train_ratio=0.1, syn_type='gnp', syn_val_ratio=0.2, weight_decay=0.0005)
----loading dataset----
>>> dataset loaded, name: ogbn-products, task: multiclass classification
----processing data for training----
----Data statistics------'
      #Edges 123718280
      #Classes 47
      #Train samples 196615
      #Val samples 39323
      #Test samples 2213091
----building train model----
----train start----
Epoch 00000 | Time(s) 144.2490 | Loss 3.9525 | Accuracy 0.0862 | ETputs(KTEPS) 857.67 
```



### Sample

```bash
Using backend: pytorch
----loading dataset----
>>> dataset loaded, name: ogbn-products, task: multiclass classification
----processing data for training----
----Data statistics------'
         #Edges 123718280
         #Classes 47
         #Train samples 196615
         #Val samples 39323
         #Test samples 2213091
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Epoch 00000 | Step 00000 | Loss 5.9565 | Train Acc 0.0120 | Speed (samples/sec) nan | GPU 0.0 MiB
Epoch 00000 | Step 00020 | Loss 3.4752 | Train Acc 0.1660 | Speed (samples/sec) 7055.7566 | GPU 0.0 MiB
Epoch 00000 | Step 00040 | Loss 2.4751 | Train Acc 0.4320 | Speed (samples/sec) 7311.4500 | GPU 0.0 MiB
Epoch 00000 | Step 00060 | Loss 1.8636 | Train Acc 0.5520 | Speed (samples/sec) 7365.4742 | GPU 0.0 MiB
Epoch 00000 | Step 00080 | Loss 1.5652 | Train Acc 0.5940 | Speed (samples/sec) 6937.0436 | GPU 0.0 MiB
Epoch 00000 | Step 00100 | Loss 1.3464 | Train Acc 0.6680 | Speed (samples/sec) 6908.9544 | GPU 0.0 MiB
Epoch 00000 | Step 00120 | Loss 1.2990 | Train Acc 0.6680 | Speed (samples/sec) 6900.0517 | GPU 0.0 MiB
Epoch 00000 | Step 00140 | Loss 1.2639 | Train Acc 0.6930 | Speed (samples/sec) 6794.7656 | GPU 0.0 MiB
Epoch 00000 | Step 00160 | Loss 1.0664 | Train Acc 0.7200 | Speed (samples/sec) 6801.7960 | GPU 0.0 MiB
Epoch 00000 | Step 00180 | Loss 1.0457 | Train Acc 0.7330 | Speed (samples/sec) 6779.7247 | GPU 0.0 MiB
Epoch Time(s): 81.2209
```



### Sample   num_worker=3

```bash
Using backend: pytorch
----loading dataset----
>>> dataset loaded, name: ogbn-products, task: multiclass classification
----processing data for training----
----Data statistics------'
         #Edges 123718280
         #Classes 47
         #Train samples 196615
         #Val samples 39323
         #Test samples 2213091
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Epoch 00000 | Step 00000 | Loss 5.5451 | Train Acc 0.0130 | Speed (samples/sec) nan | GPU 0.0 MiB
Epoch 00000 | Step 00020 | Loss 3.4023 | Train Acc 0.1580 | Speed (samples/sec) 6004.0902 | GPU 0.0 MiB
Epoch 00000 | Step 00040 | Loss 2.3429 | Train Acc 0.3900 | Speed (samples/sec) 6817.0109 | GPU 0.0 MiB
Epoch 00000 | Step 00060 | Loss 1.8813 | Train Acc 0.5020 | Speed (samples/sec) 7136.3890 | GPU 0.0 MiB
Epoch 00000 | Step 00080 | Loss 1.5615 | Train Acc 0.6030 | Speed (samples/sec) 7434.3897 | GPU 0.0 MiB
Epoch 00000 | Step 00100 | Loss 1.6158 | Train Acc 0.6090 | Speed (samples/sec) 7618.0141 | GPU 0.0 MiB
Epoch 00000 | Step 00120 | Loss 1.2090 | Train Acc 0.6870 | Speed (samples/sec) 7681.9407 | GPU 0.0 MiB
Epoch 00000 | Step 00140 | Loss 1.1294 | Train Acc 0.7200 | Speed (samples/sec) 7707.8821 | GPU 0.0 MiB
Epoch 00000 | Step 00160 | Loss 1.0523 | Train Acc 0.7390 | Speed (samples/sec) 7493.2281 | GPU 0.0 MiB
Epoch 00000 | Step 00180 | Loss 0.9750 | Train Acc 0.7640 | Speed (samples/sec) 7312.3345 | GPU 0.0 MiB
Epoch Time(s): 44.5098
```



### Sample   num_worker=6

```bash
Using backend: pytorch
----loading dataset----
>>> dataset loaded, name: ogbn-products, task: multiclass classification
----processing data for training----
----Data statistics------'
         #Edges 123718280
         #Classes 47
         #Train samples 196615
         #Val samples 39323
         #Test samples 2213091
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Epoch 00000 | Step 00000 | Loss 5.9072 | Train Acc 0.0110 | Speed (samples/sec) nan | GPU 0.0 MiB
Epoch 00000 | Step 00020 | Loss 3.4584 | Train Acc 0.1670 | Speed (samples/sec) 4749.8286 | GPU 0.0 MiB
Epoch 00000 | Step 00040 | Loss 2.6044 | Train Acc 0.4030 | Speed (samples/sec) 5328.0592 | GPU 0.0 MiB
Epoch 00000 | Step 00060 | Loss 1.8345 | Train Acc 0.5220 | Speed (samples/sec) 5493.7840 | GPU 0.0 MiB
Epoch 00000 | Step 00080 | Loss 1.5135 | Train Acc 0.5910 | Speed (samples/sec) 5579.7483 | GPU 0.0 MiB
Epoch 00000 | Step 00100 | Loss 1.3532 | Train Acc 0.6450 | Speed (samples/sec) 5607.5560 | GPU 0.0 MiB
Epoch 00000 | Step 00120 | Loss 1.3376 | Train Acc 0.6440 | Speed (samples/sec) 5632.9287 | GPU 0.0 MiB
Epoch 00000 | Step 00140 | Loss 1.1743 | Train Acc 0.7010 | Speed (samples/sec) 5640.8340 | GPU 0.0 MiB
Epoch 00000 | Step 00160 | Loss 1.1818 | Train Acc 0.6810 | Speed (samples/sec) 5552.3214 | GPU 0.0 MiB
Epoch 00000 | Step 00180 | Loss 1.0252 | Train Acc 0.7310 | Speed (samples/sec) 5514.3496 | GPU 0.0 MiB
Epoch Time(s): 44.3474
```



### Sample   num_worker=8

```bash
Using backend: pytorch
----loading dataset----
>>> dataset loaded, name: ogbn-products, task: multiclass classification
----processing data for training----
----Data statistics------'
         #Edges 123718280
         #Classes 47
         #Train samples 196615
         #Val samples 39323
         #Test samples 2213091
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3257: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/usr/local/anaconda3/lib/python3.7/site-packages/numpy/core/_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Epoch 00000 | Step 00000 | Loss 6.1079 | Train Acc 0.0090 | Speed (samples/sec) nan | GPU 0.0 MiB
Epoch 00000 | Step 00020 | Loss 3.7939 | Train Acc 0.1370 | Speed (samples/sec) 4584.3769 | GPU 0.0 MiB
Epoch 00000 | Step 00040 | Loss 2.6546 | Train Acc 0.3440 | Speed (samples/sec) 5135.1277 | GPU 0.0 MiB
Epoch 00000 | Step 00060 | Loss 2.0444 | Train Acc 0.5220 | Speed (samples/sec) 5402.3732 | GPU 0.0 MiB
Epoch 00000 | Step 00080 | Loss 1.7528 | Train Acc 0.5660 | Speed (samples/sec) 5507.9817 | GPU 0.0 MiB
Epoch 00000 | Step 00100 | Loss 1.5192 | Train Acc 0.6240 | Speed (samples/sec) 5601.4987 | GPU 0.0 MiB
Epoch 00000 | Step 00120 | Loss 1.4377 | Train Acc 0.5990 | Speed (samples/sec) 5659.6093 | GPU 0.0 MiB
Epoch 00000 | Step 00140 | Loss 1.1641 | Train Acc 0.7080 | Speed (samples/sec) 5703.1124 | GPU 0.0 MiB
Epoch 00000 | Step 00160 | Loss 1.1807 | Train Acc 0.6980 | Speed (samples/sec) 5724.3732 | GPU 0.0 MiB
Epoch 00000 | Step 00180 | Loss 0.9646 | Train Acc 0.7460 | Speed (samples/sec) 5756.1762 | GPU 0.0 MiB
Epoch Time(s): 43.4082
```





```python
# torch.utils.data.dataloader: DataLoader
def __iter__(self):
    if self.num_workers == 0:
        return _SingleProcessDataLoaderIter(self)
    else:
        return _MultiProcessingDataLoaderIter(self)
```



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
        :return: [由采样出的顶点们构成的子网（未合成版本）]
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

# Create sampler
sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

# Create PyTorch DataLoader for constructing blocks
dataloader = DataLoader(
    dataset=train_nid.numpy(),
    batch_size=args.batch_size,
    collate_fn=sampler.sample_blocks,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers)  # 此配置了并行采样
```

