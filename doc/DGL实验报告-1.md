# DGL实验报告-1



## 实验环境

设备：Macbook Pro 13‘ （2018）

数据集：ogbn-products

主要依赖：pytorch + DGL

模型：GraphSAGE

代码仓库：https://github.com/aSeriousCoder/GNN-DGL

<!--本来想做一个docker image，但是docker container联网有点问题，暂时作罢-->



## 实验内容

1. 使用DGL在Benchmark（OGB）数据集上**完整运行GNN的training, validation, inference全过程**
2. 探索DGL-**Sampling**的使用
3. 探索DGL-Sampling的CPU**多核并行加速**情况（无GPU资源）



## 运行结果

<!--此处只记录第一个Epoch的日志，用于对比模型训练效果和效率-->

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

可以看到，使用**采样-minibatch**训练时，训练速度明显加快；启动多线程后又可以进一步加速，但是并非线程越多越快（可能受到了本机环境的影响）



## 分析

在**PyTorch**提供的数据加载器中找到了（还有很多地方也表明了对多线程的支持，此处比较清晰）

```python
# torch.utils.data.dataloader: DataLoader
def __iter__(self):
    if self.num_workers == 0:
        return _SingleProcessDataLoaderIter(self)
    else:
        return _MultiProcessingDataLoaderIter(self)
```

PyTorch的这个加载器在多线程采样时维护了多线程上下文，然后是在生成**同一Block**时生效的。

所以在采样时就可以很方便地用起来并行加速。

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



## 完整运行 Sample   num_worker=3 结果

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
Epoch 00000 | Step 00000 | Loss 5.3401 | Train Acc 0.0240 | Speed (samples/sec) nan | GPU 0.0 MiB
Epoch 00000 | Step 00020 | Loss 3.1619 | Train Acc 0.2710 | Speed (samples/sec) 5293.2378 | GPU 0.0 MiB
Epoch 00000 | Step 00040 | Loss 2.3504 | Train Acc 0.4070 | Speed (samples/sec) 6256.0820 | GPU 0.0 MiB
Epoch 00000 | Step 00060 | Loss 1.8358 | Train Acc 0.5470 | Speed (samples/sec) 6481.7465 | GPU 0.0 MiB
Epoch 00000 | Step 00080 | Loss 1.6226 | Train Acc 0.5560 | Speed (samples/sec) 5696.2185 | GPU 0.0 MiB
Epoch 00000 | Step 00100 | Loss 1.3571 | Train Acc 0.6550 | Speed (samples/sec) 5432.7939 | GPU 0.0 MiB
Epoch 00000 | Step 00120 | Loss 1.2592 | Train Acc 0.6660 | Speed (samples/sec) 5261.0063 | GPU 0.0 MiB
Epoch 00000 | Step 00140 | Loss 1.1873 | Train Acc 0.6940 | Speed (samples/sec) 5149.4418 | GPU 0.0 MiB
Epoch 00000 | Step 00160 | Loss 1.0406 | Train Acc 0.7200 | Speed (samples/sec) 5143.3849 | GPU 0.0 MiB
Epoch 00000 | Step 00180 | Loss 0.9448 | Train Acc 0.7530 | Speed (samples/sec) 5069.7227 | GPU 0.0 MiB
Epoch Time(s): 52.9029
Epoch 00001 | Step 00000 | Loss 0.8457 | Train Acc 0.7720 | Speed (samples/sec) 5089.3105 | GPU 0.0 MiB
Epoch 00001 | Step 00020 | Loss 0.8873 | Train Acc 0.7740 | Speed (samples/sec) 5028.9685 | GPU 0.0 MiB
Epoch 00001 | Step 00040 | Loss 0.8144 | Train Acc 0.7830 | Speed (samples/sec) 5048.0274 | GPU 0.0 MiB
Epoch 00001 | Step 00060 | Loss 0.8696 | Train Acc 0.7750 | Speed (samples/sec) 5085.3786 | GPU 0.0 MiB
Epoch 00001 | Step 00080 | Loss 0.8348 | Train Acc 0.7870 | Speed (samples/sec) 5087.1661 | GPU 0.0 MiB
Epoch 00001 | Step 00100 | Loss 0.9053 | Train Acc 0.7650 | Speed (samples/sec) 5115.1679 | GPU 0.0 MiB
Epoch 00001 | Step 00120 | Loss 0.7943 | Train Acc 0.7830 | Speed (samples/sec) 5147.5575 | GPU 0.0 MiB
Epoch 00001 | Step 00140 | Loss 0.7725 | Train Acc 0.7850 | Speed (samples/sec) 5183.6334 | GPU 0.0 MiB
Epoch 00001 | Step 00160 | Loss 0.7111 | Train Acc 0.8120 | Speed (samples/sec) 5192.4179 | GPU 0.0 MiB
Epoch 00001 | Step 00180 | Loss 0.7609 | Train Acc 0.8020 | Speed (samples/sec) 5200.1851 | GPU 0.0 MiB
Epoch Time(s): 41.6561
Epoch 00002 | Step 00000 | Loss 0.6832 | Train Acc 0.8150 | Speed (samples/sec) 5218.6851 | GPU 0.0 MiB
Epoch 00002 | Step 00020 | Loss 0.8085 | Train Acc 0.8220 | Speed (samples/sec) 5199.3891 | GPU 0.0 MiB
Epoch 00002 | Step 00040 | Loss 0.7708 | Train Acc 0.8040 | Speed (samples/sec) 5202.1251 | GPU 0.0 MiB
Epoch 00002 | Step 00060 | Loss 0.6971 | Train Acc 0.8200 | Speed (samples/sec) 5207.5278 | GPU 0.0 MiB
Epoch 00002 | Step 00080 | Loss 0.6994 | Train Acc 0.8160 | Speed (samples/sec) 5227.2398 | GPU 0.0 MiB
Epoch 00002 | Step 00100 | Loss 0.6481 | Train Acc 0.8340 | Speed (samples/sec) 5228.7979 | GPU 0.0 MiB
Epoch 00002 | Step 00120 | Loss 0.7315 | Train Acc 0.8010 | Speed (samples/sec) 5247.1831 | GPU 0.0 MiB
Epoch 00002 | Step 00140 | Loss 0.6069 | Train Acc 0.8390 | Speed (samples/sec) 5262.6212 | GPU 0.0 MiB
Epoch 00002 | Step 00160 | Loss 0.6532 | Train Acc 0.8350 | Speed (samples/sec) 5279.9505 | GPU 0.0 MiB
Epoch 00002 | Step 00180 | Loss 0.6224 | Train Acc 0.8300 | Speed (samples/sec) 5300.5383 | GPU 0.0 MiB
Epoch Time(s): 39.8219
Epoch 00003 | Step 00000 | Loss 0.6332 | Train Acc 0.8330 | Speed (samples/sec) 5317.3799 | GPU 0.0 MiB
Epoch 00003 | Step 00020 | Loss 0.6325 | Train Acc 0.8390 | Speed (samples/sec) 5319.1487 | GPU 0.0 MiB
Epoch 00003 | Step 00040 | Loss 0.6991 | Train Acc 0.8160 | Speed (samples/sec) 5332.6513 | GPU 0.0 MiB
Epoch 00003 | Step 00060 | Loss 0.7003 | Train Acc 0.8200 | Speed (samples/sec) 5341.1068 | GPU 0.0 MiB
Epoch 00003 | Step 00080 | Loss 0.6202 | Train Acc 0.8330 | Speed (samples/sec) 5313.2482 | GPU 0.0 MiB
Epoch 00003 | Step 00100 | Loss 0.6588 | Train Acc 0.8240 | Speed (samples/sec) 5196.7398 | GPU 0.0 MiB
Epoch 00003 | Step 00120 | Loss 0.5728 | Train Acc 0.8540 | Speed (samples/sec) 5105.6887 | GPU 0.0 MiB
Epoch 00003 | Step 00140 | Loss 0.6836 | Train Acc 0.8270 | Speed (samples/sec) 5092.7819 | GPU 0.0 MiB
Epoch 00003 | Step 00160 | Loss 0.6718 | Train Acc 0.8280 | Speed (samples/sec) 5101.3505 | GPU 0.0 MiB
Epoch 00003 | Step 00180 | Loss 0.6831 | Train Acc 0.8260 | Speed (samples/sec) 5108.3511 | GPU 0.0 MiB
Epoch Time(s): 73.5100
Epoch 00004 | Step 00000 | Loss 0.5535 | Train Acc 0.8430 | Speed (samples/sec) 5110.8131 | GPU 0.0 MiB
Epoch 00004 | Step 00020 | Loss 0.5930 | Train Acc 0.8400 | Speed (samples/sec) 5124.4178 | GPU 0.0 MiB
Epoch 00004 | Step 00040 | Loss 0.6503 | Train Acc 0.8360 | Speed (samples/sec) 5139.9976 | GPU 0.0 MiB
Epoch 00004 | Step 00060 | Loss 0.6373 | Train Acc 0.8320 | Speed (samples/sec) 5147.4134 | GPU 0.0 MiB
Epoch 00004 | Step 00080 | Loss 0.5468 | Train Acc 0.8610 | Speed (samples/sec) 5149.5934 | GPU 0.0 MiB
Epoch 00004 | Step 00100 | Loss 0.6390 | Train Acc 0.8380 | Speed (samples/sec) 5162.1006 | GPU 0.0 MiB
Epoch 00004 | Step 00120 | Loss 0.6663 | Train Acc 0.8350 | Speed (samples/sec) 5171.8064 | GPU 0.0 MiB
Epoch 00004 | Step 00140 | Loss 0.5695 | Train Acc 0.8530 | Speed (samples/sec) 5178.8367 | GPU 0.0 MiB
Epoch 00004 | Step 00160 | Loss 0.5954 | Train Acc 0.8480 | Speed (samples/sec) 5177.4354 | GPU 0.0 MiB
Epoch 00004 | Step 00180 | Loss 0.5405 | Train Acc 0.8550 | Speed (samples/sec) 5187.8982 | GPU 0.0 MiB
Epoch Time(s): 38.9022
Epoch 00005 | Step 00000 | Loss 0.6152 | Train Acc 0.8420 | Speed (samples/sec) 5196.4770 | GPU 0.0 MiB
Epoch 00005 | Step 00020 | Loss 0.5589 | Train Acc 0.8460 | Speed (samples/sec) 5190.2539 | GPU 0.0 MiB
Epoch 00005 | Step 00040 | Loss 0.5627 | Train Acc 0.8560 | Speed (samples/sec) 5189.8808 | GPU 0.0 MiB
Epoch 00005 | Step 00060 | Loss 0.5709 | Train Acc 0.8500 | Speed (samples/sec) 5189.0625 | GPU 0.0 MiB
Epoch 00005 | Step 00080 | Loss 0.5479 | Train Acc 0.8360 | Speed (samples/sec) 5192.9005 | GPU 0.0 MiB
Epoch 00005 | Step 00100 | Loss 0.6244 | Train Acc 0.8510 | Speed (samples/sec) 5200.4567 | GPU 0.0 MiB
Epoch 00005 | Step 00120 | Loss 0.5319 | Train Acc 0.8610 | Speed (samples/sec) 5212.6694 | GPU 0.0 MiB
Epoch 00005 | Step 00140 | Loss 0.5724 | Train Acc 0.8430 | Speed (samples/sec) 5221.8062 | GPU 0.0 MiB
Epoch 00005 | Step 00160 | Loss 0.5892 | Train Acc 0.8450 | Speed (samples/sec) 5228.8233 | GPU 0.0 MiB
Epoch 00005 | Step 00180 | Loss 0.5291 | Train Acc 0.8530 | Speed (samples/sec) 5239.4186 | GPU 0.0 MiB
Epoch Time(s): 40.6781
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [03:24<00:00, 11.98it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [02:28<00:00, 16.50it/s]
Eval Acc 0.8707
Epoch 00006 | Step 00000 | Loss 0.6181 | Train Acc 0.8210 | Speed (samples/sec) 5229.6032 | GPU 0.0 MiB
Epoch 00006 | Step 00020 | Loss 0.5934 | Train Acc 0.8410 | Speed (samples/sec) 5225.3418 | GPU 0.0 MiB
Epoch 00006 | Step 00040 | Loss 0.5801 | Train Acc 0.8440 | Speed (samples/sec) 5230.5494 | GPU 0.0 MiB
Epoch 00006 | Step 00060 | Loss 0.5936 | Train Acc 0.8350 | Speed (samples/sec) 5237.8944 | GPU 0.0 MiB
Epoch 00006 | Step 00080 | Loss 0.5763 | Train Acc 0.8550 | Speed (samples/sec) 5242.9987 | GPU 0.0 MiB
Epoch 00006 | Step 00100 | Loss 0.6685 | Train Acc 0.8360 | Speed (samples/sec) 5243.6482 | GPU 0.0 MiB
Epoch 00006 | Step 00120 | Loss 0.5786 | Train Acc 0.8460 | Speed (samples/sec) 5225.2044 | GPU 0.0 MiB
Epoch 00006 | Step 00140 | Loss 0.6983 | Train Acc 0.8270 | Speed (samples/sec) 5210.1093 | GPU 0.0 MiB
Epoch 00006 | Step 00160 | Loss 0.5334 | Train Acc 0.8540 | Speed (samples/sec) 5214.4590 | GPU 0.0 MiB
Epoch 00006 | Step 00180 | Loss 0.5676 | Train Acc 0.8400 | Speed (samples/sec) 5222.5702 | GPU 0.0 MiB
Epoch Time(s): 46.9943
Epoch 00007 | Step 00000 | Loss 0.6615 | Train Acc 0.8390 | Speed (samples/sec) 5231.9042 | GPU 0.0 MiB
Epoch 00007 | Step 00020 | Loss 0.5628 | Train Acc 0.8410 | Speed (samples/sec) 5240.1839 | GPU 0.0 MiB
Epoch 00007 | Step 00040 | Loss 0.6122 | Train Acc 0.8390 | Speed (samples/sec) 5249.1269 | GPU 0.0 MiB
Epoch 00007 | Step 00060 | Loss 0.5844 | Train Acc 0.8520 | Speed (samples/sec) 5255.2820 | GPU 0.0 MiB
Epoch 00007 | Step 00080 | Loss 0.5243 | Train Acc 0.8510 | Speed (samples/sec) 5263.0053 | GPU 0.0 MiB
Epoch 00007 | Step 00100 | Loss 0.5700 | Train Acc 0.8570 | Speed (samples/sec) 5271.1055 | GPU 0.0 MiB
Epoch 00007 | Step 00120 | Loss 0.5606 | Train Acc 0.8460 | Speed (samples/sec) 5280.0429 | GPU 0.0 MiB
Epoch 00007 | Step 00140 | Loss 0.6013 | Train Acc 0.8480 | Speed (samples/sec) 5288.5880 | GPU 0.0 MiB
Epoch 00007 | Step 00160 | Loss 0.6349 | Train Acc 0.8420 | Speed (samples/sec) 5295.6921 | GPU 0.0 MiB
Epoch 00007 | Step 00180 | Loss 0.4993 | Train Acc 0.8570 | Speed (samples/sec) 5302.7509 | GPU 0.0 MiB
Epoch Time(s): 36.3522
Epoch 00008 | Step 00000 | Loss 0.6292 | Train Acc 0.8500 | Speed (samples/sec) 5310.3684 | GPU 0.0 MiB
Epoch 00008 | Step 00020 | Loss 0.4991 | Train Acc 0.8760 | Speed (samples/sec) 5318.5386 | GPU 0.0 MiB
Epoch 00008 | Step 00040 | Loss 0.6272 | Train Acc 0.8280 | Speed (samples/sec) 5326.2236 | GPU 0.0 MiB
Epoch 00008 | Step 00060 | Loss 0.6302 | Train Acc 0.8430 | Speed (samples/sec) 5334.8044 | GPU 0.0 MiB
Epoch 00008 | Step 00080 | Loss 0.6481 | Train Acc 0.8370 | Speed (samples/sec) 5343.8837 | GPU 0.0 MiB
Epoch 00008 | Step 00100 | Loss 0.5722 | Train Acc 0.8420 | Speed (samples/sec) 5352.6771 | GPU 0.0 MiB
Epoch 00008 | Step 00120 | Loss 0.5397 | Train Acc 0.8520 | Speed (samples/sec) 5360.4720 | GPU 0.0 MiB
Epoch 00008 | Step 00140 | Loss 0.5313 | Train Acc 0.8620 | Speed (samples/sec) 5362.8450 | GPU 0.0 MiB
Epoch 00008 | Step 00160 | Loss 0.5849 | Train Acc 0.8460 | Speed (samples/sec) 5367.8469 | GPU 0.0 MiB
Epoch 00008 | Step 00180 | Loss 0.5298 | Train Acc 0.8390 | Speed (samples/sec) 5370.0600 | GPU 0.0 MiB
Epoch Time(s): 36.0993
Epoch 00009 | Step 00000 | Loss 0.5630 | Train Acc 0.8460 | Speed (samples/sec) 5376.6966 | GPU 0.0 MiB
Epoch 00009 | Step 00020 | Loss 0.5475 | Train Acc 0.8480 | Speed (samples/sec) 5383.8676 | GPU 0.0 MiB
Epoch 00009 | Step 00040 | Loss 0.6019 | Train Acc 0.8430 | Speed (samples/sec) 5391.0825 | GPU 0.0 MiB
Epoch 00009 | Step 00060 | Loss 0.6705 | Train Acc 0.8320 | Speed (samples/sec) 5398.1645 | GPU 0.0 MiB
Epoch 00009 | Step 00080 | Loss 0.5705 | Train Acc 0.8560 | Speed (samples/sec) 5404.8109 | GPU 0.0 MiB
Epoch 00009 | Step 00100 | Loss 0.5661 | Train Acc 0.8630 | Speed (samples/sec) 5398.4922 | GPU 0.0 MiB
Epoch 00009 | Step 00120 | Loss 0.5712 | Train Acc 0.8540 | Speed (samples/sec) 5404.0185 | GPU 0.0 MiB
Epoch 00009 | Step 00140 | Loss 0.5620 | Train Acc 0.8500 | Speed (samples/sec) 5409.2712 | GPU 0.0 MiB
Epoch 00009 | Step 00160 | Loss 0.5876 | Train Acc 0.8380 | Speed (samples/sec) 5415.3664 | GPU 0.0 MiB
Epoch 00009 | Step 00180 | Loss 0.5920 | Train Acc 0.8490 | Speed (samples/sec) 5421.7159 | GPU 0.0 MiB
Epoch Time(s): 37.6953
Epoch 00010 | Step 00000 | Loss 0.6296 | Train Acc 0.8230 | Speed (samples/sec) 5422.3859 | GPU 0.0 MiB
Epoch 00010 | Step 00020 | Loss 0.5165 | Train Acc 0.8800 | Speed (samples/sec) 5424.8958 | GPU 0.0 MiB
Epoch 00010 | Step 00040 | Loss 0.5388 | Train Acc 0.8590 | Speed (samples/sec) 5424.7811 | GPU 0.0 MiB
Epoch 00010 | Step 00060 | Loss 0.6079 | Train Acc 0.8500 | Speed (samples/sec) 5428.2873 | GPU 0.0 MiB
Epoch 00010 | Step 00080 | Loss 0.4900 | Train Acc 0.8810 | Speed (samples/sec) 5431.6503 | GPU 0.0 MiB
Epoch 00010 | Step 00100 | Loss 0.5105 | Train Acc 0.8640 | Speed (samples/sec) 5430.8819 | GPU 0.0 MiB
Epoch 00010 | Step 00120 | Loss 0.4985 | Train Acc 0.8570 | Speed (samples/sec) 5435.5188 | GPU 0.0 MiB
Epoch 00010 | Step 00140 | Loss 0.5680 | Train Acc 0.8630 | Speed (samples/sec) 5440.2338 | GPU 0.0 MiB
Epoch 00010 | Step 00160 | Loss 0.5173 | Train Acc 0.8650 | Speed (samples/sec) 5444.6925 | GPU 0.0 MiB
Epoch 00010 | Step 00180 | Loss 0.5412 | Train Acc 0.8550 | Speed (samples/sec) 5449.7733 | GPU 0.0 MiB
Epoch Time(s): 37.1251
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [03:32<00:00, 11.54it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [02:32<00:00, 16.08it/s]
Eval Acc 0.8769
Epoch 00011 | Step 00000 | Loss 0.6087 | Train Acc 0.8530 | Speed (samples/sec) 5451.6331 | GPU 0.0 MiB
Epoch 00011 | Step 00020 | Loss 0.5479 | Train Acc 0.8680 | Speed (samples/sec) 5442.1251 | GPU 0.0 MiB
Epoch 00011 | Step 00040 | Loss 0.5385 | Train Acc 0.8600 | Speed (samples/sec) 5443.3545 | GPU 0.0 MiB
Epoch 00011 | Step 00060 | Loss 0.5701 | Train Acc 0.8480 | Speed (samples/sec) 5447.9328 | GPU 0.0 MiB
Epoch 00011 | Step 00080 | Loss 0.5036 | Train Acc 0.8620 | Speed (samples/sec) 5452.0994 | GPU 0.0 MiB
Epoch 00011 | Step 00100 | Loss 0.5189 | Train Acc 0.8570 | Speed (samples/sec) 5455.8987 | GPU 0.0 MiB
Epoch 00011 | Step 00120 | Loss 0.5935 | Train Acc 0.8350 | Speed (samples/sec) 5459.6299 | GPU 0.0 MiB
Epoch 00011 | Step 00140 | Loss 0.5371 | Train Acc 0.8630 | Speed (samples/sec) 5462.9965 | GPU 0.0 MiB
Epoch 00011 | Step 00160 | Loss 0.5552 | Train Acc 0.8620 | Speed (samples/sec) 5461.7496 | GPU 0.0 MiB
Epoch 00011 | Step 00180 | Loss 0.5720 | Train Acc 0.8280 | Speed (samples/sec) 5462.3172 | GPU 0.0 MiB
Epoch Time(s): 42.7688
Epoch 00012 | Step 00000 | Loss 0.5612 | Train Acc 0.8410 | Speed (samples/sec) 5466.2056 | GPU 0.0 MiB
Epoch 00012 | Step 00020 | Loss 0.5474 | Train Acc 0.8520 | Speed (samples/sec) 5467.9488 | GPU 0.0 MiB
Epoch 00012 | Step 00040 | Loss 0.5652 | Train Acc 0.8530 | Speed (samples/sec) 5470.5638 | GPU 0.0 MiB
Epoch 00012 | Step 00060 | Loss 0.5078 | Train Acc 0.8560 | Speed (samples/sec) 5474.4240 | GPU 0.0 MiB
Epoch 00012 | Step 00080 | Loss 0.5423 | Train Acc 0.8630 | Speed (samples/sec) 5478.0699 | GPU 0.0 MiB
Epoch 00012 | Step 00100 | Loss 0.5832 | Train Acc 0.8410 | Speed (samples/sec) 5481.4504 | GPU 0.0 MiB
Epoch 00012 | Step 00120 | Loss 0.6202 | Train Acc 0.8280 | Speed (samples/sec) 5485.0519 | GPU 0.0 MiB
Epoch 00012 | Step 00140 | Loss 0.6528 | Train Acc 0.8300 | Speed (samples/sec) 5488.7814 | GPU 0.0 MiB
Epoch 00012 | Step 00160 | Loss 0.5512 | Train Acc 0.8550 | Speed (samples/sec) 5492.7428 | GPU 0.0 MiB
Epoch 00012 | Step 00180 | Loss 0.5038 | Train Acc 0.8630 | Speed (samples/sec) 5489.9345 | GPU 0.0 MiB
Epoch Time(s): 37.3602
Epoch 00013 | Step 00000 | Loss 0.5325 | Train Acc 0.8630 | Speed (samples/sec) 5493.9765 | GPU 0.0 MiB
Epoch 00013 | Step 00020 | Loss 0.5457 | Train Acc 0.8560 | Speed (samples/sec) 5489.2907 | GPU 0.0 MiB
Epoch 00013 | Step 00040 | Loss 0.5181 | Train Acc 0.8600 | Speed (samples/sec) 5475.6272 | GPU 0.0 MiB
Epoch 00013 | Step 00060 | Loss 0.4930 | Train Acc 0.8700 | Speed (samples/sec) 5474.4355 | GPU 0.0 MiB
Epoch 00013 | Step 00080 | Loss 0.5428 | Train Acc 0.8450 | Speed (samples/sec) 5476.5534 | GPU 0.0 MiB
Epoch 00013 | Step 00100 | Loss 0.5596 | Train Acc 0.8400 | Speed (samples/sec) 5479.3583 | GPU 0.0 MiB
Epoch 00013 | Step 00120 | Loss 0.5865 | Train Acc 0.8540 | Speed (samples/sec) 5481.5199 | GPU 0.0 MiB
Epoch 00013 | Step 00140 | Loss 0.5078 | Train Acc 0.8490 | Speed (samples/sec) 5484.0163 | GPU 0.0 MiB
Epoch 00013 | Step 00160 | Loss 0.5855 | Train Acc 0.8380 | Speed (samples/sec) 5486.7390 | GPU 0.0 MiB
Epoch 00013 | Step 00180 | Loss 0.5954 | Train Acc 0.8560 | Speed (samples/sec) 5489.6849 | GPU 0.0 MiB
Epoch Time(s): 41.7086
Epoch 00014 | Step 00000 | Loss 0.6103 | Train Acc 0.8260 | Speed (samples/sec) 5493.0104 | GPU 0.0 MiB
Epoch 00014 | Step 00020 | Loss 0.5472 | Train Acc 0.8590 | Speed (samples/sec) 5495.0229 | GPU 0.0 MiB
Epoch 00014 | Step 00040 | Loss 0.5289 | Train Acc 0.8700 | Speed (samples/sec) 5498.0750 | GPU 0.0 MiB
Epoch 00014 | Step 00060 | Loss 0.6082 | Train Acc 0.8410 | Speed (samples/sec) 5501.0887 | GPU 0.0 MiB
Epoch 00014 | Step 00080 | Loss 0.6005 | Train Acc 0.8460 | Speed (samples/sec) 5504.0147 | GPU 0.0 MiB
Epoch 00014 | Step 00100 | Loss 0.5233 | Train Acc 0.8550 | Speed (samples/sec) 5506.7417 | GPU 0.0 MiB
Epoch 00014 | Step 00120 | Loss 0.5503 | Train Acc 0.8480 | Speed (samples/sec) 5508.2303 | GPU 0.0 MiB
Epoch 00014 | Step 00140 | Loss 0.5414 | Train Acc 0.8380 | Speed (samples/sec) 5511.4654 | GPU 0.0 MiB
Epoch 00014 | Step 00160 | Loss 0.5154 | Train Acc 0.8800 | Speed (samples/sec) 5514.0413 | GPU 0.0 MiB
Epoch 00014 | Step 00180 | Loss 0.6185 | Train Acc 0.8360 | Speed (samples/sec) 5517.0419 | GPU 0.0 MiB
Epoch Time(s): 36.1628
Epoch 00015 | Step 00000 | Loss 0.5297 | Train Acc 0.8480 | Speed (samples/sec) 5520.0300 | GPU 0.0 MiB
Epoch 00015 | Step 00020 | Loss 0.5957 | Train Acc 0.8490 | Speed (samples/sec) 5522.3035 | GPU 0.0 MiB
Epoch 00015 | Step 00040 | Loss 0.5876 | Train Acc 0.8420 | Speed (samples/sec) 5524.6484 | GPU 0.0 MiB
Epoch 00015 | Step 00060 | Loss 0.5631 | Train Acc 0.8480 | Speed (samples/sec) 5526.0472 | GPU 0.0 MiB
Epoch 00015 | Step 00080 | Loss 0.5951 | Train Acc 0.8530 | Speed (samples/sec) 5513.7901 | GPU 0.0 MiB
Epoch 00015 | Step 00100 | Loss 0.5607 | Train Acc 0.8660 | Speed (samples/sec) 5494.5598 | GPU 0.0 MiB
Epoch 00015 | Step 00120 | Loss 0.5223 | Train Acc 0.8630 | Speed (samples/sec) 5470.8998 | GPU 0.0 MiB
Epoch 00015 | Step 00140 | Loss 0.5415 | Train Acc 0.8650 | Speed (samples/sec) 5453.9182 | GPU 0.0 MiB
Epoch 00015 | Step 00160 | Loss 0.5467 | Train Acc 0.8730 | Speed (samples/sec) 5431.1768 | GPU 0.0 MiB
Epoch 00015 | Step 00180 | Loss 0.5661 | Train Acc 0.8330 | Speed (samples/sec) 5414.6912 | GPU 0.0 MiB
Epoch Time(s): 98.9501
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [03:26<00:00, 11.85it/s]
100%|███████████████████████████████████████████████████████████████████████████████████|  2450/2450 [02:13<00:00, 18.39it/s]
Eval Acc 0.8773
Epoch 00016 | Step 00000 | Loss 0.5796 | Train Acc 0.8570 | Speed (samples/sec) 5410.6502 | GPU 0.0 MiB
Epoch 00016 | Step 00020 | Loss 0.5283 | Train Acc 0.8550 | Speed (samples/sec) 5414.1572 | GPU 0.0 MiB
Epoch 00016 | Step 00040 | Loss 0.5030 | Train Acc 0.8600 | Speed (samples/sec) 5418.9281 | GPU 0.0 MiB
Epoch 00016 | Step 00060 | Loss 0.4877 | Train Acc 0.8710 | Speed (samples/sec) 5424.1672 | GPU 0.0 MiB
Epoch 00016 | Step 00080 | Loss 0.5214 | Train Acc 0.8660 | Speed (samples/sec) 5429.0063 | GPU 0.0 MiB
Epoch 00016 | Step 00100 | Loss 0.6172 | Train Acc 0.8250 | Speed (samples/sec) 5433.9245 | GPU 0.0 MiB
Epoch 00016 | Step 00120 | Loss 0.5346 | Train Acc 0.8560 | Speed (samples/sec) 5439.0587 | GPU 0.0 MiB
Epoch 00016 | Step 00140 | Loss 0.5405 | Train Acc 0.8430 | Speed (samples/sec) 5443.7313 | GPU 0.0 MiB
Epoch 00016 | Step 00160 | Loss 0.5225 | Train Acc 0.8620 | Speed (samples/sec) 5448.5190 | GPU 0.0 MiB
Epoch 00016 | Step 00180 | Loss 0.4823 | Train Acc 0.8690 | Speed (samples/sec) 5453.0525 | GPU 0.0 MiB
Epoch Time(s): 34.5671
Epoch 00017 | Step 00000 | Loss 0.5830 | Train Acc 0.8530 | Speed (samples/sec) 5456.7534 | GPU 0.0 MiB
Epoch 00017 | Step 00020 | Loss 0.5701 | Train Acc 0.8350 | Speed (samples/sec) 5459.7646 | GPU 0.0 MiB
Epoch 00017 | Step 00040 | Loss 0.5418 | Train Acc 0.8650 | Speed (samples/sec) 5462.9347 | GPU 0.0 MiB
Epoch 00017 | Step 00060 | Loss 0.5834 | Train Acc 0.8440 | Speed (samples/sec) 5466.4072 | GPU 0.0 MiB
Epoch 00017 | Step 00080 | Loss 0.5798 | Train Acc 0.8500 | Speed (samples/sec) 5462.8542 | GPU 0.0 MiB
Epoch 00017 | Step 00100 | Loss 0.5500 | Train Acc 0.8380 | Speed (samples/sec) 5465.6063 | GPU 0.0 MiB
Epoch 00017 | Step 00120 | Loss 0.4719 | Train Acc 0.8690 | Speed (samples/sec) 5468.7413 | GPU 0.0 MiB
Epoch 00017 | Step 00140 | Loss 0.6273 | Train Acc 0.8360 | Speed (samples/sec) 5472.2271 | GPU 0.0 MiB
Epoch 00017 | Step 00160 | Loss 0.5432 | Train Acc 0.8650 | Speed (samples/sec) 5475.5296 | GPU 0.0 MiB
Epoch 00017 | Step 00180 | Loss 0.5074 | Train Acc 0.8690 | Speed (samples/sec) 5479.0416 | GPU 0.0 MiB
Epoch Time(s): 36.9982
Epoch 00018 | Step 00000 | Loss 0.6121 | Train Acc 0.8290 | Speed (samples/sec) 5481.9582 | GPU 0.0 MiB
Epoch 00018 | Step 00020 | Loss 0.5665 | Train Acc 0.8410 | Speed (samples/sec) 5485.1627 | GPU 0.0 MiB
Epoch 00018 | Step 00040 | Loss 0.5741 | Train Acc 0.8400 | Speed (samples/sec) 5488.2503 | GPU 0.0 MiB
Epoch 00018 | Step 00060 | Loss 0.5845 | Train Acc 0.8360 | Speed (samples/sec) 5491.5478 | GPU 0.0 MiB
Epoch 00018 | Step 00080 | Loss 0.5587 | Train Acc 0.8390 | Speed (samples/sec) 5495.1764 | GPU 0.0 MiB
Epoch 00018 | Step 00100 | Loss 0.5382 | Train Acc 0.8520 | Speed (samples/sec) 5498.6310 | GPU 0.0 MiB
Epoch 00018 | Step 00120 | Loss 0.6332 | Train Acc 0.8470 | Speed (samples/sec) 5501.7536 | GPU 0.0 MiB
Epoch 00018 | Step 00140 | Loss 0.5644 | Train Acc 0.8540 | Speed (samples/sec) 5500.1403 | GPU 0.0 MiB
Epoch 00018 | Step 00160 | Loss 0.5208 | Train Acc 0.8510 | Speed (samples/sec) 5502.7436 | GPU 0.0 MiB
Epoch 00018 | Step 00180 | Loss 0.5507 | Train Acc 0.8450 | Speed (samples/sec) 5505.9487 | GPU 0.0 MiB
Epoch Time(s): 36.2852
Epoch 00019 | Step 00000 | Loss 0.5429 | Train Acc 0.8610 | Speed (samples/sec) 5509.1086 | GPU 0.0 MiB
Epoch 00019 | Step 00020 | Loss 0.5431 | Train Acc 0.8430 | Speed (samples/sec) 5512.1172 | GPU 0.0 MiB
Epoch 00019 | Step 00040 | Loss 0.6500 | Train Acc 0.8320 | Speed (samples/sec) 5515.2831 | GPU 0.0 MiB
Epoch 00019 | Step 00060 | Loss 0.5925 | Train Acc 0.8380 | Speed (samples/sec) 5518.4398 | GPU 0.0 MiB
Epoch 00019 | Step 00080 | Loss 0.5713 | Train Acc 0.8400 | Speed (samples/sec) 5521.7259 | GPU 0.0 MiB
Epoch 00019 | Step 00100 | Loss 0.5082 | Train Acc 0.8520 | Speed (samples/sec) 5525.1049 | GPU 0.0 MiB
Epoch 00019 | Step 00120 | Loss 0.5272 | Train Acc 0.8470 | Speed (samples/sec) 5528.3756 | GPU 0.0 MiB
Epoch 00019 | Step 00140 | Loss 0.5445 | Train Acc 0.8550 | Speed (samples/sec) 5531.6688 | GPU 0.0 MiB
Epoch 00019 | Step 00160 | Loss 0.5385 | Train Acc 0.8620 | Speed (samples/sec) 5534.5603 | GPU 0.0 MiB
Epoch 00019 | Step 00180 | Loss 0.5093 | Train Acc 0.8650 | Speed (samples/sec) 5538.3537 | GPU 0.0 MiB
Epoch Time(s): 34.5503
100%|███████████████████████████████████████████████████████████████████████████████████|  2450/2450 [02:58<00:00, 13.72it/s]
100%|███████████████████████████████████████████████████████████████████████████████████|  2450/2450 [02:07<00:00, 19.16it/s]
Test Acc 0.6980
100%|███████████████████████████████████████████████████████████████████████████████████| 2450/2450 [02:59<00:00, 13.67it/s]
100%|███████████████████████████████████████████████████████████████████████████████████|  2450/2450 [02:30<00:00, 16.25it/s]
Full Acc 0.7161
Avg epoch time: 42.28637817700704
```

### 数据结论

由于数据分割是按照以下规定来的：

Specifically, we sort the products according to their sales ranking and use the top 10% for training, next top 2% for validation, and the rest for testing.

可以看出，头部产品和尾部产品还是有较大差别，但是整体效果尚可。