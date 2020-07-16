# DGL实验报告-2



## 实验环境

设备：Macbook Pro 13‘ （2018）

数据集：ogbl-collab

主要依赖：pytorch + DGL

模型：SAGE

代码仓库：https://github.com/aSeriousCoder/GNN-DGL



### 数据集：Dataset `ogbl-collab` ([Leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-collab)):

**Graph:** The `ogbl-collab` dataset is an undirected graph, representing a subset of the collaboration network between authors indexed by MAG. Each node represents an author and edges indicate the collaboration between authors. All nodes come with 128-dimensional features, obtained by averaging the word embeddings of papers that are published by the authors. All edges are associated with two meta-information: the year and the edge weight, representing the number of co-authored papers published in that year. The graph can be viewed as a dynamic multi-graph since there can be multiple edges between two nodes if they collaborate in more than one year.

**Prediction task:** The task is to predict the future author collaboration relationships given the past collaborations. The goal is to rank true collaborations higher than false collaborations. Specifically, we rank each true collaboration among a set of 100,000 randomly-sampled negative collaborations, and count the ratio of positive edges that are ranked at K-place or above (Hits@K). We found K = 50 to be a good threshold in our preliminary experiments.

**Dataset splitting:** We split the data according to time, in order to simulate a realistic application in collaboration recommendation. Specifically, we use the collaborations until 2017 as training edges, those in 2018 as validation edges, and those in 2019 as test edges.

#### References

[1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia.Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.

##### License: ODC-BY



## 实验内容

1. 使用DGL在Benchmark（OGB）数据集上进行Link Prediction实验



## 模型设计

参考OGB leaderboard给出的样例代码，采用GraphSAGE进行节点表征训练，同时结合一个线性的神经网络对两个节点之间的关联性进行打分（加入随机取样的负样本），使用Adam优化器对这两部分的参数同时进行训练，以求在训练集上，实际“正样本+负样本”的总体loss最小，而后在验证集和测试集上评估。



## 实验结论

还是**可以用**的，但是OGB和DGL的对接有一些麻烦（三个集合的分割使用反面，不过我觉得这个无法避免）。后面可以考虑出一个DGL+OGB图神经网络编程指导文档。



## 运行结果

```bash
Using backend: pytorch
Namespace(batch_size=1000, dropout=0.0, epochs=200, eval_steps=1, fan_out='10,10,10,10,10', hidden_channels=128, lr=0.001, num_layers=3, num_workers=1)
----loading dataset----
>>> dataset loaded, name: ogbl-collab, task: link prediction
----Building Models----
----Training----
236it [05:05,  1.29s/it]
Hits@10
Epoch: 01, Loss: 0.2169, Train: 4.87%, Valid: 1.53%, Test: 0.85%Epoch Time: 305.38556599617004
Hits@50
Epoch: 01, Loss: 0.2169, Train: 11.17%, Valid: 3.87%, Test: 2.54%Epoch Time: 305.38556599617004
Hits@100
Epoch: 01, Loss: 0.2169, Train: 16.35%, Valid: 6.10%, Test: 4.36%Epoch Time: 305.38556599617004
------
236it [04:57,  1.26s/it]
Hits@10
Epoch: 02, Loss: 0.1558, Train: 7.92%, Valid: 2.42%, Test: 1.49%Epoch Time: 297.9256718158722
Hits@50
Epoch: 02, Loss: 0.1558, Train: 16.84%, Valid: 6.00%, Test: 4.16%Epoch Time: 297.9256718158722
Hits@100
Epoch: 02, Loss: 0.1558, Train: 25.86%, Valid: 10.37%, Test: 7.69%Epoch Time: 297.9256718158722
```

