"""
실습 2에서는 PyTorch Geometric(PyG)을 사용해 자체 그래프 신경망을 구축한 다음,
이 모델을 두 개의 오픈 그래프 벤치마크(OGB) 데이터 세트에 적용합니다.
이 두 데이터 세트는 두 가지 그래프 기반 작업에서 모델의 성능을 벤치마킹하는 데 사용됩니다:
1) 단일 노드의 속성을 예측하는 노드 속성 예측과
2) 전체 그래프 또는 하위 그래프의 속성을 예측하는 그래프 속성 예측입니다.
먼저 파이토치 지오메트릭이 그래프를 파이토치 텐서로 저장하는 방법을 배웁니다.
그런 다음 ogb 패키지를 사용하여 오픈 그래프 벤치마크(OGB) 데이터 세트 중 하나를 로드하고 검사합니다.
OGB는 그래프에 대한 머신 러닝을 위한 사실적이고 대규모의 다양한 벤치마크 데이터 세트 모음입니다.
ogb 패키지는 각 데이터 세트에 대한 데이터 로더뿐만 아니라 모델 평가기도 제공합니다.
마지막으로 파이토치 지오메트릭을 사용해 자체 그래프 신경망을 구축합니다.
그런 다음 OGB 노드 속성 예측 및 그래프 속성 예측 작업에 대해 모델을 훈련하고 평가합니다.
참고: 중간 변수/패키지가 다음 셀로 이월되도록 각 섹션의 모든 셀을 순차적으로 실행해야 합니다.
진행 상황을 잃지 않도록 이 콜랩의 사본을 드라이브에 저장하는 것이 좋습니다!
Colab 2에서 재미와 행운을 빕니다 :)
"""

"""
Device

이 실습을 빠르게 실행하려면 GPU를 사용해야 할 수도 있습니다.
런타임을 클릭한 다음 런타임 유형 변경을 클릭하세요.
그런 다음 하드웨어 가속기를 GPU로 설정합니다.
"""

"""
Setup

Colab 0에서 설명한 것처럼 Colab에 PyG를 설치하는 것은 약간 까다로울 수 있습니다.
먼저 어떤 버전의 PyTorch를 실행 중인지 확인하겠습니다.
"""
import torch
import os
print("PyTorch has version {}".format(torch.__version__))

"""
PyG에 필요한 패키지를 다운로드합니다.
사용 중인 토치 버전이 위 셀의 출력과 일치하는지 확인하세요.
문제가 있는 경우 PyG 설치 페이지에서 자세한 정보를 확인할 수 있습니다.
"""

# Install torch geometric
# if 'IS_GRADESCOPE_ENV' not in os.environ:
#   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
#   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cu116.html
#   pip install torch-geometric
#   pip install ogb

"""
1) PyTorch Geometric (Datasets and Data)

PyTorch Geometric에는 그래프를 텐서 형식으로 저장 및/또는 변환하기 위한 두 개의 클래스가 있습니다.
하나는 다양한 일반적인 그래프 데이터 집합을 포함하는 torch_geometric.datasets입니다.
또 다른 하나는 torch_geometric.data로, 파이토치 텐서에서 그래프의 데이터 처리를 제공합니다.
이 섹션에서는 torch_geometric.datasets와 torch_geometric.data를 함께 사용하는 방법을 배우겠습니다.
"""

"""
PyG 데이터 세트

torch_geometric.datasets 클래스에는 많은 일반적인 그래프 데이터 세트가 있습니다.
여기서는 하나의 예제 데이터 세트를 통해 그 사용법을 살펴보겠습니다.
"""
from torch_geometric.datasets import TUDataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
  root = './enzymes'
  name = 'ENZYMES'

  # The ENZYMES dataset
  pyg_dataset= TUDataset(root, name)

  # You will find that there are 600 graphs in this dataset
  print(pyg_dataset)
  
"""
문제 1: ENZYMES 데이터 집합의 클래스 수와 기능 수는 얼마입니까? (5점)
"""
def get_num_classes(pyg_dataset):
  # TODO: Implement a function that takes a PyG dataset object
  # and returns the number of classes for that dataset.

  num_classes = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.

  #########################################

  return num_classes

def get_num_features(pyg_dataset):
  # TODO: Implement a function that takes a PyG dataset object
  # and returns the number of features for that dataset.

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.

  #########################################

  return num_features

if 'IS_GRADESCOPE_ENV' not in os.environ:
  num_classes = get_num_classes(pyg_dataset)
  num_features = get_num_features(pyg_dataset)
  print("{} dataset has {} classes".format(name, num_classes))
  print("{} dataset has {} features".format(name, num_features))
  
"""
PyG 데이터

각 PyG 데이터 세트는 torch_geometric.data.Data 객체 목록을 저장하며,
여기서 각 torch_geometric.data.Data 객체는 그래프를 나타냅니다.
데이터 세트에 인덱싱하여 Data 객체를 쉽게 얻을 수 있습니다.
Data 객체에 저장되는 내용 등 자세한 내용은 설명서를 참조하세요.
"""

"""
문제 2: ENZYMES 데이터 집합에서 인덱스가 100인 그래프의 레이블은 무엇입니까? (5점)
"""
def get_graph_class(pyg_dataset, idx):
  # TODO: Implement a function that takes a PyG dataset object,
  # an index of a graph within the dataset, and returns the class/label 
  # of the graph (as an integer).

  label = -1

  ############# Your code here ############
  ## (~1 line of code)

  #########################################

  return label

# Here pyg_dataset is a dataset for graph classification
if 'IS_GRADESCOPE_ENV' not in os.environ:
  graph_0 = pyg_dataset[0]
  print(graph_0)
  idx = 100
  label = get_graph_class(pyg_dataset, idx)
  print('Graph with index {} has label {}'.format(idx, label))
  
"""
문제 3: 인덱스 200의 그래프에는 몇 개의 가장자리가 있습니까? (5점)
"""
def get_graph_num_edges(pyg_dataset, idx):
  # TODO: Implement a function that takes a PyG dataset object,
  # the index of a graph in the dataset, and returns the number of 
  # edges in the graph (as an integer). You should not count an edge 
  # twice if the graph is undirected. For example, in an undirected 
  # graph G, if two nodes v and u are connected by an edge, this edge
  # should only be counted once.

  num_edges = 0

  ############# Your code here ############
  ## Note:
  ## 1. You can't return the data.num_edges directly
  ## 2. We assume the graph is undirected
  ## 3. Look at the PyG dataset built in functions
  ## (~4 lines of code)

  #########################################

  return num_edges

if 'IS_GRADESCOPE_ENV' not in os.environ:
  idx = 200
  num_edges = get_graph_num_edges(pyg_dataset, idx)
  print('Graph with index {} has {} edges'.format(idx, num_edges))
  
"""
2) Open Graph Benchmark (OGB)

오픈 그래프 벤치마크(OGB)는 그래프에 대한 머신 러닝을 위한 사실적이고 대규모의 다양한 벤치마크 데이터 세트 모음입니다.
이 데이터 세트는 OGB 데이터 로더를 사용하여 자동으로 다운로드, 처리 및 분할됩니다.
그런 다음 OGB 평가기를 사용하여 통합된 방식으로 모델 성능을 평가할 수 있습니다.
"""

"""
Dataset and Data

OGB는 PyG 데이터 세트와 데이터 클래스도 지원합니다.
여기서는 ogbn-arxiv 데이터셋을 살펴보겠습니다.
"""
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
  dataset_name = 'ogbn-arxiv'
  # Load the dataset and transform it to sparse tensor
  dataset = PygNodePropPredDataset(name=dataset_name,
                                  transform=T.ToSparseTensor())
  print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

  # Extract the graph
  data = dataset[0]
  print(data)

"""
질문 4: ogbn-아카이브 그래프에는 몇 개의 기능이 있나요? (5점)
"""
def graph_num_features(data):
  # TODO: Implement a function that takes a PyG data object,
  # and returns the number of features in the graph (as an integer).

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)

  #########################################

  return num_features

if 'IS_GRADESCOPE_ENV' not in os.environ:
  num_features = graph_num_features(data)
  print('The graph has {} features'.format(num_features))
  
"""
3) GNN: 노드 속성 예측

이 섹션에서는 파이토치 지오메트릭을 사용해 첫 번째 그래프 신경망을 구축하겠습니다.
그런 다음 노드 속성 예측(노드 분류) 작업에 적용하겠습니다.
특히, 그래프 신경망의 기초로 GCN을 사용할 것입니다(Kipf et al. (2017)).
이를 위해 PyG에 내장된 GCNConv 레이어를 사용할 것입니다.
"""

"""
Setup
"""
import torch
import pandas as pd
import torch.nn.functional as F
print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

"""
Load and Preprocess the Dataset
"""
if 'IS_GRADESCOPE_ENV' not in os.environ:
  dataset_name = 'ogbn-arxiv'
  dataset = PygNodePropPredDataset(name=dataset_name,
                                  transform=T.ToSparseTensor())
  data = dataset[0]

  # Make the adjacency matrix to symmetric
  data.adj_t = data.adj_t.to_symmetric()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # If you use GPU, the device should be cuda
  print('Device: {}'.format(device))

  data = data.to(device)
  split_idx = dataset.get_idx_split()
  train_idx = split_idx['train'].to(device)
  
"""
GCN Model

이제 GCN 모델을 구현해 보겠습니다!
아래 그림에 따라 포워드 함수를 구현하세요.
"""
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## For more information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)


        #########################################

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as shown in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)

        #########################################

        return out

def train(model, data, train_idx, optimizer, loss_fn):
    # TODO: Implement a function that trains the model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    ############# Your code here ############
    ## Note:
    ## 1. Zero grad the optimizer
    ## 2. Feed the data into the model
    ## 3. Slice the model output and label by train_idx
    ## 4. Feed the sliced output and label to loss_fn
    ## (~4 lines of code)

    #########################################

    loss.backward()
    optimizer.step()

    return loss.item()

# Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator, save_model_results=False):
    # TODO: Implement a function that tests the model by 
    # using the given split_idx and evaluator.
    model.eval()

    # The output of model on all data
    out = None

    ############# Your code here ############
    ## (~1 line of code)
    ## Note:
    ## 1. No index slicing here

    #########################################

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if save_model_results:
      print ("Saving Model Predictions")

      data = {}
      data['y_pred'] = y_pred.view(-1).cpu().detach().numpy()

      df = pd.DataFrame(data=data)
      # Save locally as csv
      df.to_csv('ogbn-arxiv_node.csv', sep=',', index=False)


    return train_acc, valid_acc, test_acc
  
# Please do not change the args
if 'IS_GRADESCOPE_ENV' not in os.environ:
  args = {
      'device': device,
      'num_layers': 3,
      'hidden_dim': 256,
      'dropout': 0.5,
      'lr': 0.01,
      'epochs': 100,
  }
  args
  
if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN(data.num_features, args['hidden_dim'],
              dataset.num_classes, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbn-arxiv')
  
# Please do not change these args
# Training should take <10min using GPU runtime
import copy
if 'IS_GRADESCOPE_ENV' not in os.environ:
  # reset the parameters to initial random value
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = F.nll_loss

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
    
"""
질문 5: 베스트_모델 검증 및 테스트 정확도는 얼마입니까?(20점)

아래 셀을 실행하여 베스트 오브 모델의 결과를 확인하고 모델의 예측을 ogbn-arxiv_node.csv라는 파일에 저장합니다.
왼쪽 패널의 폴더 아이콘을 클릭하면 이 파일을 볼 수 있습니다.
랩 1에서와 마찬가지로 과제를 요약할 때 이 파일을 다운로드하여 제출물에 첨부해야 합니다.
"""
if 'IS_GRADESCOPE_ENV' not in os.environ:
  best_result = test(best_model, data, split_idx, evaluator, save_model_results=True)
  train_acc, valid_acc, test_acc = best_result
  print(f'Best model: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
  
"""
4) GNN: Graph Property Prediction

이 섹션에서는 그래프 속성 예측(그래프 분류)을 위한 그래프 신경망을 만들어 보겠습니다.
"""

"""
Load and preprocess the dataset
"""
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm

if 'IS_GRADESCOPE_ENV' not in os.environ:
  # Load the dataset 
  dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device: {}'.format(device))

  split_idx = dataset.get_idx_split()

  # Check task type
  print('Task type: {}'.format(dataset.task_type))
  
# Load the dataset splits into corresponding dataloaders
# We will train the graph classification task on a batch of 32 graphs
# Shuffle the order of graphs for training set
if 'IS_GRADESCOPE_ENV' not in os.environ:
  train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
  valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
  test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)
  
if 'IS_GRADESCOPE_ENV' not in os.environ:
  # Please do not change the args
  args = {
      'device': device,
      'num_layers': 5,
      'hidden_dim': 256,
      'dropout': 0.5,
      'lr': 0.001,
      'epochs': 30,
  }
  args
  
"""
Graph Prediction Model
"""

"""
Graph Mini-Batching

실제 모델을 살펴보기 전에 그래프 미니 배칭의 개념을 소개합니다.
그래프의 미니 배치 처리를 병렬화하기 위해 PyG는 그래프를연결이 끊긴 단일 그래프 데이터 객체(torch_geometric.data.Batch)로 결합합니다.
torch_geometric.data.Batch는 앞서 소개한 torch_geometric.data.Data를 상속하며 batch라는 추가 어트리뷰트를 포함하고 있습니다.
배치 속성은 각 노드를 미니 배치 내의 해당 그래프의 인덱스에 매핑하는 벡터입니다:
배치 = [0, ..., 0, 1, ..., n - 2, n - 1, ..., n - 1]
이 속성은 각 노드가 속한 그래프를 연결하는 데 중요하며,
예를 들어 각 그래프의 노드 임베딩을 개별적으로 평균화하여 그래프 레벨 임베딩을 계산하는 데 사용할 수 있습니다.
"""

"""
Implemention

이제 GCN 그래프 예측 모델을 구현하기 위한 모든 도구가 준비되었습니다!
기존 GCN 모델을 재사용하여 node_embedding을 생성한 다음
노드에 대한 글로벌 풀링을 사용하여 각 그래프의 프로퍼티를 예측하는 데 사용할 수 있는 그래프 수준 임베딩을 생성하겠습니다.
미니 배치 그래프에 대해 글로벌 풀링을 수행하려면 배치 속성이 필수적이라는 점을 기억하세요.
"""
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        ############# Your code here ############
        ## Note:
        ## 1. Initialize self.pool as a global mean pooling layer
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers

        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # TODO: Implement a function that takes as input a 
        # mini-batch of graphs (torch_geometric.data.Batch) and 
        # returns the predicted graph property for each graph. 
        #
        # NOTE: Since we are predicting graph level properties,
        # your output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

    
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct node embeddings using existing GCN model
        ## 2. Use the global pooling layer to aggregate features for each individual graph
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 3. Use a linear layer to predict each graph's property
        ## (~3 lines of code)

        #########################################

        return out

def train(model, device, data_loader, optimizer, loss_fn):
    # TODO: Implement a function that trains your model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          pass
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        ############# Your code here ############
        ## Note:
        ## 1. Zero grad the optimizer
        ## 2. Feed the data into the model
        ## 3. Use `is_labeled` mask to filter output and labels
        ## 4. You may need to change the type of label to torch.float32
        ## 5. Feed the output and label to the loss_fn
        ## (~3 lines of code)

        #########################################

        loss.backward()
        optimizer.step()

    return loss.item()
  
# The evaluation function
def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print ("Saving Model Predictions")
        
        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        # Save to csv
        df.to_csv('ogbg-molhiv_graph_' + save_file + '.csv', sep=',', index=False)

    return evaluator.eval(input_dict)
  
if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN_Graph(args['hidden_dim'],
              dataset.num_tasks, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbg-molhiv')
  
# Please do not change these args
# Training should take <10min using GPU runtime
import copy

if 'IS_GRADESCOPE_ENV' not in os.environ:
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = torch.nn.BCEWithLogitsLoss()

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    val_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)

    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
    
"""
질문 6: 베스트_모델 검증 및 테스트 ROC-AUC 점수는 얼마입니까? (20점)

아래 셀을 실행하여 베스트 오브 모델의 결과를 확인하고 검증 및 테스트 데이터 세트에 대한 모델의 예측을 저장합니다.
결과 파일의 이름은 ogbn-arxiv_graph_valid.csv 및 ogbn-arxiv_graph_test.csv입니다.
왼쪽 패널의 폴더 아이콘을 클릭하면 이러한 파일을 볼 수 있습니다.
랩 1에서와 마찬가지로 과제를 요약할 때 이러한 파일을 다운로드하여 제출물에 첨부해야 합니다.
"""
if 'IS_GRADESCOPE_ENV' not in os.environ:
  train_acc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
  valid_acc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")[dataset.eval_metric]
  test_acc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")[dataset.eval_metric]

  print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')
    
"""
문제 7 (선택 사항): 파이토치 지오메트릭에서 다른 두 개의 글로벌 풀링 레이어를 실험해 보세요.
"""

"""
Submission

제출
랩 2를 제출하려면 성적 관리에서 다음 과제를 제출하십시오:

"실습 2": 이 과제의 질문에 대한 답을 제출합니다.
"실습 2 코드": 완성한 CS224W_Colab2.ipynb를 제출합니다.
"파일" 메뉴에서 ".ipynb 다운로드"를 선택하여 완료된 실습의 로컬 사본을 저장합니다.
이름을 변경하지 마세요! 자동 채점기는 "CS224W_Colab2.ipynb"라는 .ipynb 파일에 의존합니다.
"""
