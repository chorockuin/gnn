"""
랩 0은 채점되지 않으므로 이 노트북을 제출할 필요가 없습니다.
하지만 그래프 마이닝과 그래프 신경망의 기본 개념에 익숙해질 수 있도록 이 노트북을 실행하는 것을 적극 권장합니다.
이번 콜랩에서는 두 가지 패키지, 즉 NetworkX와 PyTorch Geometric을 소개합니다.
PyTorch Geometric 섹션의 경우 모든 세부 사항을 이미 이해할 필요는 없습니다.
그래프 신경망의 개념과 구현은 향후 강의와 콜랩에서 다룰 예정입니다.
계속 진행하기 전에 복사본을 만들어 두시기 바랍니다.
"""

"""
NetworkX 튜토리얼

NetworkX는 그래프를 생성, 조작, 마이닝하는 데 가장 자주 사용되는 Python 패키지 중 하나입니다.
이 튜토리얼의 주요 부분은 아래 링크에서 가져온 것입니다.
https://colab.research.google.com/github/jdwittenauer/ipython-notebooks/blob/master/notebooks/libraries/NetworkX.ipynb#scrollTo=zA1OO6huHeV6 
"""

"""
Setup
"""
# Upgrade packages
# pip install --upgrade scipy networkx

import networkx as nx

"""
Graph

NetworkX는 방향성 및 비방향성 그래프와 같은 다양한 유형의 그래프를 저장할 수 있는 여러 클래스를 제공합니다.
또한 다중 그래프(방향성 및 비방향성 모두)를 생성하는 클래스도 제공합니다.
자세한 내용은 NetworkX 그래프 유형을 참조하세요.
"""
# Create an undirected graph G
G = nx.Graph()
print(G.is_directed())

# Create a directed graph H
H = nx.DiGraph()
print(H.is_directed())

# Add graph level attribute
G.graph["Name"] = "Bar"
print(G.graph)

"""
Node

노드(속성 포함)를 NetworkX 그래프에 쉽게 추가할 수 있습니다.
"""
# Add one node with node level attributes
G.add_node(0, feature=5, label=0)

# Get attributes of the node 0
node_0_attr = G.nodes[0]
print("Node 0 has the attributes {}".format(node_0_attr))

print(G.nodes(data=True))

# Add multiple nodes with attributes
G.add_nodes_from([
  (1, {"feature": 1, "label": 1}),
  (2, {"feature": 2, "label": 2})
]) #(node, attrdict)

# Loop through all the nodes
# Set data=True will return node attributes
for node in G.nodes(data=True):
  print(node)

# Get number of nodes
num_nodes = G.number_of_nodes()
print("G has {} nodes".format(num_nodes))

"""
Edge

노드와 마찬가지로 에지(속성 포함)도 NetworkX 그래프에 쉽게 추가할 수 있습니다.
"""
# Add one edge with edge weight 0.5
G.add_edge(0, 1, weight=0.5)

# Get attributes of the edge (0, 1)
edge_0_1_attr = G.edges[(0, 1)]
print("Edge (0, 1) has the attributes {}".format(edge_0_1_attr))

# Add multiple edges with edge weights
G.add_edges_from([
  (1, 2, {"weight": 0.3}),
  (2, 0, {"weight": 0.1})
])

# Loop through all the edges
# Here there is no data=True, so only the edge will be returned
for edge in G.edges():
  print(edge)

# Get number of edges
num_edges = G.number_of_edges()
print("G has {} edges".format(num_edges))

"""
Visualization
"""
import matplotlib.pyplot as plt

# Draw the graph
nx.draw(G, with_labels = True)
plt.show()

"""
Node Degree and Neighbor
"""
node_id = 1

# Degree of node 1
print("Node {} has degree {}".format(node_id, G.degree[node_id]))

# Get neighbor of node 1
for neighbor in G.neighbors(node_id):
  print("Node {} has neighbor {}".format(node_id, neighbor))
  
"""
Other Functionalities

네트워크엑스는 그래프를 연구하는 데 유용한 방법도 많이 제공합니다.
다음은 노드의 페이지랭크를 구하는 예제입니다(자세한 내용은 페이지랭크에 대한 작년 슬라이드를 참조하세요!).
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank
http://snap.stanford.edu/class/cs224w-2020/slides/04-pagerank.pdf
"""
import matplotlib.pyplot as plt

# Returns linearly connected graph
nx.path_graph(num_nodes)

num_nodes = 4
# Create a new path like graph and change it to a directed graph
G = nx.DiGraph(nx.path_graph(num_nodes))
nx.draw(G, with_labels = True)
plt.show()

# Get the PageRank
pr = nx.pagerank(G, alpha=0.8)
print(pr)

"""
Documentation

설명서를 통해 더 많은 NetworkX 기능을 살펴볼 수 있습니다.
https://networkx.org/documentation/stable/
"""

"""
PyTorch Geometric Tutorial

PyTorch Geometric(PyG)은 PyTorch의 확장 라이브러리입니다.
다양한 그래프 신경망 레이어와 수많은 벤치마크 데이터 세트를 포함해 그래프 딥 러닝 모델을 개발하는 데 유용한 기본 요소를 제공합니다.
GCNConv와 같은 일부 개념을 이해하지 못하더라도 걱정하지 마세요. 다음 강의에서 모두 다룰 것입니다 :)
이 튜토리얼은 Matthias Fey의 https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=ci-LpZWhRJoI 을 각색한 것입니다.
"""

"""
Installing dependencies

Colab에 PyG를 설치하는 것은 약간 까다로울 수 있습니다.
아래 셀을 실행해 보세요. 문제가 있는 경우 PyG의 설치 페이지에서 자세한 정보를 확인할 수 있습니다.
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
참고: 이 셀을 실행하는 데 시간이 걸릴 수 있습니다(최대 10분).
"""
# Install torch geometric
# pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
# pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
# pip install -q torch-geometric

############################################# PyTorch Geometric Tutorial
import torch
print("PyTorch has version {}".format(torch.__version__))

"""
Visualization
"""
# Helper function for visualization.
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()
    
"""
Introduction

최근 그래프에 대한 딥러닝은 딥러닝 커뮤니티에서 가장 뜨거운 연구 분야 중 하나로 떠오르고 있습니다.
여기서 그래프 신경망(GNN)은 고전적인 딥 러닝 개념을 이미지나 텍스트와 달리
불규칙한 구조의 데이터에 일반화하고 신경망이 객체와 그 관계에 대해 추론할 수 있도록 하는 것을 목표로 합니다.
이 튜토리얼에서는 PyTorch Geometric(PyG) 라이브러리(https://github.com/pyg-team/pytorch_geometric)를 기반으로 하는 그래프 신경망을 통해
그래프에 대한 딥 러닝에 관한 몇 가지 기본 개념을 소개합니다.
PyTorch Geometric은 널리 사용되는 딥 러닝 프레임워크 PyTorch의 확장 라이브러리로,
그래프 신경망을 쉽게 구현할 수 있는 다양한 메서드와 유틸리티로 구성되어 있습니다.
Kipf 외(2017)에 이어, 간단한 그래프 구조의 예시인 잘 알려진 재커리의 가라테 클럽 네트워크(https://en.wikipedia.org/wiki/Zachary%27s_karate_club)를 살펴보면서 GNN의 세계에 대해 알아봅시다.
이 그래프는 가라테 클럽 회원 34명으로 구성된 소셜 네트워크를 설명하며 클럽 외부에서 상호작용한 회원 간의 링크를 문서화합니다.
여기서 우리는 회원들의 상호 작용에서 발생하는 커뮤니티를 감지하는 데 관심이 있습니다.
"""

"""
Dataset

PyTorch Geometric은 torch_geometric.datosets 서브 패키지를 통해 데이터 세트에 쉽게 액세스할 수 있습니다:
"""

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

"""
KarateClub 데이터 집합을 초기화한 후, 먼저 몇 가지 속성을 살펴볼 수 있습니다.
예를 들어, 이 데이터 세트에는 정확히 하나의 그래프가 있으며,
이 데이터 세트의 각 노드에는 가라테 클럽의 회원을 고유하게 설명하는 34차원 특징 벡터가 할당되어 있음을 알 수 있습니다.
또한 그래프에는 정확히 4개의 클래스가 있으며, 이는 각 노드가 속한 커뮤니티를 나타냅니다.
이제 기본 그래프를 좀 더 자세히 살펴보겠습니다:
"""
data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {(2*data.num_edges) / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print(data.edge_index.T)

"""
Data

PyTorch Geometric의 각 그래프는 그래프 표현을 설명하는 모든 정보를 담고 있는 단일 데이터 객체로 표현됩니다.
print(data)를 통해 언제든지 데이터 객체를 인쇄하여 속성과 그 모양에 대한 간략한 요약을 얻을 수 있습니다:
Data(edge_index=[2, 156], x=[34, 34], y=[34], train_mask=[34])
이 데이터 객체는 4개의 속성을 가지고 있음을 알 수 있습니다.
(1) edge_index 속성은 그래프 연결에 대한 정보, 즉 각 에지에 대한 소스 및 대상 노드 인덱스의 튜플을 보유합니다.
PyG는 
(2) 노드 특징을 x(34개 노드 각각에 34딤 특징 벡터가 할당됨)로,
(3) 노드 레이블을 y(각 노드는 정확히 하나의 클래스에 할당됨)로 참조합니다.
(4) 또한 어떤 노드에 대한 커뮤니티 할당을 이미 알고 있는지를 설명하는 train_mask라는 추가 속성이 존재합니다.
총 4개의 노드(각 커뮤니티에 하나씩)에 대한 실사 기반 레이블만 알고 있으며, 나머지 노드에 대한 커뮤니티 할당을 추론하는 것이 과제입니다.
또한 데이터 객체는 기본 그래프의 몇 가지 기본 속성을 유추할 수 있는 몇 가지 유틸리티 함수를 제공합니다.
예를 들어, 그래프에 고립된 노드가 있는지(즉, 어떤 노드에 대한 에지가 없는지),
그래프에 자기 루프가 있는지(즉, (v,v)∈E) 또는 
그래프가 무방향성인지(즉, 각 에지 (v,w)∈E에 대해 에지 (w,v)∈E도 존재하는지) 쉽게 추론할 수 있습니다.
"""

print(data)

"""
Edge Index

다음으로 그래프의 edge_index를 인쇄하겠습니다:
"""

edge_index = data.edge_index
print(edge_index.t())

"""
edge_index를 출력하면 PyG가 그래프 연결성을 내부적으로 어떻게 표현하는지 더 이해할 수 있습니다.
각 에지에 대해 edge_index는 두 개의 노드 인덱스 튜플을 보유하며,
첫 번째 값은 소스 노드의 노드 인덱스를, 두 번째 값은 에지의 대상 노드의 노드 인덱스를 설명하는 것을 볼 수 있습니다.
이 표현은 스파스 행렬을 표현하는 데 일반적으로 사용되는 COO 형식(좌표 형식)으로 알려져 있습니다.
인접성 정보를 밀도 표현 A∈{0,1}|V|×|V|로 보유하는 대신 PyG는 그래프를 희소하게 표현하는데,
이는 A의 항목이 0이 아닌 좌표/값만 보유하는 것을 의미합니다.
그래프 조작 기능 외에도 시각화를 위한 강력한 도구를 구현하는 networkx 라이브러리 형식으로 변환하여 그래프를 더욱 시각화할 수 있습니다:
"""

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)

"""
Implementing Graph Neural Networks (GNNs)

PyG의 데이터 처리에 대해 배웠다면, 이제 첫 번째 그래프 신경망을 구현할 차례입니다!
이를 위해 가장 간단한 GNN 연산자 중 하나인 GCN 계층을 사용할 것입니다(Kipf et al. (2017)).
PyG는 노드 특징 표현 x와 COO 그래프 연결 표현 edge_index를 전달하여 실행할 수 있는 GCNConv를 통해 이 계층을 구현합니다.
GNN의 출력은 무엇인가요?
GNN의 목표는 각 노드 vi∈V가 입력 특징 벡터 X(0)i 를 갖는 입력 그래프 G=(V,E)를 취하는 것입니다.
우리가 학습하고자 하는 함수는 노드와 그 특징 벡터, 그래프 구조를 입력으로 받아 
다운스트림 작업에 유용한 방식으로 해당 노드를 나타내는 벡터인 임베딩을 출력하는 함수 f→V×Rd 입니다.
노드와 노드의 초기 특징을 학습된 임베딩에 매핑한 후에는 이러한 임베딩을 사용하여
노드 수준, 에지 수준 또는 그래프 수준의 회귀/분류 등 다양한 작업을 수행할 수 있습니다.
이 콜랩에서는 각 노드를 커뮤니티로 분류하는 데 유용한 임베딩을 학습하고자 합니다.
이제 torch.nn.Module 클래스에서 네트워크 아키텍처를 정의하여 첫 번째 그래프 신경망을 만들 준비가 되었습니다:
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)

"""
여기서는 먼저 __init__에서 모든 빌딩 블록을 초기화하고 네트워크의 계산 흐름을 앞으로 정의합니다.
먼저 세 개의 그래프 컨볼루션 레이어를 정의하고 스택합니다.
각 레이어는 각 노드의 1홉 이웃(직접 이웃)의 정보를 집계하는 데 해당하지만,
레이어를 함께 구성하면 각 노드의 3홉 이웃(최대 3 "홉" 떨어진 모든 노드)의 정보를 집계할 수 있습니다.
또한 GCNConv 레이어는 노드 특징 차원을 2, 즉 34→4→4→2로 줄입니다. 각 GCNConv 레이어는 탄 비선형성으로 향상됩니다.
그 후, 노드를 4개의 클래스/커뮤니티 중 1개에 매핑하는 분류기 역할을 하는 단일 선형 변환(torch.nn.Linear)을 적용합니다.
최종 분류기의 출력과 GNN이 생성한 최종 노드 임베딩을 모두 반환합니다.
GCN()을 통해 최종 모델을 초기화하고 모델을 인쇄하면 사용된 모든 하위 모듈의 요약이 생성됩니다.
"""

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize(h, color=data.y)

"""
놀랍게도, 모델의 가중치를 훈련하기 전에도 이 모델은 그래프의 커뮤니티 구조와 매우 유사한 노드 임베딩을 생성합니다.
모델의 가중치가 완전히 무작위로 초기화되어 있고 아직 학습을 수행하지 않았음에도 불구하고 임베딩 공간에서 같은 색(커뮤니티)의 노드가 이미 서로 밀접하게 클러스터링되어 있습니다!
이는 GNN이 강력한 귀납적 편향을 도입하여 입력 그래프에서 서로 가까운 노드에 대해 유사한 임베딩을 유도한다는 결론으로 이어집니다.
가라테 클럽 네트워크 훈련
하지만 더 나은 방법이 있을까요?
그래프에 있는 4개의 노드(각 커뮤니티에 하나씩)의 커뮤니티 할당에 대한 지식을 기반으로 네트워크 파라미터를 훈련하는 방법에 대한 예시를 살펴보겠습니다:
모델의 모든 것이 차별화되고 파라미터화되어 있으므로 몇 가지 레이블을 추가하고 모델을 훈련한 후 임베딩이 어떻게 반응하는지 관찰할 수 있습니다.
여기서는 준지도 또는 전이 학습 절차를 사용합니다:
클래스당 하나의 노드에 대해 간단히 학습하지만, 전체 입력 그래프 데이터를 사용할 수 있습니다.
우리 모델을 훈련하는 것은 다른 PyTorch 모델과 매우 유사합니다.
네트워크 아키텍처를 정의하는 것 외에도 손실 기준(여기서는 CrossEntropyLoss)을 정의하고 확률적 기울기 최적화기(여기서는 Adam)를 초기화합니다.
그 후 여러 차례의 최적화를 수행하며, 각 라운드는 포워드 패스와 백워드 패스로 구성되어 포워드 패스에서 도출된 손실에 대한 모델 파라미터의 기울기를 계산합니다.
PyTorch를 처음 사용하는 분이라면 이 방식이 익숙하게 느껴질 것입니다.
그렇지 않은 경우, PyTorch 문서에서 PyTorch에서 신경망을 훈련하는 방법에 대한 좋은 소개를 제공합니다.
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-loss-function-and-optimizer
반지도 학습 시나리오는 다음 줄에 의해 달성된다는 점에 유의하세요:
loss = criterion(out[data.train_mask], data.y[data.train_mask])
모든 노드에 대한 노드 임베딩을 계산하는 동안 손실 계산에는 훈련 노드만 사용합니다.
여기서는 분류기의 출력을 필터링하여 train_mask에 있는 노드만 포함하도록 기준값 레이블 data.y를 구현합니다.
이제 훈련을 시작하고 시간이 지남에 따라 노드 임베딩이 어떻게 변화하는지 살펴보겠습니다(코드를 명시적으로 실행하면 가장 잘 경험할 수 있습니다):
"""

import time

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    accuracy = {}
    # Calculate training accuracy on our four examples
    predicted_classes = torch.argmax(out[data.train_mask], axis=1) # [0.6, 0.2, 0.7, 0.1] -> 2
    target_classes = data.y[data.train_mask]
    accuracy['train'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())
    
    # Calculate validation accuracy on the whole graph
    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    return loss, h, accuracy

for epoch in range(500):
    loss, h, accuracy = train(data)
    # Visualize the node embeddings every 10 epochs
    if epoch % 10 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss, accuracy=accuracy)
        time.sleep(0.3)
        
"""
보시다시피, 3계층 GCN 모델은 커뮤니티를 매우 잘 분리하고 대부분의 노드를 올바르게 분류합니다.
또한, 데이터 처리와 GCN 구현에 도움을 준 PyTorch 지오메트릭 라이브러리 덕분에 몇 줄의 코드만으로 이 모든 작업을 수행할 수 있었습니다.
"""

"""
Documentation

문서를 통해 더 많은 PyG 함수를 살펴볼 수 있습니다.
https://pytorch-geometric.readthedocs.io/en/latest/
"""