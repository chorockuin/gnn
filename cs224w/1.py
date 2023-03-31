"""
이 실습에서는 노드 임베딩을 학습하기 위한 전체 파이프라인을 작성해 보겠습니다.
다음 3단계로 진행하겠습니다.
먼저 네트워크 과학의 고전적인 그래프인 가라테 클럽 네트워크를 로드합니다.
해당 그래프에 대한 여러 그래프 통계를 살펴봅니다.
그런 다음 그래프에 대한 머신 러닝을 수행할 수 있도록 그래프 구조를 PyTorch 텐서로 변환하는 작업을 함께 할 것입니다.
마지막으로 그래프에 대한 첫 번째 학습 알고리즘인 노드 임베딩 모델을 완성하겠습니다.
여기서는 강의에서 배운 DeepWalk/Node2vec 알고리즘보다 간단한 모델을 사용합니다.
하지만 파이토치를 통해 처음부터 작성할 것이기 때문에 여전히 보람 있고 도전적입니다.
이제 시작해보겠습니다!
참고: 중간 변수/패키지가 다음 셀로 넘어갈 수 있도록 모든 셀을 순차적으로 실행해야 합니다.
"""

"""
Graph Basics

먼저 네트워크 과학의 고전적인 그래프인 가라테 클럽 네트워크를 로드하겠습니다.
해당 그래프에 대한 여러 그래프 통계를 살펴보겠습니다.
"""

"""
Setup

이 랩에서는 NetworkX를 많이 사용할 것입니다.
"""
import networkx as nx
import matplotlib.pyplot as plt

"""
재커리의 가라테 클럽 네트워크
가라테 클럽 네트워크는 34명의 가라테 클럽 회원으로 구성된 소셜 네트워크를 설명하는 그래프로,
클럽 외부에서 교류한 회원 간의 연결 고리를 문서화합니다.
"""
G = nx.karate_club_graph()

# G is an undirected graph
print(type(G))

# Visualize the graph
nx.draw(G, with_labels = True)
plt.show()

"""
질문 1: 가라테 클럽 네트워크의 평균 degree는 어느 정도인가요? (5점)
"""
def average_degree(num_edges, num_nodes):
  # TODO: Implement this function that takes number of edges
  # and number of nodes, and returns the average node degree of 
  # the graph. Round the result to nearest integer (for example 
  # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

  avg_degree = 0

  ############# Your code here ############

  # 하나의 엣지는 2개의 노드가 공유하므로
  avg_degree = round((num_edges * 2) / num_nodes)  

  #########################################

  return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print(f"Average degree of karate club network is {avg_degree}")

"""
질문 2: 가라테 클럽 네트워크의 평균 클러스터링 계수는 얼마입니까? (5점)
"""
def average_clustering_coefficient(G):
  # TODO: Implement this function that takes a nx.Graph
  # and returns the average clustering coefficient. Round 
  # the result to 2 decimal places (for example 3.333 will
  # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

  avg_cluster_coef = 0

  ############# Your code here ############
  ## Note: 
  ## 1: Please use the appropriate NetworkX clustering function
  
  # 노드 v의 클러스터링 계수는
  # 노드 v의 이웃 노드들의 수를
  # 노드 v의 이웃 노드들 2개를 골라 Edge를 만들 수 있는 조합의 수로 나눈 값
  # nx 모듈을 사용해 간단히 구할 수 있다. 각 노드 별로 클러스터링 계수를 구해준다
  avg_cluster_coef = nx.average_clustering(G)
  
  #########################################

  return avg_cluster_coef
print(f"Average clustering coefficient of karate club network is {average_clustering_coefficient(G)}")

"""
문제 3: 페이지랭크 반복을 한 번 수행한 후 노드 0(id 0인 노드)의 페이지랭크 값은 얼마입니까? (5점)

페이지 순위는 웹의 링크 구조를 사용하여 그래프에서 노드의 중요도를 측정합니다.
중요한 페이지가 '투표'하면 더 가치가 있습니다.
구체적으로, 중요도가 r_i인 페이지 i에 아웃링크가 d_i 개인 경우 각 링크는 r_i/d_i '투표'값을 갖습니다.
따라서 r_j로 표시되는 페이지 j의 중요도는 페이지로 들어오는 인링크에 대한 '투표' 값의 합입니다.
r_j=∑i→j r_i/d_i
여기서 d_i는 노드 i의 아웃 degree 입니다.
즉, r_i의 아웃 degree 중, j로 가는 것들의 '투표' 값을 다 합친 것이 r_j의 중요도 입니다

예를 들어, 중요도가 10인 페이지 A에 아웃링크가 5개가 있는 경우, 각 아웃링크는 10/5=2 만큼의 투표 값을 갖습니다.
그 아웃링크 5개 중에 1개가 페이지 B를 향한다면, B의 중요도는 2가 됩니다

페이지랭크 알고리즘(Google에서 사용)은 링크를 클릭한 무작위 서퍼가 특정 페이지에 도달할 가능성을 나타내는 확률 분포를 출력합니다.
각 시간 단계에서 무작위 서퍼에게는 두 가지 옵션이 있습니다.
확률 β로 랜덤하게 링크를 따라갑니다.
확률 1-β로 링크를 따라가지 않고, 랜덤하게 페이지로 이동합니다.
따라서 특정 페이지의 중요도는 다음 PageRank 방정식을 사용하여 계산됩니다:
r_j=∑i→j β r_i/d_i + (1-β) 1/N 
링크를 따라간다면 중요도를 계산할 수 있고, 링크를 따라가지 않는다면 1/N(노드의 갯수) 만큼의 중요도만 갖는다는 뜻입니다 
노드 0에 대해 위의 PageRank 방정식을 구현하여 코드 블록을 완성하세요.
참고 - 자세한 내용은 다음 슬라이드에서 확인할 수 있습니다.
http://snap.stanford.edu/class/cs224w-2020/slides/04-pagerank.pdf
"""
def one_iter_pagerank(G, beta, r0, node_id):
  # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
  # The return value r1 is one interation PageRank value for the input node.
  # Please round r1 to 2 decimal places.

  r1 = 0
  
  ############# Your code here ############
  ## Note: 
  ## 1: You should not use nx.pagerank
  
  # 위 설명 참고
  r1 = beta * (r0/len(list(G.neighbors(node_id)))) + (1.0-beta) * (1/G.number_of_nodes())
  
  #########################################

  return r1

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))
print(nx.pagerank(G)[1])

"""
문제 4: 가라테 클럽 네트워크 노드 5의 (원시) 근접 중심성은 얼마입니까? (5점)

근접성 중심성에 대한 방정식은 c(v)=1/(∑u≠v u와 v 사이의 최단 경로 길이입니다)
"""
def closeness_centrality(G, node=5):
  # TODO: Implement the function that calculates closeness centrality 
  # for a node in karate club network. G is the input karate club 
  # network and node is the node id in the graph. Please round the 
  # closeness centrality result to 2 decimal places.

  ## Note:
  ## 1: You can use networkx closeness centrality function.
  ## 2: Notice that networkx closeness centrality returns the normalized 
  ## closeness directly, which is different from the raw (unnormalized) 
  ## one that we learned in the lecture.

  # 근접 중심성은
  # 일단 노드 v와 노드 v와 연결된 노드들 사이에 최단 경로의 길이를 다 더한다.
  # 그 값이 작을수록, 노드 v와 노드 v와 연결된 노드들은 가깝다는 뜻이다. 즉, 근접 중심성이 크다는 뜻
  # 따라서 그 값의 역수를 취해서 근접 중심성으로 사용한다
  closeness = nx.closeness_centrality(G, node)
  
  #########################################

  return closeness

node = 5
closeness = closeness_centrality(G, node=node)
print("The node 5 has closeness centrality {}".format(closeness))

"""
Graph to Tensor

그런 다음 그래프 G를 파이토치 텐서로 변환하여 그래프에 대해 머신 러닝을 수행할 수 있도록 함께 작업합니다.
"""

"""
Setup

PyTorch가 제대로 설치되었는지 확인
"""
import torch
# print(torch.__version__)

"""
PyTorch tensor basics

모든 0, 1 또는 임의의 값으로 PyTorch 텐서를 생성할 수 있습니다.
"""
# Generate 3 x 4 tensor with all ones
ones = torch.ones(3, 4)
# print(ones)

# Generate 3 x 4 tensor with all zeros
zeros = torch.zeros(3, 4)
# print(zeros)

# Generate 3 x 4 tensor with random values on the interval [0, 1)
random_tensor = torch.rand(3, 4)
# print(random_tensor)

# Get the shape of the tensor
# print(ones.shape)

"""
PyTorch 텐서에는 단일 데이터 유형인 dtype에 대한 요소가 포함되어 있습니다.
"""
# Create a 3 x 4 tensor with all 32-bit floating point zeros
zeros = torch.zeros(3, 4, dtype=torch.float32)
# print(zeros.dtype)

# Change the tensor dtype to 64-bit integer
zeros = zeros.type(torch.long)
# print(zeros.dtype)

"""
질문 5: 가라테 클럽 네트워크의 에지 목록을 가져와 torch.LongTensor로 변환합니다.
pos_edge_index 텐서의 torch.sum 값은 얼마입니까? (10점)
"""
def graph_to_edge_list(G):
  # TODO: Implement the function that returns the edge list of
  # an nx.Graph. The returned edge_list should be a list of tuples
  # where each tuple is a tuple representing an edge connected 
  # by two nodes.

  edge_list = []

  ############# Your code here ############
  
  edge_list = list(G.edges())
  
  #########################################

  return edge_list

def edge_list_to_tensor(edge_list):
  # TODO: Implement the function that transforms the edge_list to
  # tensor. The input edge_list is a list of tuples and the resulting
  # tensor should have the shape [2 x len(edge_list)].

  edge_index = torch.tensor(edge_list)

  ############# Your code here ############

  #########################################

  return edge_index

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

"""
질문 6: 음의 가장자리를 샘플링하는 다음 함수를 구현해 주세요.
그런 다음 가라데 클럽 네트워크에서 음의 에지(edge_1 ~ edge_5)는 어느 에지일까요? (10점)

"음의" 에지는 그래프에 존재하지 않는 에지/링크를 의미합니다.
"네거티브"라는 용어는 링크 예측의 "네거티브 샘플링"에서 차용한 것입니다.
에지 가중치와는 아무런 관련이 없습니다.
예를 들어 에지 (src, dst)가 주어졌을 때 그래프에서 (src, dst) 또는 (dst, src)가 에지가 아닌지 확인해야 합니다.
이것이 참이면 음의 에지입니다.
"""
import random

def sample_negative_edges(G, num_neg_samples):
  # TODO: Implement the function that returns a list of negative edges.
  # The number of sampled negative edges is num_neg_samples. You do not
  # need to consider the corner case when the number of possible negative edges
  # is less than num_neg_samples. It should be ok as long as your implementation 
  # works on the karate club network. In this implementation, self loops should 
  # not be considered as either a positive or negative edge. Also, notice that 
  # the karate club network is an undirected graph, if (0, 1) is a positive 
  # edge, do you think (1, 0) can be a negative one?
  
  neg_edge_list = []

  ############# Your code here ############

  # 그래프의 노드 수를 모두 구한 후
  node_num = G.number_of_nodes()
  # 그래프의 엣지 리스트도 구한다
  positive_edge_list = list(G.edges())
  
  # 노드의 id를 돌면서
  for i in range(node_num):
    for j in range(node_num):
      # 스스로 엣지 연결이 되는 것은 제외하라고 했으므로
      if i == j:
        continue
      # 그래프의 엣지 리스트에 있으면 네거티브 엣지가 아니므로 제외
      if (i, j) in positive_edge_list:
        continue
      if (j, i) in positive_edge_list:
        continue
      # 나머지들은 네거티브 엣지가 될 수 있다
      neg_edge_list.append((i, j))
      
  # num_neg_samples 갯수만큼만 쓰자
  neg_edge_list = neg_edge_list[:num_neg_samples]

  #########################################

  return neg_edge_list

# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

# Which of following edges can be negative ones?
edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)

############# Your code here ############
## Note:
## 1: For each of the 5 edges, print whether it can be negative edge
print(edge_1 in neg_edge_list)
print(edge_2 in neg_edge_list)
print(edge_3 in neg_edge_list)
print(edge_4 in neg_edge_list)
print(edge_5 in neg_edge_list)
"""
Node Emebedding Learning

마지막으로 그래프에 대한 첫 번째 학습 알고리즘인 노드 임베딩 모델을 완성하겠습니다.
"""

"""
Setup

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# print(torch.__version__)

"""
자체 노드 임베딩 학습 메서드를 작성하기 위해 파이토치에서 nn.Embedding 모듈을 많이 사용할 것입니다.
nn.Embedding을 사용하는 방법을 살펴봅시다:
"""
# Initialize an embedding layer
# Suppose we want to have embedding for 4 items (e.g., nodes)
# Each item is represented with 8 dimensional vector

# 1~4개의 입력을 8차원으로 임베딩한다
emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
print('Sample embedding layer: {}'.format(emb_sample))

"""
텐서 인덱스를 사용하여 임베딩 행렬에서 항목을 선택할 수 있습니다.
"""
# Select an embedding in emb_sample
id = torch.LongTensor([1])
# 8차원으로 임베딩한 행렬의 1번째 인덱스에 해당하는 리스트를 가져옴
print(emb_sample(id))

ids = torch.LongTensor([1, 0, 3, 2])
# 8차원으로 임베딩한 행렬의 1,0,3,2번째 인덱스에 해당하는 리스트를 가져옴
print(emb_sample(ids))

# Select multiple embeddings
ids = torch.LongTensor([1, 3])
# 8차원으로 임베딩한 행렬의 1,3번째 인덱스에 해당하는 리스트를 가져옴
print(emb_sample(ids))

# Get the shape of the embedding weight matrix
# 임베딩 행렬의 차원은 당연히 4 x 8이다
shape = emb_sample.weight.data.shape
print(shape)

# Overwrite the weight to tensor with all ones
# 임베딩 행렬의 요소를 모두 1로 바꾼다
emb_sample.weight.data = torch.ones(shape)

# Let's check if the emb is indeed initilized
ids = torch.LongTensor([0, 3])
print(emb_sample(ids))

"""
이제 그래프에 대한 노드 임베딩 행렬을 만들 차례입니다!

가라테 클럽 네트워크의 각 노드에 대해 16차원 벡터를 갖고 싶습니다.
이 행렬을 [0,1) 범위의 균등 분포로 초기화하고자 합니다. torch.rand를 사용하는 것이 좋습니다.
"""
# Please do not change / reset the random seed
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):
  # TODO: Implement this function that will create the node embedding matrix.
  # A torch.nn.Embedding layer will be returned. You do not need to change 
  # the values of num_node and embedding_dim. The weight matrix of returned 
  # layer should be initialized under uniform distribution. 

  emb = None

  ############# Your code here ############
  
  # 34 x 16의 임베딩 행렬을 만들고
  emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
  # 0~1 사이의 랜덤한 값으로 요소를 초기화 한다
  emb.weight.data = torch.rand(emb.weight.data.shape)
  
  #########################################

  return emb

emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))

# 임베딩을 시각화
def visualize_emb(emb):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()

"""
Visualize the initial node embeddings

임베딩 행렬을 이해하는 좋은 방법 중 하나는 2D 공간에서 시각화하는 것입니다.
여기에서는 임베딩 시각화 기능을 구현했습니다.
먼저 PCA를 수행하여 임베딩의 차원을 2D 공간으로 축소합니다.
그런 다음 각 점을 시각화하여 해당 점이 속한 커뮤니티에 따라 색상을 지정합니다.
"""
visualize_emb(emb)

"""
질문 7: 임베딩 훈련! 얻을 수 있는 최고의 성능은 무엇입니까?
최고의 손실과 정확도를 모두 성적표에 보고해 주세요. (20점)

에지를 양수 또는 음수로 분류하는 작업을 위해 임베딩을 최적화하려고 합니다.
에지와 각 노드의 임베딩이 주어지면,
임베딩의 내적에 시그모이드가 뒤따르면 해당 에지가 양수(시그모이드의 출력 0.5 이상)이거나 음수(시그모이드의 출력 0.5 미만)일 가능성을 알 수 있어야 합니다.
이전 질문에서 작성한 함수와 이전 셀에서 초기화된 변수를 사용하고 있다는 점에 유의하세요.
문제가 발생하면 1~6번 문제의 정답이 맞는지 확인하세요.
"""
from torch.optim import SGD
import torch.nn as nn

def accuracy(pred, label):
  # TODO: Implement the accuracy function. This function takes the 
  # pred tensor (the resulting tensor after sigmoid) and the label 
  # tensor (torch.LongTensor). Predicted value greater than 0.5 will 
  # be classified as label 1. Else it will be classified as label 0.
  # The returned accuracy should be rounded to 4 decimal places. 
  # For example, accuracy 0.82956 will be rounded to 0.8296.

  accu = 0.0

  ############# Your code here ############
  pred_label = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
  correct = (pred_label == label).sum().item()
  total = label.size(0)
  accu = round(correct / total, 4)
  #########################################

  return accu

def train(emb, loss_fn, sigmoid, train_label, train_edge):
  # TODO: Train the embedding layer here. You can also change epochs and 
  # learning rate. In general, you need to implement: 
  # (1) Get the embeddings of the nodes in train_edge
  # (2) Dot product the embeddings between each node pair
  # (3) Feed the dot product result into sigmoid
  # (4) Feed the sigmoid output into the loss_fn
  # (5) Print both loss and accuracy of each epoch 
  # (6) Update the embeddings using the loss and optimizer 
  # (as a sanity check, the loss should decrease during training)

  epochs = 500
  learning_rate = 0.1

  optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

  for i in range(epochs):

    ############# Your code here ############
    emb.train()
    optimizer.zero_grad()

    # 노드 임베딩
    node_embeddings = emb(train_edge)
 
    # # Dot product the embeddings between each node pair
    dot_products = torch.sum(node_embeddings[::] * node_embeddings[:,[1,0,3,2],:], dim=2)

    # Feed the dot product result into sigmoid
    pred = sigmoid(dot_products)

    # Feed the sigmoid output into the loss_fn
    loss = loss_fn(pred, train_label.repeat(len(node_embeddings), 1))

    # Print both loss and accuracy of each epoch 
    accu = accuracy(pred, train_label)
    print(f"Epoch {i}: Loss={loss.item()}, Accuracy={accu}")

    # Update the embeddings using the loss and optimizer 
    loss.backward()
    optimizer.step()
    #########################################
    
loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
# 가라데 그래프에는 방향성이 없기 때문에,
# 예를 들어 (3,1)이 순방향 positive 엣지이면, 역방향 엣지인 (1,3)도 positive 엣지다
# 따라서 (3,1)을 임베딩할 때 (1,3)도 임베딩해야 한다.
# 각 임베딩에 맞는 정답 레이블이 있어야 하기 때문에 정답 레이블의 형식은 [1,1]이 된다
# 마찬가지 이유로 오답 레이블의 형식도 [0,0]이 된다
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
# 학습 데이터 형식이 [positive 엣지, negative 엣지] 이고
# 이를 임베딩 하면 [순방향 positive 엣지 임베딩, 역방향 positive 엣지 임베딩, 순방향 negative 엣지 임베딩, 역방향 negative 엣지 임베딩] 형식이 된다
# 따라서 레이블 형식도 [1,1,0,0] 이어야 한다
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
# 학습 데이터 형식은 [positive 엣지, negative 엣지] 이다
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)

"""
Visualize the final node embeddings

여기에서 최종 임베딩을 시각화하세요! 이 그림을 이전 임베딩 그림과 시각적으로 비교할 수 있습니다.
교육 후에는 두 클래스가 더 명확하게 구분되는 것을 관찰해야 합니다.
이는 구현에 대한 훌륭한 위생 점검이기도 합니다.
"""
# Visualize the final learned embedding
visualize_emb(emb)

"""
Submission

학점을 받으려면 성적 범위에서 답안을 제출해야 합니다.
"""