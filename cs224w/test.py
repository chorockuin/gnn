import numpy as np

def permutation_invariance_equivariance():
    # |V|=4, 1->4, 2->3, 3->2, 3->4, 4->1, 4->3
    A = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,0]]) # 4x4

    # |V|=4, m=2
    X = np.array([[0,1],[2,3],[4,5],[6,7]]) # 4x2

    # 순서를 거꾸로 바꾸는 순열 변환 행렬
    P = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]) # 4x4

    def invariance_f(A, X):
        return np.ones(X.shape[0]).dot(X) # f(A,X) = 1_t dot X

    def equivariance_f(A, X): 
        return A.dot(X) # f(A,X) = A dot X

    # invariance 검증(f(A,X)=f(PAPt,PX))
    print(invariance_f(A,X))
    print(invariance_f(P.dot(A.dot(P.T)), P.dot(X)))

    # equivarinace 검증(Pf(A,X)=f(PAPt,PX))
    print(P.dot(equivariance_f(A,X)))
    print(equivariance_f(P.dot(A.dot(P.T)), P.dot(X)))

def label_propagation():
    A = np.array([[0,1,1,1,0,0,0,0,0],
                [1,0,1,0,0,0,0,0,0],
                [1,1,0,1,0,0,0,0,0],
                [1,0,1,0,1,1,0,0,0],
                [0,0,0,1,0,1,1,1,0],
                [0,0,0,1,1,0,1,1,0],
                [0,0,0,0,1,1,0,1,1],
                [0,0,0,0,1,1,1,0,0],
                [0,0,0,0,0,0,1,0,0]])

    Y = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5])

    P = A.dot(Y)/A.dot([1]*len(A))
    for i, _ in enumerate(Y):
        if Y[i] != 0.0 and Y[i] != 1.0:
            Y[i] = P[i]
    print(Y)

    P = A.dot(Y)/A.dot([1]*len(A))
    for i, _ in enumerate(Y):
        if Y[i] != 0.0 and Y[i] != 1.0:
            Y[i] = P[i]
    print(Y)
    
def diffusion_matrix():
    # 인접 행렬 A
    A = np.array([[1,1,1,1,0,0,0,0,0],
                  [1,1,1,0,0,0,0,0,0],
                  [1,1,1,1,0,0,0,0,0],
                  [1,0,1,1,1,1,0,0,0],
                  [0,0,0,1,1,1,1,1,0],
                  [0,0,0,1,1,1,1,1,0],
                  [0,0,0,0,1,1,1,1,1],
                  [0,0,0,0,1,1,1,1,0],
                  [0,0,0,0,0,0,1,0,1]])
    print(A)
    
    A_dot_1 = A.dot([1]*len(A))
    print(A_dot_1)
    
    # Node Degree 대각 행렬 D
    D = np.diag([3,2,3,4,4,4,4,3,1])
    print(D)
    
    D_dot_1 = D.dot([1]*len(D))
    print(D_dot_1)
    
    # 대각 행렬 D의 각요소에 제곱근+역수 연산을 적용
    D_square_root_reciprocal = np.vectorize(lambda x: 1/np.sqrt(x) if x > 0 else 0)(D)
    print(np.vectorize(lambda x: round(x,2))(D_square_root_reciprocal))
    
    # Node Degree에 기반한 diffusion 행렬
    A_diffusion = (D_square_root_reciprocal.dot(A)).dot(D_square_root_reciprocal)
    print(np.vectorize(lambda x: round(x,2))(A_diffusion))

permutation_invariance_equivariance()
label_propagation()
diffusion_matrix()