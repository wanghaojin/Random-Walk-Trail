import numpy as np
import random


def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity


def getsimilarity(X, A, self_loop=False):
    N = X.shape[0]
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 or A[j, i] == 1:
                S[i, j] = cosine_similarity(X[i], X[j])
                S[j, i] = S[i, j]
    if self_loop:
        np.fill_diagonal(S, 0)
    return S


def randomwalk(X, A, S, num_walks, walk_length):
    N = X.shape[0]
    edge_strength = np.zeros_like(A, dtype=np.float64)

    for v in range(N):
        for _ in range(num_walks):
            current_node = v
            for _ in range(walk_length):
                neighbors = np.nonzero(A[current_node])[0]
                if len(neighbors) == 0:
                    break
                next_node = np.random.choice(neighbors, p=S[current_node][neighbors]/S[current_node][neighbors].sum())
                edge_strength[current_node][next_node] += 1
                current_node = next_node

    return edge_strength / num_walks


# X = np.random.rand(10, 5)  # Feature matrix
# A = np.random.randint(0, 2, (10, 10))  # Adjacency matrix
# S = getsimilarity(X, A, self_loop=True)
# num_walks = 100
# walk_length = 10
#
# edge_strength = randomwalk(X, A, S, num_walks, walk_length)
# print(edge_strength)
