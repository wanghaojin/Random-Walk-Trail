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

def randomwalk(X,A,S,num_walks):
    N = X.shape[0]
    for row in range(N):
        S[row] /= np.sum(S[row])
    for v in range(N):
        current = v
        for walk in range(num_walks):
            pass