import numpy as np


def euclidean(A, B):
    return np.linalg.norm(B - A, axis=1)

# TODO: add more distance
def Manhattan(A, B):
    return np.sum(np.abs(B - A))

def Chebyshev(A, B):
    return np.max(np.abs(B - A))

def Cosine(A, B):
    dot_product = np.sum(A * B)
    norm1 = np.sqrt(np.sum(A ** 2))
    norm2 = np.sqrt(np.sum(B ** 2))
    return 1 - (dot_product / (norm1 * norm2))
