import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.utils.distance import euclidean, manhattan, chebyshev, DTW

from scipy.fft import fft

def lag_m_embedding(x, tau, m):
    # the length of x_embedding equals m
    if len(x.shape) == 2:
        n = x.shape[0]
        x_embedding = []
        for i in range(1,m+1):
            assert n>=(m-i)*tau
            x_embedding.append(x[(n-(m-i)*tau)-1,:].tolist())
            # x_embedding = np.squeeze(x_embedding, axis=0)
    elif len(x.shape) == 3:
        n = x.shape[1]
        x_embedding = []
        for i in range(1, m + 1):
            assert n >= (m - i) * tau
            x_embedding.append(x[:, (n - (m - i) * tau) - 1, :].tolist())
            # x_embedding = np.squeeze(x_embedding, axis=0)
    return x_embedding

def fourier_embedding(x):
    # data shape: (n_samples, timesteps, channels)
    if len(x.shape) == 2:
        timesteps, channels = x.shape
        fft_data = np.zeros((timesteps, channels))
        for i in range(channels):
            fft_data[:, i] = np.abs(fft(x[:, i]))
    elif len(x.shape) == 3:
        n_samples, timesteps, channels = x.shape
        fft_data = np.zeros((n_samples, timesteps, channels))
        for i in range(n_samples):
            for j in range(channels):
                fft_data[i, :, j] = np.abs(fft(x[i, :, j]))

        fft_data = fft_data / np.max(fft_data)
    else:
        raise ValueError

    return fft_data

class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        elif args.distance == 'chebyshev':
            self.distance = chebyshev
        else:
            raise ValueError
        self.msas = args.msas

        self.embedding = args.embedding
        self.tau = 32
        self.m = 3
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == 'MIMO':
            if self.embedding == 'lag_m_embedding':
                x_embedding = lag_m_embedding(x, self.tau, self.m)
                x_embedding = np.array(x_embedding)
                #print('x_embedding is:' ,x_embedding, 'x_embedding shape:', x_embedding.shape) # (6, 7)
                X_s_embedding = lag_m_embedding(X_s, self.tau, self.m)
                X_s_embedding = np.array(X_s_embedding)
                X_s_embedding = np.transpose(X_s_embedding, (1, 0, 2))
                #print('X_s_embedding is:', X_s_embedding, 'X_s_embedding shape:', X_s_embedding.shape) # (12833, 6, 7)
            elif self.embedding == 'fourier_embedding':
                x_embedding = fourier_embedding(x)
                x_embedding = np.array(x_embedding)
                # print('x_embedding is:' ,x_embedding, 'x_embedding shape:', x_embedding.shape) # (96, 7)
                X_s_embedding = fourier_embedding(X_s[:, :seq_len, :])
                # X_s_embedding = np.transpose(X_s_embedding, (1, 0, 2))
                # print('X_s_embedding is:', X_s_embedding, 'X_s_embedding shape:', X_s_embedding.shape) # (12833, 96, 7)
            else:
                raise ValueError

            # distances = self.distance(x, X_s[:, :seq_len, :])
            distances = self.distance(x_embedding, X_s_embedding)
            # print('x is:', x, 'x.shape is:', x.shape)# (96 7)
            # print('X_s[:, :seq_len, :] is:', X_s[:, :seq_len, :], 'X_s[:, :seq_len, :].shape is:', X_s[:, :seq_len, :].shape) # (12833 96 7)
            # print('x_s is:', X_s, 'x_s.shape is:', X_s.shape) # (12833 128 7)
            # print('distances is:', distances, 'distances.shape is:', distances.shape) #(12833,)
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore


    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        for i in range(X.shape[0]):
            x = X[i, :, :]
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
