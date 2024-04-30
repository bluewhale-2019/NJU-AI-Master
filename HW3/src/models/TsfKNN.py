import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.utils.decomposition import STL_decomposition, X11_decomposition, differential_decomposition, moving_average
from src.models.base import MLForecastModel
from src.utils.distance import euclidean


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        self.msas = args.msas

        if args.decomposition == 'moving_average':
            self.decomposition = moving_average
        elif args.decomposition == 'differential_decomposition':
            self.decomposition = differential_decomposition
        elif args.decomposition == 'STL_decomposition':
            self.decomposition = STL_decomposition
        elif args.decomposition == 'X11_decomposition':
            self.decomposition = X11_decomposition
        else:
            self.decomposition = None

        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.decomposition == None:
            distances = self.distance(x, X_s[:, :seq_len, :])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        else:
            if self.decomposition == STL_decomposition or self.decomposition == X11_decomposition:
                X_s_trend, X_s_seasonal, X_s_residual = self.X_s_trend, self.X_s_seasonal, self.X_s_residual
                x_trend, x_seasonal, x_residual = self.decomposition(np.expand_dims(x, 0))
                x_trend = x_trend[0]
                x_seasonal = x_seasonal[0]
                x_residual = x_residual[0]
            else:
                X_s_trend, X_s_seasonal = self.X_s_trend, self.X_s_seasonal
                x_trend, x_seasonal = self.decomposition(np.expand_dims(x, 0))
                x_trend = x_trend[0].cpu().detach().numpy()
                x_seasonal = x_seasonal[0].cpu().detach().numpy()

            # print(type(x_trend))
            trend_distances = self.distance(x_trend, X_s_trend[:, :seq_len, :])
            indices_of_smallest_k = np.argsort(trend_distances)[:self.k]
            neighbor_trend = X_s_trend[indices_of_smallest_k, seq_len:, :]
            if 'torch' in str(type(neighbor_trend)):
                neighbor_trend = neighbor_trend.cpu().detach().numpy()
            trend_fore = np.mean(neighbor_trend, axis=0, keepdims=True)

            seasonal_distances = self.distance(x_seasonal, X_s_seasonal[:, :seq_len, :])
            indices_of_smallest_k = np.argsort(seasonal_distances)[:self.k]
            neighbor_seasonal = X_s_seasonal[indices_of_smallest_k, seq_len:, :]
            if 'torch' in str(type(neighbor_seasonal)):
                neighbor_seasonal = neighbor_seasonal.cpu().detach().numpy()
            seasonal_fore = np.mean(neighbor_seasonal, axis=0, keepdims=True)

            if self.decomposition == STL_decomposition or self.decomposition == X11_decomposition:
                residual_distances = self.distance(x_residual, X_s_residual[:, :seq_len, :])
                indices_of_smallest_k = np.argsort(residual_distances)[:self.k]
                neighbor_residual = X_s_residual[indices_of_smallest_k, seq_len:, :]
                residual_fore = np.mean(neighbor_residual, axis=0, keepdims=True)
                return trend_fore + seasonal_fore + residual_fore

            else:
                return trend_fore + seasonal_fore

    def forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        batch_size, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        if self.decomposition == STL_decomposition:
            self.X_s_trend, self.X_s_seasonal, self.X_s_residual = self.decomposition(X_s)
        elif self.decomposition == X11_decomposition:
            self.X_s_trend, self.X_s_seasonal, self.X_s_residual = self.decomposition(X_s)
        elif self.decomposition == moving_average:
            self.X_s_trend, self.X_s_seasonal = self.decomposition(X_s)
        elif self.decomposition == differential_decomposition:
            self.X_s_trend, self.X_s_seasonal = self.decomposition(X_s)
        else:
            pass
        for i in range(X.shape[0]):
            x = X[i, :, :]
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore