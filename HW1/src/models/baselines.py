import numpy as np
from src.models.base import MLForecastModel

from numpy.lib.stride_tricks import sliding_window_view

class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=-1).reshape(X.shape[0], 1)
        return np.repeat(mean, pred_len, axis=1)


# TODO: add other models based on MLForecastModel
class AutoRegressiveLinearForecast(MLForecastModel):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.omega = None


    def _fit(self, X: np.ndarray) -> None:
        test_data = X[:, :, -1]
        subseries = np.concatenate(([sliding_window_view(v, self.seq_len + self.pred_len) for v in test_data]))
        test_X = subseries[:, :self.seq_len]
        test_Y = subseries[:, self.seq_len:]

        test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)

        #计算最优的权重参数
        self.omega = np.matmul(np.matmul(np.linalg.pinv(np.matmul(test_X.T, test_X)), test_X.T), test_Y)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        X = np.concatenate((np.ones( (X.shape[0], 1) ), X), axis=1)
        return np.matmul(X, self.omega)


# EMA描述的是一个递归的过程
class ExponentialSmoothingForecast(MLForecastModel):
    def __init__(self, args):
        super().__init__()
        self.alpha = 0.5

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred = np.zeros((len(X), pred_len))
        for i in range(len(X)):
            pass_tsa = 0
            for sample in X[i]:
                pass_tsa = self.alpha * pass_tsa + sample
            for j in range(pred_len):
                pred[i, j] = (1 - self.alpha) * pass_tsa
                pass_tsa = self.alpha * pass_tsa + pred[i, j]
        return pred
