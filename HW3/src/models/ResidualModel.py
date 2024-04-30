import numpy as np

from src.models.base import MLForecastModel
from numpy.lib.stride_tricks import sliding_window_view
from src.utils.decomposition import STL_decomposition, X11_decomposition, differential_decomposition, moving_average

from src.models.DLinear import DLinear
from src.models.TsfKNN import TsfKNN
from src.models.ARIMA import ARIMA
from src.models.ThetaMethod import ThetaMethod

class ResidualModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        # self.model1 = DLinear(args)
        self.model2 = TsfKNN(args)
        # self.model3 = ARIMA(args)
        self.model4 = ThetaMethod(args)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        if args.decomposition == 'STL_decomposition':
            self.decomposition = STL_decomposition
        elif args.decomposition == 'X11_decomposition':
            self.decomposition = X11_decomposition
        elif args.decomposition == 'differential_decomposition':
            self.decomposition = differential_decomposition
        elif args.decomposition == 'moving_average':
            self.decomposition = moving_average
        else:
            self.decomposition = None

    def _fit(self, X: np.ndarray) -> None:
        # print(X.shape)
        # self.model1.fit(X)
        self.model2.fit(X)
        # self.model3.fit(X)
        self.model4.fit(X)


    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # x1_pred = self.model1._forecast(X, pred_len)
        x2_pred = self.model2._forecast(X, pred_len)
        # x3_pred = self.model3._forecast(X, pred_len)
        x4_pred = self.model4._forecast(X, pred_len)

        ensemble_pred = (x2_pred + x4_pred) / 2
        return ensemble_pred

