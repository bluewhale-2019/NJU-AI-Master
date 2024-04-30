import numpy as np

from src.models.base import MLForecastModel
from statsmodels.tsa.forecasting.theta import ThetaModel


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.theta = args.theta

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for cidx in range(X.shape[2]):
            for i, batch in enumerate(X):
                inp = X[i, :, cidx]
                Theta_Model = ThetaModel(inp, method='additive', period=12)
                result = Theta_Model.fit()
                forecast = result.forecast(steps=pred_len, theta=self.theta)
                pred[i, :, cidx] = np.array(forecast)
        return pred