import numpy as np

from src.models.base import MLForecastModel

import itertools
import statsmodels.api as sm


class ARIMA(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        p = range(2)
        d = range(2)
        q = range(2)
        seasonal_period = args.seasonal_period
        self.pdq = list(itertools.product(p, d, q))
        self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_period)
                             for x in list(itertools.product(p, d, q))]
        self.param = []

    def _fit(self, X: np.ndarray) -> None:
        # data shape: (n_samples, timesteps, channels)
        for cidx in range(X.shape[2]):
            aic = []
            para = []
            for param in self.pdq:
                for param_seasonal in self.seasonal_pdq:
                    ARIMA_model = sm.tsa.ARIMA(X[0, :, cidx], order=param, seasonal_order=param_seasonal)
                    try:
                        result = ARIMA_model.fit()
                        aic.append(result.aic)
                        para.append([param, param_seasonal])
                    except:
                        pass

            lowest_AIC = np.argmin(aic)
            self.param.append(para[lowest_AIC])

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred = np.zeros((X.shape[0], pred_len, X.shape[2]))
        for cidx in range(X.shape[2]):
            param = self.param[cidx]
            for i, batch in enumerate(X):
                input = X[i, :, cidx]
                ARIMA_model = sm.tsa.ARIMA(input, order=param[0], seasonal_order=param[1])
                try:
                    result = ARIMA_model.fit()
                except:
                    ARIMA_model = sm.tsa.ARIMA(input)
                    result = ARIMA_model.fit()
                pred_res = result.get_forecast(steps=pred_len)
                pred_residual = pred_res.prediction_results.predicted_signal[0]
                pred[i, :, cidx] = pred_residual
        return pred
