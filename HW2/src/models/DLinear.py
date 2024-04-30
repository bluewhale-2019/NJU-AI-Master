import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from src.models.base import MLForecastModel
from src.utils.decomposition import moving_average, differential_decomposition
from numpy.lib.stride_tricks import sliding_window_view

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = Model(args).to(device)

    def _fit(self, X: np.ndarray) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        criterion = nn.MSELoss()

        n_epochs = 50
        batch_size = 64
        n_steps = len(X) / batch_size

        seq_len = self.model.seq_len
        pred_len = self.model.pred_len
        channels = self.model.channels
        X_s = sliding_window_view(X[0, :, :], (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        X_s = torch.tensor(X_s).to(device)

        for i in range(n_epochs):
            count = 0
            total_loss = []

            self.model.train()
            i = 0
            for idx in range(0, len(X_s), batch_size):
                count += 1
                i += 1
                optimizer.zero_grad()
                batch_data = X_s[idx: min(len(X_s), idx + batch_size)]
                batch_x = batch_data[:, :seq_len, :]
                batch_y = batch_data[:, seq_len:, :]

                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y.to(torch.float))
                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()

            total_loss = np.average(total_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                i, n_steps, total_loss))

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return self.model(X).cpu().detach().numpy()


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf

    Decomposition-Linear
    """

    def __init__(self, args):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        # self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.individual = args.individual
        self.channels = args.enc_in
        if args.decomposition == 'Moving_Average':
            self.decomposition = moving_average
        elif args.decomposition == 'Differential_Decomposition':
            self.decomposition = differential_decomposition


        # TODO: implement the following layers
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # TODO: implement the forward pass
        # x: [Batch, Input length, Channel]
        x = torch.tensor(x).to(device)
        seasonal_init, trend_init = self.decomposition(x.cpu().detach().numpy())
        seasonal_init = torch.tensor(seasonal_init, dtype=torch.float).to(device)
        trend_init = torch.tensor(trend_init, dtype=torch.float).to(device)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]







