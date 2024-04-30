import numpy as np
import math
class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# TODO: add other transforms
"""
self.data: np.ndarray, shape=(n_samples, timesteps, channels), where the last channel is the target -> data.shape = 3
self.data: np.ndarray, shape=(n_samples, timesteps) -> data.shape = 2
"""
class Normalization(Transform):
    def __init__(self, args):
        self.maxdata = []
        self.mindata = []
        self.min_aug = []

    def transform(self, data):
        self.maxdata = []
        self.mindata = []
        self.min_aug = []
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                self.mindata.append(np.min(data[0, :, i]))
                self.maxdata.append(np.max(data[0, :, i]))
                self.min_aug.append(np.ones(data[0, :, i].shape) * self.mindata[-1])
                data[0, :, i] = (data[0, :, i] - self.min_aug[-1]) / (self.maxdata[-1] - self.mindata[-1])
        else:
            for sample in range(data.shape[0]):
                self.mindata.append(np.min(data[sample]))
                self.maxdata.append(np.max(data[sample]))
                self.min_aug.append(np.ones(data[sample].shape) * self.mindata[-1])
                data[sample] = (data[sample] - self.min_aug[-1]) / (self.maxdata[-1] - self.mindata[-1])

        return data

    def inverse_transform(self, data):
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                data[0, :, i] = data[0, :, i] * (self.maxdata[i] - self.mindata[i]) + self.min_aug[i]
        else:
            for sample in range(data.shape[0]):
                data[sample] = data[sample] * (self.maxdata[sample] - self.mindata[sample]) + self.min_aug[
                    sample][:len(data[sample])]

        return data


class Standardization(Transform):
    def __init__(self, args):
        self.mu = []
        self.sigma = []
        self.aug = []

    def transform(self, data):
        self.mu = []
        self.sigma = []
        self.aug = []
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                self.mu.append(np.mean(data[0, :, i]))
                self.sigma.append(np.var(data[0, :, i]))
                self.aug.append(np.ones(data[0, :, i].shape) * self.mu[-1])
                data[0, :, i] = (data[0, :, i] - self.aug[-1]) / (self.sigma[-1])
        else:
            for sample in range(data.shape[0]):
                self.mu.append(np.mean(data[sample]))
                self.sigma.append(np.var(data[sample]))
                self.aug.append(np.ones(data[sample].shape) * self.mu[-1])
                data[sample] = (data[sample] - self.aug[-1]) / (self.sigma[-1])

        return data

    def inverse_transform(self, data):
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                data[0, :, i] = data[0, :, i] * (self.sigma[i]) + self.aug[i]
        else:
            for sample in range(data.shape[0]):
                data[sample] = data[sample] * (self.sigma[sample]) + self.aug[
                    sample][:len(data[sample])]

        return data


class MeanNormalization(Transform):
    def __init__(self, args):
        self.mindata = []
        self.maxdata = []
        self.mu_aug = []

    def transform(self, data):
        self.mindata = []
        self.maxdata = []
        self.mu_aug = []
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                self.mindata.append(np.min(data[0, :, i]))
                self.maxdata.append(np.max(data[0, :, i]))
                self.mu_aug.append(np.ones(data[0, :, i].shape) * np.mean(data[0, :, i]))
                data[0, :, i] = (data[0, :, i] - self.mu_aug[-1]) / (self.maxdata[-1] - self.mindata[-1])
        else:
            for sample in range(data.shape[0]):
                self.mindata.append(np.min(data[sample]))
                self.maxdata.append(np.max(data[sample]))
                self.mu_aug.append(np.ones(data[sample].shape) * np.mean(data[sample]))
                data[sample] = (data[sample] - self.mu_aug[-1]) / (self.maxdata[-1] - self.mindata[-1])

        return data

    def inverse_transform(self, data):
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                data[0, :, i] = data[0, :, i] * (self.maxdata[i] - self.mindata[i]) + self.mu_aug[i]
        else:
            for sample in range(data.shape[0]):
                data[sample] = data[sample] * (self.maxdata[sample] - self.mindata[sample]) + self.mu_aug[
                    sample][:len(data[sample])]

        return data


class BoxCox(Transform):
    def __init__(self, args):
        self.lamda = args.lamda
        self.case = None

    def transform(self, data):
        self.case = None
        if len(data.shape) == 3:
            self.case = np.zeros(data.shape)
            for i in range(data.shape[2]):
                for timestep in range(data.shape[1]):
                    if self.lamda != 0. and data[0, timestep, i] >= 0:
                        data[0, timestep, i] = (np.power(data[0, timestep, i] + 1, self.lamda) - 1) / self.lamda
                        self.case[0, timestep, i] = 1
                    if self.lamda == 0. and data[0, timestep, i] >= 0:
                        data[0, timestep, i] = np.log(data[0, timestep, i] + 1)
                        self.case[0, timestep, i] = 2
                    if self.lamda != 2. and data[0, timestep, i] < 0:
                        data[0, timestep, i] = -(np.power((-data[0, timestep, i] + 1), 2 - self.lamda) - 1) / (2 - self.lamda)
                        self.case[0, timestep, i] = 3
                    if self.lamda == 2. and data[0, timestep, i] < 0:
                        data[0, timestep, i] = -np.log(-data[0, timestep, i] + 1)
                        self.case[0, timestep, i] = 4
        else:
            self.case = []
            for sample in range(data.shape[0]):
                self.case.append([])
                for timestep in range(len(data[sample])):
                    if self.lamda != 0. and data[sample][timestep] >= 0:
                        data[sample][timestep] = (np.power(data[sample][timestep] + 1, self.lamda) - 1) / self.lamda
                        self.case[-1].append(1)
                    if self.lamda == 0. and data[sample][timestep] >= 0:
                        data[sample][timestep] = np.log(data[sample][timestep] + 1)
                        self.case[-1].append(2)
                    if self.lamda != 2. and data[sample][timestep] < 0:
                        data[sample][timestep] = -(np.power((-data[sample][timestep] + 1), 2 - self.lamda) - 1) / (
                                    2 - self.lamda)
                        self.case[-1].append(3)
                    if self.lamda == 2. and data[sample][timestep] < 0:
                        data[sample][timestep] = -np.log(-data[sample][timestep] + 1)
                        self.case[-1].append(4)

        return data

    def inverse_transform(self, data):
        if len(data.shape) == 3:
            for i in range(data.shape[2]):
                for timestep in range(data.shape[1]):
                    if self.case[0, timestep, i] == 1:
                        data[0, timestep, i] = np.power(data[0, timestep, i] * self.lamda + 1, 1 / self.lamda) - 1
                    if self.case[0, timestep, i] == 2:
                        data[0, timestep, i] = np.exp(data[0, timestep, i]) - 1
                    if self.case[0, timestep, i] == 3:
                        data[0, timestep, i] = -np.power(((-data[0, timestep, i]) * (2 - self.lamda) + 1),
                                                      1 / (2 - self.lamda)) + 1
                    if self.case[0, timestep, i] == 4:
                        data[0, timestep, i] = -np.exp(-data[0, timestep, i]) + 1
        else:
            for sample in range(data.shape[0]):
                for timestep in range(len(data[sample])):
                    if self.case[sample][timestep] == 1:
                        data[sample][timestep] = np.power(data[sample][timestep] * self.lamda + 1, 1 / self.lamda) - 1
                    if self.case[sample][timestep] == 2:
                        data[sample][timestep] = np.exp(data[sample][timestep]) - 1
                    if self.case[sample][timestep] == 3:
                        data[sample][timestep] = -np.power(((-data[sample][timestep]) * (2 - self.lamda) + 1),
                                                       1 / (2 - self.lamda)) + 1
                    if self.case[sample][timestep] == 4:
                        data[sample][timestep] = -np.exp(-data[sample][timestep]) + 1
        return data